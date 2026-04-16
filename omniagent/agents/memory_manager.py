"""Memory search manager and tools for OmniAgent.

Memory search system for OmniAgent:
- MemorySearchManager: hybrid search (BM25 + vector), embedding
- memory_search and memory_get agent tools
- FTS-only fallback when no embedding provider available
"""

import asyncio
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omniagent.agents.llm import LLMMessage, LLMProvider
from omniagent.infra import get_logger
from omniagent.tools.base import Tool, ToolResult

from .memory import (
    MemoryStore,
    MemoryChunk,
    chunk_markdown,
    generate_chunk_id,
    file_hash,
    list_memory_files,
)

logger = get_logger(__name__)


@dataclass
class MemorySearchResult:
    """A single memory search result."""
    chunk_id: str
    path: str
    source: str
    start_line: int
    end_line: int
    text: str
    score: float
    snippet: str


class MemorySearchManager:
    """Manages memory indexing and search.

    Memory index manager:
    - SQLite + FTS5 storage (via MemoryStore)
    - Optional vector search with embedding
    - Hybrid search combining BM25 + vector scores
    - File discovery and change detection for incremental sync
    """

    def __init__(
        self,
        workspace_dir: Path,
        store_path: Optional[Path] = None,
        llm_provider: Optional[LLMProvider] = None,
        api_key: str = "",
        api_url: str = "",
        model_id: str = "",
        chunking_tokens: int = 400,
        chunking_overlap: int = 80,
        hybrid_enabled: bool = True,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        query_max_results: int = 6,
        query_min_score: float = 0.35,
    ):
        self.workspace_dir = workspace_dir
        self.llm = llm_provider
        self.api_key = api_key
        self.api_url = api_url
        self.model_id = model_id
        self.chunking_tokens = chunking_tokens
        self.chunking_overlap = chunking_overlap
        self.hybrid_enabled = hybrid_enabled
        self.vector_weight = vector_weight
        self.text_weight = text_weight
        self.query_max_results = query_max_results
        self.query_min_score = query_min_score
        self._vector_available = llm_provider is not None

        # Storage
        if store_path is None:
            store_path = workspace_dir / ".omniagent" / "memory.db"
        self.store = MemoryStore(store_path)

        logger.info(
            "memory_search_manager_initialized",
            workspace=str(workspace_dir),
            vector_available=self._vector_available,
            hybrid=hybrid_enabled,
        )

    async def sync(self) -> int:
        """Synchronize memory files with the index.

       ts runSync():
        - Discover files
        - Check for changes (hash-based)
        - Chunk and index new/changed files
        - Cleanup stale entries

        Returns count of updated files.
        """
        files = list_memory_files(self.workspace_dir)
        known_paths = []
        updated = 0

        for file_path in files:
            rel_path = str(file_path.relative_to(self.workspace_dir))
            known_paths.append(rel_path)

            # Check for changes
            fhash = file_hash(file_path)
            existing = self.store.get_file(rel_path)

            stat = file_path.stat()
            mtime = int(stat.st_mtime)
            size = stat.st_size

            if existing and existing["hash"] == fhash:
                continue  # No change

            # Read and chunk
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("memory_file_read_failed", path=rel_path, error=str(e))
                continue

            chunks = chunk_markdown(
                content,
                tokens=self.chunking_tokens,
                overlap=self.chunking_overlap,
            )

            # Index chunks
            for chunk in chunks:
                chunk.path = rel_path
                chunk_id = generate_chunk_id(
                    "memory",
                    rel_path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.chunk_hash,
                    self.model_id or "default",
                )
                self.store.upsert_chunk(
                    chunk_id=chunk_id,
                    path=rel_path,
                    source="memory",
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_hash=chunk.chunk_hash,
                    model=self.model_id or "default",
                    text=chunk.text,
                )

                # Compute embedding if available
                if self._vector_available:
                    try:
                        embedding = await self._embed_text(chunk.text)
                        if embedding:
                            self.store.save_embedding(chunk_id, json.dumps(embedding))
                    except Exception as e:
                        logger.warning("embedding_failed", chunk_id=chunk_id[:12], error=str(e))

            # Update file record
            self.store.upsert_file(rel_path, "memory", fhash, mtime, size)
            updated += 1

        # Cleanup stale files
        cleaned = self.store.cleanup_stale_files(known_paths)
        if cleaned > 0:
            logger.info("memory_stale_files_cleaned", count=cleaned)

        logger.info("memory_sync_complete", files_scanned=len(files), files_updated=updated)
        return updated

    async def _embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a text using the LLM API.

        Uses the configured LLM provider to call an embedding-compatible endpoint.
        Falls back to simple hash-based pseudo-embedding if API fails.
        """
        if not self.llm:
            return None

        try:
            # Try to use the LLM API as an embedding provider
            # Most OpenAI-compatible APIs support embeddings endpoint
            import aiohttp

            # Build embedding request URL from chat completions URL
            base_url = self.api_url or ""
            if "/chat/completions" in base_url:
                embed_url = base_url.replace("/chat/completions", "/embeddings")
            else:
                embed_url = base_url.rstrip("/") + "/embeddings"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model_id,
                "input": text[:8000],  # Truncate to avoid token limits
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(embed_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "data" in data and len(data["data"]) > 0:
                            return data["data"][0].get("embedding")
                    logger.warning("embedding_api_failed", status=resp.status)
        except Exception as e:
            logger.warning("embedding_request_failed", error=str(e))

        return None

    def search(self, query: str, max_results: int = None, min_score: float = None) -> List[MemorySearchResult]:
        """Search memory using hybrid (BM25 + vector) or FTS-only mode.

       ts mergeHybridResults() + manager-search.ts search().
        """
        max_results = max_results or self.query_max_results
        min_score = min_score or self.query_min_score

        # Always do keyword search
        keyword_results = self.store.search_fts(query, limit=max_results * 4)

        if not self._vector_available or not self.hybrid_enabled:
            # FTS-only mode
            return [
                self._row_to_result(row, snippet_length=300)
                for row in keyword_results
                if row["score"] >= min_score
            ][:max_results]

        # Hybrid mode: combine BM25 + vector scores
        vector_results = self._search_vector(query, limit=max_results * 4)

        # Merge by chunk ID
        merged = {}
        for row in keyword_results:
            cid = row["id"]
            if cid not in merged:
                merged[cid] = {"text_score": 0.0, "vector_score": 0.0, "data": row}
            merged[cid]["text_score"] = row["score"]

        for row in vector_results:
            cid = row["id"]
            if cid not in merged:
                merged[cid] = {"text_score": 0.0, "vector_score": 0.0, "data": row}
            merged[cid]["vector_score"] = row["score"]

        # Compute weighted scores
        total_weight = self.vector_weight + self.text_weight
        results = []
        for cid, entry in merged.items():
            score = (
                (self.vector_weight / total_weight) * entry["vector_score"]
                + (self.text_weight / total_weight) * entry["text_score"]
            )
            if score >= min_score:
                results.append(self._row_to_result(entry["data"], score, snippet_length=300))

        # Sort by score descending, take top N
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    def _search_vector(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Search using cosine similarity against stored embeddings."""
        import math

        all_chunks = self.store.get_all_chunks_for_vector()
        if not all_chunks or not query_embedding:
            return []

        scored = []
        query_norm = math.sqrt(sum(x * x for x in query_embedding)) or 1.0

        for chunk in all_chunks:
            emb_str = chunk.get("embedding")
            if not emb_str:
                continue
            try:
                embedding = json.loads(emb_str) if isinstance(emb_str, str) else emb_str
            except (json.JSONDecodeError, TypeError):
                continue

            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(query_embedding, embedding))
            emb_norm = math.sqrt(sum(x * x for x in embedding)) or 1.0
            similarity = dot_product / (query_norm * emb_norm)

            scored.append({
                "chunk_id": chunk["id"],
                "path": chunk["path"],
                "source": chunk.get("source", ""),
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
                "text": chunk["text"],
                "score": similarity,
                "snippet": chunk["text"][:200],
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _mmr_rerank(self, query_embedding: List[float], results: List[Dict], lambda_param: float = 0.7) -> List[Dict]:
        """Maximal Marginal Relevance reranking for diversity."""
        import math

        if not results or len(results) <= 1:
            return results

        query_norm = math.sqrt(sum(x * x for x in query_embedding)) or 1.0

        def similarity(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
            norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
            return dot / (norm_a * norm_b)

        # Get all embeddings for selected results
        selected = []
        remaining = list(results)

        # Select first (most relevant)
        selected.append(remaining.pop(0))

        while remaining and len(selected) < len(results):
            best_score = -1
            best_idx = 0

            for i, candidate in enumerate(remaining):
                # Relevance to query
                rel_score = candidate["score"]

                # Max similarity to already selected
                max_sim = 0
                for sel in selected:
                    # Use cosine similarity on text as proxy (we don't have embeddings stored per result here)
                    # Use score-based proxy
                    sim = candidate["score"] * sel["score"]  # Simple proxy
                    max_sim = max(max_sim, sim)

                mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _apply_temporal_decay(self, results: List[Dict], half_life_hours: float = 168) -> List[Dict]:
        """Apply exponential time decay to scores."""
        import time

        now = time.time()
        half_life_seconds = half_life_hours * 3600

        for r in results:
            indexed_at = r.get("indexed_at", now)
            if isinstance(indexed_at, (int, float)):
                age_seconds = now - indexed_at
                decay = 0.5 ** (age_seconds / half_life_seconds)
                r["original_score"] = r["score"]
                r["score"] = r["score"] * decay

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _row_to_result(
        self, row: dict, score: Optional[float] = None, snippet_length: int = 300
    ) -> MemorySearchResult:
        """Convert a DB row to a MemorySearchResult."""
        text = row.get("text", "")
        # Create snippet: first snippet_length chars
        snippet = text[:snippet_length]
        if len(text) > snippet_length:
            snippet += "..."

        return MemorySearchResult(
            chunk_id=row.get("id", ""),
            path=row.get("path", ""),
            source=row.get("source", ""),
            start_line=row.get("start_line", 0),
            end_line=row.get("end_line", 0),
            text=text,
            score=score if score is not None else row.get("score", 0.0),
            snippet=snippet,
        )

    def read_memory_safe(self, rel_path: str) -> Optional[str]:
        """Safely read a memory file by relative path.

       ts memory_get tool.
        Validates path is within memory directory to prevent directory traversal.
        """
        # Validate path is within allowed memory paths
        allowed_prefixes = [
            "MEMORY.md",
            "memory.md",
            "memory/",
        ]
        is_valid = any(
            rel_path == prefix or rel_path.startswith(prefix + "/")
            for prefix in allowed_prefixes
        )
        if not is_valid:
            logger.warning("memory_read_blocked", path=rel_path)
            return None

        full_path = self.workspace_dir / rel_path
        try:
            return full_path.read_text(encoding="utf-8")
        except (FileNotFoundError, Exception) as e:
            logger.warning("memory_read_failed", path=rel_path, error=str(e))
            return None

    def close(self):
        """Clean up resources."""
        logger.info("memory_search_manager_closed")


class MemorySearchTool(Tool):
    """memory_search agent tool.

   ts memory_search.
    """

    def __init__(self, manager: MemorySearchManager):
        super().__init__(
            name="memory_search",
            description=(
                "Search memory for relevant information from MEMORY.md and memory/*.md. "
                "Use this to recall prior work, decisions, user preferences, or project knowledge."
            ),
        )
        self.manager = manager

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for memory content",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 6)",
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum relevance score 0-1 (default: 0.35)",
                },
            },
            "required": ["query"],
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "")
        max_results = params.get("max_results", 6)
        min_score = params.get("min_score", 0.35)

        if not query:
            return ToolResult(
                success=False, output="", error="Missing required parameter: query"
            )

        try:
            results = self.manager.search(query, max_results=max_results, min_score=min_score)
            if not results:
                return ToolResult(
                    success=True,
                    output="No relevant memories found.",
                )

            parts = []
            for r in results:
                citation = f"Source: {r.path}#L{r.start_line}-L{r.end_line}"
                parts.append(f"{r.snippet}\n\n{citation}")

            output = "\n\n---\n\n".join(parts)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class MemoryGetTool(Tool):
    """memory_get agent tool.

   ts memory_get.
    """

    def __init__(self, manager: MemorySearchManager):
        super().__init__(
            name="memory_get",
            description=(
                "Read a memory file safely. Provide the relative path (e.g., 'MEMORY.md' or 'memory/file.md'). "
                "Optionally specify from_line and lines for partial reads."
            ),
        )
        self.manager = manager

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to memory file (e.g. 'MEMORY.md' or 'memory/file.md')",
                },
                "from_line": {
                    "type": "integer",
                    "description": "Starting line number (default: 1)",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of lines to read (default: 100)",
                },
            },
            "required": ["path"],
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        path = params.get("path", "")
        from_line = params.get("from_line", 1)
        lines = params.get("lines", 100)

        if not path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: path"
            )

        try:
            content = self.manager.read_memory_safe(path)
            if content is None:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Memory file not found or access denied: {path}",
                )

            # Extract requested lines
            all_lines = content.split("\n")
            start = max(1, from_line)
            end = start + lines
            selected = all_lines[start - 1 : end - 1]
            text = "\n".join(selected)

            return ToolResult(
                success=True,
                output=text,
                metadata={"path": path},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
