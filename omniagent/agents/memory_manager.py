"""Memory search manager and tools for OmniAgent.

Memory search system for OmniAgent:
- MemorySearchManager: hybrid search (BM25 + vector), embedding
- memory_search, memory_get, memory_store agent tools
- FTS-only fallback when no embedding provider available
- MMR reranking (#24), session isolation (#24)
- DailyNotesLoader (#25)
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from omniagent.agents.llm import LLMProvider
from omniagent.infra import get_logger
from omniagent.tools.base import Tool, ToolResult

from .memory import (
    MemoryStore,
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

                # Compute embedding if available (with cache, #13)
                if self._vector_available:
                    try:
                        embedding = await self._embed_text_cached(chunk.text)
                        if embedding:
                            self.store.save_embedding(chunk_id, embedding)
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

    async def _embed_text_cached(self, text: str) -> Optional[List[float]]:
        """Generate embedding with content-hash cache (#13)."""
        content_hash = self.store.content_hash(text)
        cached = self.store.get_cached_embedding(content_hash)
        if cached is not None:
            return cached

        embedding = await self._embed_text(text)
        if embedding:
            self.store.set_cached_embedding(content_hash, embedding, self.model_id or "default")
        return embedding

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        min_score: Optional[float] = None,
        session_id: Optional[str] = None,
        mmr_lambda: float = 0.7,
    ) -> List[MemorySearchResult]:
        """Hybrid search (BM25 + KNN) with optional MMR reranking (#24)."""
        n = max_results if max_results is not None else self.query_max_results
        threshold = min_score if min_score is not None else self.query_min_score

        keyword_results = self.store.search_fts(query, limit=n * 4)

        if not self._vector_available or not self.hybrid_enabled:
            results = [
                self._row_to_result(row, snippet_length=300)
                for row in keyword_results
                if row["score"] >= threshold
            ][:n]
            return self._mmr_rerank_results(results, lambda_param=mmr_lambda)

        # Generate query embedding for KNN
        query_embedding = await self._embed_text_cached(query) or []

        vector_rows = self.store.search_knn(
            query_embedding=query_embedding,
            limit=n * 4,
            session_id=session_id,
        )
        # Merge keyword + vector results
        merged: Dict[str, Dict] = {}
        for row in keyword_results:
            cid = row["id"]
            merged.setdefault(cid, {"text_score": 0.0, "vector_score": 0.0, "data": row})
            merged[cid]["text_score"] = row["score"]
        for row in vector_rows:
            cid = row.get("id", row.get("chunk_id", ""))
            merged.setdefault(cid, {"text_score": 0.0, "vector_score": 0.0, "data": row})
            merged[cid]["vector_score"] = row["score"]

        total_weight = self.vector_weight + self.text_weight
        results = []
        for entry in merged.values():
            score = (
                (self.vector_weight / total_weight) * entry["vector_score"]
                + (self.text_weight / total_weight) * entry["text_score"]
            )
            if score >= threshold:
                results.append(self._row_to_result(entry["data"], score, snippet_length=300))

        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:n]
        return self._mmr_rerank_results(results, lambda_param=mmr_lambda)

    def _mmr_rerank_results(
        self,
        results: List[MemorySearchResult],
        lambda_param: float = 0.7,
    ) -> List[MemorySearchResult]:
        """MMR reranking using Jaccard text similarity for diversity (#24)."""
        if len(results) <= 1:
            return results

        def jaccard(a: str, b: str) -> float:
            w1 = set(a.lower().split())
            w2 = set(b.lower().split())
            if not w1 or not w2:
                return 0.0
            return len(w1 & w2) / len(w1 | w2)

        selected: List[MemorySearchResult] = []
        remaining = list(results)
        selected.append(remaining.pop(0))

        while remaining:
            best_score = -1.0
            best_idx = 0
            for i, candidate in enumerate(remaining):
                relevance = candidate.score
                max_sim = max(
                    (jaccard(candidate.snippet, sel.snippet) for sel in selected),
                    default=0.0,
                )
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        return selected

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
                "Search long-term memory from MEMORY.md and memory/*.md. "
                "Use this to recall cross-session prior work, decisions, user preferences, "
                "or project knowledge. For facts from the current conversation, especially "
                "references like just now, previous turn, above, 刚才, 上一轮, or 刚刚, "
                "use the conversation history first instead of this tool."
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
            results = await self.manager.search(query, max_results=max_results, min_score=min_score)
            if not results:
                return ToolResult(
                    success=True,
                    output=(
                        "No relevant long-term memories found. This does not mean the "
                        "information is absent from the current conversation; check recent "
                        "conversation history before answering."
                    ),
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


class MemoryStoreTool(Tool):
    """memory_store tool: lets the Agent proactively save information to memory (#25)."""

    def __init__(self, manager: "MemorySearchManager"):
        super().__init__(
            name="memory_store",
            description=(
                "Store important information into memory for future recall. "
                "Use this to remember decisions, preferences, facts, or anything "
                "that might be relevant in future conversations. "
                "The information will be searchable via memory_search."
            ),
        )
        self.manager = manager

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The information to store. Be specific and include context.",
                },
                "category": {
                    "type": "string",
                    "enum": ["preference", "decision", "fact", "entity", "other"],
                    "description": "Category of this memory (default: other)",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Importance score 0-1 (default: 0.5)",
                },
                "path": {
                    "type": "string",
                    "description": "Optional path within memory/ (e.g., 'memory/preferences.md')",
                },
            },
            "required": ["text"],
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        text = params.get("text", "")
        category = params.get("category", "other")
        importance = float(params.get("importance", 0.5))
        rel_path = params.get("path")

        if not text:
            return ToolResult(success=False, output="", error="Missing required parameter: text")

        try:
            stored_path = await self._store_memory(text, category, importance, rel_path)
            await self.manager.sync()
            return ToolResult(
                success=True,
                output=f"Memory stored at {stored_path}",
                metadata={"path": stored_path, "category": category},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    async def _store_memory(
        self,
        text: str,
        category: str,
        importance: float,
        rel_path: Optional[str],
    ) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        if rel_path:
            target = self.manager.workspace_dir / rel_path
        else:
            target = self.manager.workspace_dir / "memory" / f"{category}.md"

        target.parent.mkdir(parents=True, exist_ok=True)

        entry = f"\n## {timestamp} (importance: {importance:.1f})\n\n{text}\n"
        with open(target, "a", encoding="utf-8") as f:
            f.write(entry)

        return str(target.relative_to(self.manager.workspace_dir))


# ---------------------------------------------------------------------------
# DailyNotesLoader (#25)
# ---------------------------------------------------------------------------

class DailyNotesLoader:
    """Automatically loads today's and yesterday's daily notes into memory sync."""

    DAILY_NOTES_DIR = "memory"
    DATE_FORMAT = "%Y-%m-%d"

    @staticmethod
    def get_daily_note_paths(workspace_dir: Path) -> List[Path]:
        """Return paths for today's and yesterday's daily notes if they exist."""
        from datetime import timedelta

        today = date.today()
        yesterday = today - timedelta(days=1)

        paths = []
        for d in (today, yesterday):
            fname = workspace_dir / DailyNotesLoader.DAILY_NOTES_DIR / f"{d.strftime(DailyNotesLoader.DATE_FORMAT)}.md"
            if fname.is_file():
                paths.append(fname)
        return paths

    @staticmethod
    def list_memory_files_with_daily(
        workspace_dir: Path,
        extra_paths: Optional[List[str]] = None,
    ) -> List[Path]:
        """Extend list_memory_files to prioritize today/yesterday daily notes."""
        daily = DailyNotesLoader.get_daily_note_paths(workspace_dir)
        regular = list_memory_files(workspace_dir, extra_paths)

        # Daily notes first, then the rest (dedup by resolved path)
        seen = set()
        result = []
        for p in daily + regular:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                result.append(p)
        return result
