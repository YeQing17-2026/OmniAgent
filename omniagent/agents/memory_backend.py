"""Memory backend abstraction layer (Issue #22, #23).

Defines MemoryBackend abstract interface and EmbeddingProvider abstraction,
decoupling storage implementation from search logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChunkData:
    """Memory chunk data structure."""

    id: str
    path: str
    source: str
    start_line: int
    end_line: int
    text: str
    hash: str
    model: str
    updated_at: int
    embedding: Optional[bytes] = None  # BLOB (float32 array)


@dataclass
class SearchResult:
    """Search result from backend."""

    id: str
    path: str
    start_line: int
    end_line: int
    text: str
    source: str
    score: float
    snippet: str


# ---------------------------------------------------------------------------
# MemoryBackend abstract interface (#22)
# ---------------------------------------------------------------------------

class MemoryBackend(ABC):
    """Abstract interface for memory storage backends."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize backend (create tables, indices, etc.)."""

    @abstractmethod
    async def sync(self, workspace_dir: Path) -> int:
        """Sync memory files from workspace.

        Returns:
            Number of files updated.
        """

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 6,
        min_score: float = 0.35,
    ) -> List[SearchResult]:
        """Search memory using hybrid (BM25 + vector) or keyword mode."""

    @abstractmethod
    def search_by_path(self, path: str) -> List[ChunkData]:
        """Get all chunks for a specific file path."""

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[ChunkData]:
        """Get a single chunk by ID."""

    @abstractmethod
    def upsert_chunk(self, chunk: ChunkData) -> None:
        """Insert or update a chunk."""

    @abstractmethod
    def save_embedding(self, chunk_id: str, embedding: bytes) -> None:
        """Save embedding BLOB for a chunk."""

    @abstractmethod
    def get_recent_files(self, since_timestamp: int, limit: int) -> List[Dict[str, Any]]:
        """Get files modified since timestamp."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""


# ---------------------------------------------------------------------------
# EmbeddingProvider abstract interface (#23)
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider name (e.g., 'openai', 'ollama', 'local')."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Returns:
            List of embedding vectors in the same order as input.
        """

    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        results = await self.embed([text])
        return results[0] if results else []

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available."""


# ---------------------------------------------------------------------------
# Concrete EmbeddingProvider implementations (#23)
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding API provider."""

    _DIMENSIONS: Dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        api_url: str = "https://api.openai.com/v1/embeddings",
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self._dim: Optional[int] = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = self._DIMENSIONS.get(self.model, 1536)
        return self._dim

    async def embed(self, texts: List[str]) -> List[List[float]]:
        import aiohttp

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.model, "input": [t[:8000] for t in texts]}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"OpenAI embedding failed ({resp.status}): {text}")
                data = await resp.json()
                return [item["embedding"] for item in data["data"]]

    async def health_check(self) -> bool:
        try:
            result = await self.embed(["health check"])
            return len(result) > 0 and len(result[0]) > 0
        except Exception:
            return False


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama local embedding provider."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dim: Optional[int] = None

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def dimension(self) -> int:
        return self._dim or 768

    async def embed(self, texts: List[str]) -> List[List[float]]:
        import aiohttp

        url = f"{self.base_url}/api/embeddings"
        results = []
        async with aiohttp.ClientSession() as session:
            for text in texts:
                payload = {"model": self.model, "prompt": text}
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"Ollama embedding failed ({resp.status}): {body}")
                    data = await resp.json()
                    vec = data.get("embedding", [])
                    if self._dim is None and vec:
                        self._dim = len(vec)
                    results.append(vec)
        return results

    async def health_check(self) -> bool:
        try:
            result = await self.embed(["health check"])
            return len(result) > 0 and len(result[0]) > 0
        except Exception:
            return False


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers local embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def dimension(self) -> int:
        model = self._get_model()
        return model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        import asyncio

        model = self._get_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: model.encode(texts)
        )
        return [e.tolist() for e in embeddings]

    async def health_check(self) -> bool:
        try:
            result = await self.embed(["health check"])
            return len(result) > 0 and len(result[0]) > 0
        except Exception:
            return False


def create_embedding_provider(
    provider_name: str,
    api_key: str = "",
    api_url: str = "",
    model_id: str = "",
) -> Optional[EmbeddingProvider]:
    """Factory for EmbeddingProvider based on provider name."""
    name = provider_name.lower()

    if name in ("openai", "openai_compatible"):
        url = api_url or "https://api.openai.com/v1/embeddings"
        if "/chat/completions" in url:
            url = url.replace("/chat/completions", "/embeddings")
        elif not url.endswith("/embeddings"):
            url = url.rstrip("/") + "/embeddings"
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model_id or "text-embedding-3-small",
            api_url=url,
        )

    if name == "ollama":
        base = api_url or "http://localhost:11434"
        return OllamaEmbeddingProvider(
            model=model_id or "nomic-embed-text",
            base_url=base,
        )

    if name == "local":
        return LocalEmbeddingProvider(
            model_name=model_id or "all-MiniLM-L6-v2",
        )

    logger.warning("unknown_embedding_provider", provider=provider_name)
    return None
