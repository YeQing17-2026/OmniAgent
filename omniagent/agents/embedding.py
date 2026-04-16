"""Embedding providers for vector search."""

from abc import ABC, abstractmethod
from typing import List
import aiohttp


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", api_url: str = None):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url or "https://api.openai.com/v1/embeddings"

    async def embed(self, text: str) -> List[float]:
        results = await self.embed_batch([text])
        return results[0] if results else []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "input": texts,
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            async with session.post(self.api_url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Embedding request failed: {error}")
                data = await resp.json()
                return [item["embedding"] for item in data["data"]]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )

    async def embed(self, text: str) -> List[float]:
        self._ensure_model()
        import asyncio
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, lambda: self._model.encode(text))
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        import asyncio
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, lambda: self._model.encode(texts))
        return [e.tolist() for e in embeddings]
