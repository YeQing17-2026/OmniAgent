"""Similarity helpers for context-evolution rule dedup.

Two layers, both used only when deciding whether two short rule/lesson
sentences mean the same thing:

- ``lexical_similarity``: stdlib only, always available, zero IO.
- ``SemanticEmbedder``: optional *local* embeddings (Ollama -> fastembed),
  resolved lazily and cached. Returns ``None`` when no backend is available so
  callers fall back to lexical. Vectors are computed on the fly and never
  persisted, so this never touches the memory vector store.
"""

from __future__ import annotations

import difflib
import math
import re
from typing import List, Optional

import aiohttp

from omniagent.infra import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
# Small English stopword set; lessons are extracted in English today. Dropping
# these stops them from inflating word overlap (the flaw in the old check).
_STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "is", "are", "be", "in", "on",
    "for", "with", "that", "this", "it", "as", "at", "by", "from", "when",
    "should", "must", "do", "not", "you", "your", "user", "agent", "instead",
    "than", "use", "using", "if", "then", "its",
}


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def lexical_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] from the max of three cheap signals.

    Combines normalized token Jaccard, overlap coefficient, and a character
    SequenceMatcher ratio. Stdlib only — no dependency, no IO.
    """
    if not a or not b:
        return 0.0
    scores = [difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()]
    ta, tb = _tokens(a), _tokens(b)
    if ta and tb:
        inter = len(ta & tb)
        scores.append(inter / len(ta | tb))            # Jaccard
        scores.append(inter / min(len(ta), len(tb)))   # overlap coefficient
    return max(scores)


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SemanticEmbedder:
    """Resolve a local embedding backend lazily: ollama -> fastembed -> none.

    ``embed`` returns vectors, or ``None`` when no backend is usable. Resolution
    is cached for the process. Vectors are ephemeral (never stored).
    """

    def __init__(
        self,
        backend: str = "auto",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "nomic-embed-text",
        fastembed_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.backend = backend
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.fastembed_model = fastembed_model
        self._resolved: Optional[str] = None  # "ollama" | "fastembed" | "none"
        self._fastembed_obj = None

    async def available(self) -> bool:
        return (await self._resolve()) != "none"

    async def _resolve(self) -> str:
        if self._resolved is not None:
            return self._resolved
        if self.backend == "off":
            self._resolved = "none"
            return self._resolved
        if self.backend in ("auto", "ollama") and await self._ollama_alive():
            self._resolved = "ollama"
            logger.info("semantic_backend_resolved", backend="ollama", model=self.ollama_model)
            return self._resolved
        if self.backend in ("auto", "fastembed"):
            try:
                import fastembed  # noqa: F401
                self._resolved = "fastembed"
                logger.info("semantic_backend_resolved", backend="fastembed", model=self.fastembed_model)
                return self._resolved
            except ImportError:
                pass
        self._resolved = "none"
        logger.info("semantic_backend_resolved", backend="none")
        return self._resolved

    async def _ollama_alive(self) -> bool:
        try:
            timeout = aiohttp.ClientTimeout(total=1.5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.ollama_url}/api/tags") as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        backend = await self._resolve()
        if backend == "ollama":
            return await self._embed_ollama(texts)
        if backend == "fastembed":
            return await self._embed_fastembed(texts)
        return None

    async def _embed_ollama(self, texts: List[str]) -> Optional[List[List[float]]]:
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            vectors: List[List[float]] = []
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for text in texts:
                    async with session.post(
                        f"{self.ollama_url}/api/embeddings",
                        json={"model": self.ollama_model, "prompt": text},
                    ) as resp:
                        if resp.status != 200:
                            logger.warning("ollama_embed_failed", status=resp.status)
                            return None
                        data = await resp.json()
                        vectors.append(data.get("embedding") or [])
            return vectors
        except Exception as e:
            logger.warning("ollama_embed_error", error=str(e))
            return None

    async def _embed_fastembed(self, texts: List[str]) -> Optional[List[List[float]]]:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if self._fastembed_obj is None:
                from fastembed import TextEmbedding
                self._fastembed_obj = await loop.run_in_executor(
                    None, lambda: TextEmbedding(model_name=self.fastembed_model)
                )
            vectors = await loop.run_in_executor(
                None, lambda: [list(map(float, v)) for v in self._fastembed_obj.embed(list(texts))]
            )
            return vectors
        except Exception as e:
            logger.warning("fastembed_error", error=str(e))
            return None


async def is_duplicate(
    candidate: str,
    existing: List[str],
    *,
    lex_threshold: float,
    sem_threshold: float,
    embedder: Optional[SemanticEmbedder] = None,
) -> bool:
    """True if ``candidate`` is a near-duplicate of any string in ``existing``.

    Lexical is checked first and short-circuits. Only when lexical is
    inconclusive AND a semantic backend is available do we embed (ephemerally).
    OR semantics: either signal crossing its threshold flags a duplicate.
    """
    if not existing:
        return False
    for rule in existing:
        if lexical_similarity(candidate, rule) >= lex_threshold:
            return True
    if embedder is not None and await embedder.available():
        vectors = await embedder.embed([candidate] + list(existing))
        if vectors and len(vectors) == len(existing) + 1:
            cand_vec, rule_vecs = vectors[0], vectors[1:]
            for rule_vec in rule_vecs:
                if cosine(cand_vec, rule_vec) >= sem_threshold:
                    return True
    return False
