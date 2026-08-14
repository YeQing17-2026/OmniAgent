"""Memory storage layer for OmniAgent.

Memory system for OmniAgent:
- SQLite storage with FTS5 full-text search
- sqlite-vec BLOB vector storage (Issue #11)
- KNN native query support (Issue #12)
- Embedding cache with incremental computation (Issue #13)
- ShortTermRecallStore for promotion tracking (Issue #14)
- Session isolation (Issue #24)
- File discovery from MEMORY.md + memory/*.md
- Hash-based change detection for incremental sync
"""

import hashlib
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from omniagent.infra import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# sqlite-vec availability check (#11)
# ---------------------------------------------------------------------------

_sqlite_vec_available: Optional[bool] = None


def check_sqlite_vec() -> bool:
    """Check if sqlite-vec extension is available."""
    global _sqlite_vec_available
    if _sqlite_vec_available is not None:
        return _sqlite_vec_available
    try:
        import sqlite_vec  # noqa: F401
        _sqlite_vec_available = True
    except ImportError:
        _sqlite_vec_available = False
        logger.warning(
            "sqlite_vec_unavailable",
            hint="Install with: pip install sqlite-vec",
        )
    return _sqlite_vec_available


def load_sqlite_vec(conn) -> bool:
    """Load sqlite-vec extension into a connection. Returns True on success."""
    if not check_sqlite_vec():
        return False
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception as e:
        logger.warning("sqlite_vec_load_failed", error=str(e))
        return False


# ---------------------------------------------------------------------------
# Embedding BLOB helpers (#11)
# ---------------------------------------------------------------------------

def embedding_to_blob(embedding: List[float]) -> bytes:
    """Convert float32 embedding list to BLOB bytes."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def blob_to_embedding(blob: bytes) -> List[float]:
    """Convert BLOB bytes back to float32 list."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# Markdown chunking defaults
DEFAULT_CHUNK_TOKENS = 400
DEFAULT_CHUNK_OVERLAP = 80


class MemoryChunk:
    """A chunk of memory content."""

    def __init__(
        self,
        text: str,
        path: str,
        source: str,
        start_line: int,
        end_line: int,
        chunk_hash: str,
    ):
        self.text = text
        self.path = path
        self.source = source
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_hash = chunk_hash


def chunk_markdown(
    content: str,
    tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[MemoryChunk]:
    """Split markdown content into overlapping chunks.

   ts chunkMarkdown():
    - maxChars = tokens * 4 (conservative byte estimate)
    - overlapChars = overlap * 4
    - Splits by lines, carries over overlap to next chunk
    - SHA-256 hash per chunk for change detection
    """
    max_chars = tokens * 4
    overlap_chars = overlap * 4

    lines = content.split("\n")
    chunks = []
    current_lines = []
    current_start = 1
    carry_over_lines = []

    for i, line in enumerate(lines):
        current_lines.append(line)

        current_text = "\n".join(carry_over_lines + current_lines)
        if len(current_text) >= max_chars:
            # Flush chunk
            chunk_text = "\n".join(carry_over_lines + current_lines)
            chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

            chunks.append(MemoryChunk(
                text=chunk_text,
                path="",  # Set by caller
                source="memory",
                start_line=current_start,
                end_line=current_start + len(current_lines) - 1,
                chunk_hash=chunk_hash,
            ))

            # Carry over overlap lines
            carry_lines_for_next = []
            carry_chars = 0
            for line in reversed(current_lines):
                if carry_chars + len(line) > overlap_chars:
                    break
                carry_lines_for_next.insert(0, line)
                carry_chars += len(line)

            carry_over_lines = carry_lines_for_next
            current_start = current_start + len(current_lines) - len(carry_over_lines)
            current_lines = []
        elif line.strip() == "" and len("\n".join(carry_over_lines + current_lines)) >= max_chars // 2:
            # Flush at paragraph boundary if we have enough content
            chunk_text = "\n".join(carry_over_lines + current_lines)
            chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

            chunks.append(MemoryChunk(
                text=chunk_text,
                path="",
                source="memory",
                start_line=current_start,
                end_line=current_start + len(current_lines) - 1,
                chunk_hash=chunk_hash,
            ))

            carry_lines_for_next = []
            carry_chars = 0
            for line in reversed(current_lines):
                if carry_chars + len(line) > overlap_chars:
                    break
                carry_lines_for_next.insert(0, line)
                carry_chars += len(line)

            carry_over_lines = carry_lines_for_next
            current_start = current_start + len(current_lines) - len(carry_over_lines)
            current_lines = []
        else:
            # Continue accumulating
            pass

    # Flush remaining content
    if current_lines:
        chunk_text = "\n".join(carry_over_lines + current_lines)
        if chunk_text.strip():
            chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
            chunks.append(MemoryChunk(
                text=chunk_text,
                path="",
                source="memory",
                start_line=current_start,
                end_line=current_start + len(current_lines) - 1,
                chunk_hash=chunk_hash,
            ))

    return chunks


def list_memory_files(workspace_dir: Path, extra_paths: Optional[List[str]] = None) -> List[Path]:
    """Discover memory files from workspace.

    reflexion.pymemory manager listMemoryFiles():
    1. <work_dir>/MEMORY.md
    2. <work_dir>/memory.md
    3. <work_dir>/memory/**/*.md (recursive)
    4. Extra paths from config
    """
    seen = set()
    files = []

    # Root-level MEMORY.md files
    for name in ["MEMORY.md", "memory.md"]:
        p = workspace_dir / name
        if p.is_file():
            real = str(p.resolve())
            if real not in seen:
                seen.add(real)
                files.append(p)

    # Recursive memory/ directory
    memory_dir = workspace_dir / "memory"
    if memory_dir.is_dir():
        for md_file in sorted(memory_dir.glob("**/*.md")):
            if md_file.is_file():
                real = str(md_file.resolve())
                if real not in seen:
                    seen.add(real)
                    files.append(md_file)

    # Extra paths
    if extra_paths:
        for extra in extra_paths:
            p = Path(extra)
            if p.is_file() and p.suffix == ".md":
                real = str(p.resolve())
                if real not in seen:
                    seen.add(real)
                    files.append(p)
            elif p.is_dir():
                for md_file in sorted(p.glob("**/*.md")):
                    if md_file.is_file():
                        real = str(md_file.resolve())
                        if real not in seen:
                            seen.add(real)
                            files.append(md_file)

    return files


def file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file content for change detection."""
    try:
        content = path.read_text(encoding="utf-8")
        return hashlib.sha256(content.encode()).hexdigest()
    except Exception:
        return ""


def generate_chunk_id(
    source: str,
    path: str,
    start_line: int,
    end_line: int,
    chunk_hash: str,
    model: str,
) -> str:
    """Generate unique chunk ID.

    hash(f"{source}:{path}:{startLine}:{endLine}:{chunkHash}:{model}")
    """
    raw = f"{source}:{path}:{start_line}:{end_line}:{chunk_hash}:{model}"
    return hashlib.sha256(raw.encode()).hexdigest()


# SQLite schema
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'memory',
    hash TEXT NOT NULL,
    mtime INTEGER NOT NULL,
    size INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'memory',
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    hash TEXT NOT NULL,
    model TEXT NOT NULL,
    text TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    session_id TEXT,
    embedding TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);

-- chunks_vec: sqlite-vec BLOB vector table (#11)
CREATE TABLE IF NOT EXISTS chunks_vec (
    id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);

-- embedding_cache: content-hash based dedup (#13)
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

-- short_term_recall: tracks memory search hits for promotion (#14)
CREATE TABLE IF NOT EXISTS short_term_recall (
    key TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'memory',
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    snippet TEXT NOT NULL,
    recall_count INTEGER DEFAULT 0,
    total_score REAL DEFAULT 0.0,
    max_score REAL DEFAULT 0.0,
    unique_query_count INTEGER DEFAULT 0,
    first_recalled_at TEXT,
    last_recalled_at TEXT,
    query_hashes TEXT,
    recall_days TEXT,
    concept_tags TEXT,
    claim_hash TEXT,
    promoted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_recall_path ON short_term_recall(path);
CREATE INDEX IF NOT EXISTS idx_recall_count ON short_term_recall(recall_count DESC);
"""

# ---------------------------------------------------------------------------
# ShortTermRecallEntry dataclass (#14)
# ---------------------------------------------------------------------------

@dataclass
class ShortTermRecallEntry:
    """Tracks a memory chunk's recall history for promotion scoring."""

    key: str
    path: str
    source: str
    start_line: int
    end_line: int
    snippet: str
    recall_count: int = 0
    total_score: float = 0.0
    max_score: float = 0.0
    unique_query_count: int = 0
    first_recalled_at: Optional[str] = None
    last_recalled_at: Optional[str] = None
    query_hashes: List[str] = field(default_factory=list)
    recall_days: List[str] = field(default_factory=list)
    concept_tags: List[str] = field(default_factory=list)
    claim_hash: Optional[str] = None
    promoted_at: Optional[str] = None


class MemoryStore:
    """SQLite-backed memory storage.

    Provides the low-level storage operations for the memory system.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("memory_store_initialized", db_path=str(db_path))

    def _init_db(self) -> None:
        """Initialize database schema with migrations."""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript(_SCHEMA_SQL)

            # FTS5 virtual table
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        text,
                        id UNINDEXED,
                        path UNINDEXED,
                        source UNINDEXED,
                        model UNINDEXED
                    )
                """)
            except sqlite3.OperationalError as e:
                logger.warning("fts5_creation_failed", error=str(e))

            # Migration: add embedding column if missing
            try:
                conn.execute("SELECT embedding FROM chunks LIMIT 0")
            except sqlite3.OperationalError:
                logger.info("migration_adding_embedding_column")
                conn.execute("ALTER TABLE chunks ADD COLUMN embedding TEXT")

            # Migration: add session_id column if missing
            try:
                conn.execute("SELECT session_id FROM chunks LIMIT 0")
            except sqlite3.OperationalError:
                logger.info("migration_adding_session_id_column")
                conn.execute("ALTER TABLE chunks ADD COLUMN session_id TEXT")

            # Migrate TEXT embeddings to BLOB chunks_vec (#11)
            self._migrate_embeddings_to_blob(conn)

            conn.commit()
        finally:
            conn.close()

    def _migrate_embeddings_to_blob(self, conn) -> None:
        """Migrate TEXT-encoded embeddings from chunks to chunks_vec BLOB table."""
        import json as _json
        try:
            rows = conn.execute(
                "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()
            migrated = 0
            for row in rows:
                chunk_id = row[0]
                emb_str = row[1]
                # Skip if already migrated
                existing = conn.execute(
                    "SELECT id FROM chunks_vec WHERE id = ?", (chunk_id,)
                ).fetchone()
                if existing:
                    continue
                try:
                    vec = _json.loads(emb_str) if isinstance(emb_str, str) else emb_str
                    blob = embedding_to_blob(vec)
                    conn.execute(
                        "INSERT OR REPLACE INTO chunks_vec (id, embedding) VALUES (?, ?)",
                        (chunk_id, blob),
                    )
                    migrated += 1
                except Exception:
                    pass
            if migrated > 0:
                logger.info("embedding_migration_complete", migrated=migrated)
        except Exception as e:
            logger.warning("embedding_migration_failed", error=str(e))

    def _get_conn(self):
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def get_meta(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None
        finally:
            conn.close()

    def set_meta(self, key: str, value: str) -> None:
        """Set a metadata value."""
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_file(self, path: str, source: str, fhash: str, mtime: int, size: int) -> None:
        """Insert or update a file record."""
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO files (path, source, hash, mtime, size)
                   VALUES (?, ?, ?, ?, ?)""",
                (path, source, fhash, mtime, size),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_file(self, path: str) -> None:
        """Delete a file record and all its chunks."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
            conn.execute("DELETE FROM files WHERE path = ?", (path,))
            conn.commit()
        finally:
            conn.close()

    def get_all_files(self) -> List[dict]:
        """Get all file records."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM files").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_file(self, path: str) -> Optional[dict]:
        """Get a file record by path."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def upsert_chunk(
        self,
        chunk_id: str,
        path: str,
        source: str,
        start_line: int,
        end_line: int,
        chunk_hash: str,
        model: str,
        text: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Insert or update a chunk."""
        conn = self._get_conn()
        now = int(time.time())
        try:
            conn.execute(
                """INSERT OR REPLACE INTO chunks
                   (id, path, source, start_line, end_line, hash, model, text, updated_at, session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, path, source, start_line, end_line, chunk_hash, model, text, now, session_id),
            )
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO chunks_fts (text, id, path, source, model)
                       VALUES (?, ?, ?, ?, ?)""",
                    (text, chunk_id, path, source, model),
                )
            except Exception as e:
                logger.warning("fts5_insert_failed", error=str(e))
            conn.commit()
        finally:
            conn.close()

    def get_all_chunks(self) -> List[dict]:
        """Get all chunks."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM chunks").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def search_fts(self, query: str, limit: int = 6) -> List[dict]:
        """Full-text search using FTS5.

       ts searchKeyword().
        Returns results with bm25_rank (negative) and normalized score.
        """
        conn = self._get_conn()
        try:
            # JOIN chunks_fts with chunks to get full columns.
            # FTS5 only stores indexed text + UNINDEXED aux columns,
            # so start_line/end_line/text must come from the chunks table.
            rows = conn.execute(
                """SELECT ch.id, ch.path, ch.start_line, ch.end_line,
                          ch.text, ch.source, ch.model,
                          bm25(chunks_fts) AS rank
                   FROM chunks_fts
                   JOIN chunks ch ON chunks_fts.id = ch.id
                   WHERE chunks_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                # Normalize BM25 rank to [0, 1] score
                rank = d.pop("rank", -1)
                d["score"] = 1.0 / (1.0 + abs(rank))
                results.append(d)
            return results
        except Exception as e:
            logger.warning("fts_search_failed", error=str(e), query=query)
            return []
        finally:
            conn.close()

    def get_all_chunks_for_vector(self, session_id: Optional[str] = None) -> List[dict]:
        """Get all chunks that have BLOB embeddings in chunks_vec (#11).

        Falls back to TEXT-encoded embeddings in chunks if chunks_vec is empty.
        """
        conn = self._get_conn()
        try:
            # Prefer chunks_vec BLOB
            rows = conn.execute(
                """SELECT ch.id, ch.path, ch.start_line, ch.end_line,
                          ch.text, ch.source, ch.model, cv.embedding AS embedding_blob
                   FROM chunks_vec cv
                   JOIN chunks ch ON cv.id = ch.id"""
            ).fetchall()
            if rows:
                result = []
                for row in rows:
                    d = dict(row)
                    blob = d.pop("embedding_blob", None)
                    d["embedding"] = blob_to_embedding(blob) if blob else None
                    if session_id is None or d.get("session_id") == session_id:
                        result.append(d)
                return result

            # Fallback: TEXT embeddings
            import json as _json
            rows = conn.execute(
                """SELECT id, path, start_line, end_line, text, source, model, embedding, session_id
                   FROM chunks
                   WHERE embedding IS NOT NULL"""
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                if session_id is not None and d.get("session_id") != session_id:
                    continue
                emb = d.get("embedding")
                if isinstance(emb, str):
                    try:
                        d["embedding"] = _json.loads(emb)
                    except Exception:
                        d["embedding"] = None
                result.append(d)
            return result
        finally:
            conn.close()

    def save_embedding(self, chunk_id: str, embedding: List[float]) -> None:
        """Save embedding as BLOB in chunks_vec (#11) and legacy TEXT in chunks."""
        blob = embedding_to_blob(embedding)
        conn = self._get_conn()
        try:
            # Primary: BLOB storage
            conn.execute(
                "INSERT OR REPLACE INTO chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk_id, blob),
            )
            # Keep TEXT column for backward compat (cheap)
            import json as _json
            conn.execute(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                (_json.dumps(embedding), chunk_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_embedding_blob(self, chunk_id: str) -> Optional[bytes]:
        """Get raw BLOB embedding for a chunk (#11)."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT embedding FROM chunks_vec WHERE id = ?", (chunk_id,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Embedding cache (#13)
    # ------------------------------------------------------------------

    def get_cached_embedding(self, content_hash: str) -> Optional[List[float]]:
        """Get cached embedding by content hash."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT embedding FROM embedding_cache WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if row and row[0]:
                return blob_to_embedding(row[0])
            return None
        finally:
            conn.close()

    def set_cached_embedding(
        self, content_hash: str, embedding: List[float], model: str
    ) -> None:
        """Cache embedding by content hash (#13)."""
        blob = embedding_to_blob(embedding)
        now = int(time.time())
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO embedding_cache
                   (content_hash, embedding, model, created_at)
                   VALUES (?, ?, ?, ?)""",
                (content_hash, blob, model, now),
            )
            conn.commit()
        finally:
            conn.close()

    def content_hash(self, text: str) -> str:
        """Compute content hash for embedding cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    # ------------------------------------------------------------------
    # KNN search (#12) — native sqlite-vec vec_distance_l2/cosine
    # ------------------------------------------------------------------

    def search_knn(
        self,
        query_embedding: List[float],
        limit: int = 10,
        session_id: Optional[str] = None,
    ) -> List[dict]:
        """KNN search using sqlite-vec if available, else fallback to cosine scan (#12)."""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        vec_loaded = load_sqlite_vec(conn)

        try:
            if vec_loaded:
                # Native KNN via sqlite-vec
                query_blob = embedding_to_blob(query_embedding)
                sql = """
                    SELECT cv.id, ch.path, ch.start_line, ch.end_line,
                           ch.text, ch.source, ch.model,
                           vec_distance_cosine(cv.embedding, ?) AS distance
                    FROM chunks_vec cv
                    JOIN chunks ch ON cv.id = ch.id
                    ORDER BY distance
                    LIMIT ?
                """
                rows = conn.execute(sql, (query_blob, limit)).fetchall()
                results = []
                for row in rows:
                    d = dict(row)
                    if session_id is not None and d.get("session_id") != session_id:
                        continue
                    distance = d.pop("distance", 1.0)
                    d["score"] = max(0.0, 1.0 - distance)
                    results.append(d)
                return results
            else:
                # Pure Python cosine scan fallback
                return self._cosine_scan(query_embedding, limit, session_id)
        finally:
            conn.close()

    def _cosine_scan(
        self,
        query_embedding: List[float],
        limit: int,
        session_id: Optional[str],
    ) -> List[dict]:
        """O(n) cosine similarity scan fallback when sqlite-vec is unavailable."""
        import math

        all_chunks = self.get_all_chunks_for_vector(session_id=session_id)
        if not all_chunks or not query_embedding:
            return []

        q_norm = math.sqrt(sum(x * x for x in query_embedding)) or 1.0
        scored = []
        for chunk in all_chunks:
            vec = chunk.get("embedding")
            if not vec:
                continue
            dot = sum(a * b for a, b in zip(query_embedding, vec))
            v_norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            sim = dot / (q_norm * v_norm)
            scored.append({**chunk, "score": sim})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Short-term recall tracking (#14)
    # ------------------------------------------------------------------

    def upsert_recall(self, entry: "ShortTermRecallEntry") -> None:
        """Insert or update a short-term recall entry."""
        import json as _json
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO short_term_recall
                   (key, path, source, start_line, end_line, snippet,
                    recall_count, total_score, max_score, unique_query_count,
                    first_recalled_at, last_recalled_at,
                    query_hashes, recall_days, concept_tags, claim_hash, promoted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.key, entry.path, entry.source,
                    entry.start_line, entry.end_line, entry.snippet,
                    entry.recall_count, entry.total_score, entry.max_score,
                    entry.unique_query_count,
                    entry.first_recalled_at, entry.last_recalled_at,
                    _json.dumps(entry.query_hashes),
                    _json.dumps(entry.recall_days),
                    _json.dumps(entry.concept_tags),
                    entry.claim_hash, entry.promoted_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_recall(self, key: str) -> Optional["ShortTermRecallEntry"]:
        """Get a recall entry by key."""
        import json as _json
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM short_term_recall WHERE key = ?", (key,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            return ShortTermRecallEntry(
                key=d["key"],
                path=d["path"],
                source=d.get("source", "memory"),
                start_line=d["start_line"],
                end_line=d["end_line"],
                snippet=d["snippet"],
                recall_count=d.get("recall_count", 0),
                total_score=d.get("total_score", 0.0),
                max_score=d.get("max_score", 0.0),
                unique_query_count=d.get("unique_query_count", 0),
                first_recalled_at=d.get("first_recalled_at"),
                last_recalled_at=d.get("last_recalled_at"),
                query_hashes=_json.loads(d.get("query_hashes") or "[]"),
                recall_days=_json.loads(d.get("recall_days") or "[]"),
                concept_tags=_json.loads(d.get("concept_tags") or "[]"),
                claim_hash=d.get("claim_hash"),
                promoted_at=d.get("promoted_at"),
            )
        finally:
            conn.close()

    def get_top_recall_candidates(
        self, min_count: int = 3, limit: int = 50
    ) -> List["ShortTermRecallEntry"]:
        """Get top recall candidates ordered by recall_count (#16/#17)."""
        import json as _json
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM short_term_recall
                   WHERE recall_count >= ? AND promoted_at IS NULL
                   ORDER BY recall_count DESC
                   LIMIT ?""",
                (min_count, limit),
            ).fetchall()
            entries = []
            for d in (dict(r) for r in rows):
                entries.append(ShortTermRecallEntry(
                    key=d["key"],
                    path=d["path"],
                    source=d.get("source", "memory"),
                    start_line=d["start_line"],
                    end_line=d["end_line"],
                    snippet=d["snippet"],
                    recall_count=d.get("recall_count", 0),
                    total_score=d.get("total_score", 0.0),
                    max_score=d.get("max_score", 0.0),
                    unique_query_count=d.get("unique_query_count", 0),
                    first_recalled_at=d.get("first_recalled_at"),
                    last_recalled_at=d.get("last_recalled_at"),
                    query_hashes=_json.loads(d.get("query_hashes") or "[]"),
                    recall_days=_json.loads(d.get("recall_days") or "[]"),
                    concept_tags=_json.loads(d.get("concept_tags") or "[]"),
                    claim_hash=d.get("claim_hash"),
                    promoted_at=d.get("promoted_at"),
                ))
            return entries
        finally:
            conn.close()

    def mark_promoted(self, key: str) -> None:
        """Mark a recall entry as promoted to long-term memory."""
        now = datetime.now().isoformat(timespec="seconds")
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE short_term_recall SET promoted_at = ? WHERE key = ?",
                (now, key),
            )
            conn.commit()
        finally:
            conn.close()

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Get a single chunk by ID."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def cleanup_stale_files(self, known_paths: List[str]) -> int:
        """Remove file records for files that no longer exist.

        Returns count of removed records.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT path FROM files").fetchall()
            removed = 0
            for row in rows:
                if row["path"] not in known_paths:
                    self.delete_file(row["path"])
                    removed += 1
            conn.commit()
            return removed
        finally:
            conn.close()
