"""Memory storage layer for OmniAgent.

Memory system for OmniAgent:
- SQLite storage with FTS5 full-text search (memory-schema.ts)
- Markdown chunking
- File discovery from MEMORY.md + memory/*.md (listMemoryFiles)
- Hash-based change detection for incremental sync
"""

import hashlib
import time
from pathlib import Path
from typing import List, Optional, Tuple

from omniagent.infra import get_logger

logger = get_logger(__name__)


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
    embedding TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
"""


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
        """Initialize database schema."""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript(_SCHEMA_SQL)
            # Create FTS5 table
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

            # Migration: add embedding column if missing (existing DBs)
            try:
                conn.execute("SELECT embedding FROM chunks LIMIT 0")
            except sqlite3.OperationalError:
                logger.info("memory_db_migration_adding_embedding_column")
                conn.execute("ALTER TABLE chunks ADD COLUMN embedding TEXT")

            conn.commit()
        finally:
            conn.close()

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
    ) -> None:
        """Insert or update a chunk."""
        conn = self._get_conn()
        now = int(time.time())
        try:
            conn.execute(
                """INSERT OR REPLACE INTO chunks
                   (id, path, source, start_line, end_line, hash, model, text, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, path, source, start_line, end_line, chunk_hash, model, text, now),
            )
            # Also insert into FTS5
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

    def get_all_chunks_for_vector(self) -> List[dict]:
        """Get all chunks with embedding for vector search."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT id, path, start_line, end_line, text, source, model, embedding
                   FROM chunks
                   WHERE embedding IS NOT NULL"""
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def save_embedding(self, chunk_id: str, embedding: str) -> None:
        """Save embedding vector for a chunk."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                (embedding, chunk_id),
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
