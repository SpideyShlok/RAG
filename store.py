"""
Persistent, concurrency-safe storage for conversation history and indexing
status. Replaces the old load_history()/save_history() JSON file, which
re-wrote the *entire* file on every message and had no real protection
against two threads writing at once.

SQLite in WAL mode handles concurrent readers fine and serializes writers
safely, which is exactly what a handful of Flask worker threads need here.
Everything is scoped by tenant_id so multiple knowledge bases can share one
database file without colliding.
"""
import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from config import Config

_lock = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_lookup ON conversation_turns(tenant_id, session_id, id);

CREATE TABLE IF NOT EXISTS indexing_status (
    tenant_id TEXT NOT NULL,
    filepath TEXT NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (tenant_id, filepath)
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(Config.APP_DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _lock, _conn() as conn:
        conn.executescript(_SCHEMA)


# ---------------------------------------------------------------- history --

def append_turn(tenant_id: str, session_id: str, role: str, content: str):
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO conversation_turns (tenant_id, session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (tenant_id, session_id, role, content, time.time()),
        )


def get_turns(tenant_id: str, session_id: str, limit_turns: int = None):
    """Returns [{"role": ..., "content": ...}, ...] oldest-first."""
    with _conn() as conn:
        cur = conn.execute(
            "SELECT role, content FROM conversation_turns "
            "WHERE tenant_id = ? AND session_id = ? ORDER BY id DESC "
            + ("LIMIT ?" if limit_turns else ""),
            (tenant_id, session_id, limit_turns * 2) if limit_turns else (tenant_id, session_id),
        )
        rows = cur.fetchall()
    rows.reverse()
    return [{"role": r, "content": c} for r, c in rows]


def clear_session(tenant_id: str, session_id: str = None):
    with _lock, _conn() as conn:
        if session_id:
            conn.execute(
                "DELETE FROM conversation_turns WHERE tenant_id = ? AND session_id = ?",
                (tenant_id, session_id),
            )
        else:
            conn.execute("DELETE FROM conversation_turns WHERE tenant_id = ?", (tenant_id,))


def list_sessions(tenant_id: str):
    with _conn() as conn:
        cur = conn.execute(
            "SELECT DISTINCT session_id FROM conversation_turns WHERE tenant_id = ?", (tenant_id,)
        )
        return [r[0] for r in cur.fetchall()]


# ------------------------------------------------------------ index status --

def set_indexing_status(tenant_id: str, filepath: str, status: str, progress: int = 0, error: str = None):
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO indexing_status (tenant_id, filepath, status, progress, error, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(tenant_id, filepath) DO UPDATE SET "
            "status=excluded.status, progress=excluded.progress, error=excluded.error, updated_at=excluded.updated_at",
            (tenant_id, filepath, status, progress, error, time.time()),
        )


def get_indexing_status(tenant_id: str, filepath: str = None):
    with _conn() as conn:
        if filepath:
            cur = conn.execute(
                "SELECT status, progress, error, updated_at FROM indexing_status "
                "WHERE tenant_id = ? AND filepath = ?",
                (tenant_id, filepath),
            )
            row = cur.fetchone()
            if not row:
                return {"status": "unknown"}
            status, progress, error, updated_at = row
            return {"status": status, "progress": progress, "error": error, "updated_at": updated_at}
        cur = conn.execute(
            "SELECT filepath, status, progress, error, updated_at FROM indexing_status WHERE tenant_id = ?",
            (tenant_id,),
        )
        return {
            fp: {"status": s, "progress": p, "error": e, "updated_at": u}
            for fp, s, p, e, u in cur.fetchall()
        }


def delete_indexing_status(tenant_id: str, filepath: str):
    with _lock, _conn() as conn:
        conn.execute(
            "DELETE FROM indexing_status WHERE tenant_id = ? AND filepath = ?", (tenant_id, filepath)
        )
