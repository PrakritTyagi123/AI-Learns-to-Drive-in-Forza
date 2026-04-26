"""
ForzaTek AI v2 — Database Layer
===============================
Connection management, schema initialization, and the small handful of
helpers every other module reaches for.

Threading model
---------------
SQLite handles concurrent readers fine but serializes writers internally.
We make that explicit with a process-wide `threading.Lock` so writes from
the FastAPI thread, the Eel thread, the telemetry UDP listener, and the
training subprocess never collide and produce `database is locked` errors.

    `read_conn()`  — concurrent, no lock
    `write_conn()` — serialized through `_write_lock`, auto-commits on exit

Schema
------
The schema lives in `schema.sql`, not in this file. `init_db()` reads it
and runs it on first boot. Adding a column = edit one .sql file.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from backend.core.paths import DB_PATH, SCHEMA_PATH, ensure_dirs

SCHEMA_VERSION = 1

# Process-wide write lock. SQLite serializes writers internally, but we want
# the failure mode to be "wait for the lock" rather than "raise OperationalError",
# so we serialize at the application layer too.
_write_lock = threading.Lock()


# ─────────── Initialisation ───────────

def init_db(db_path: Path = DB_PATH) -> None:
    """Create the database and all tables if they don't already exist.

    Safe to call repeatedly — every CREATE statement uses IF NOT EXISTS.
    Called once at app startup from `backend/main.py`.
    """
    ensure_dirs()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema_sql)
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        conn.commit()
    finally:
        conn.close()


# ─────────── Connection context managers ───────────

@contextmanager
def read_conn(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Read-only connection. Concurrent reads are allowed; no lock taken.

    Usage:
        with read_conn() as c:
            row = c.execute("SELECT * FROM frames WHERE id=?", (fid,)).fetchone()
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def write_conn(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Serialized write connection. Only one writer at a time across the process.

    Auto-commits on clean exit, rolls back on exception.

    Usage:
        with write_conn() as c:
            c.execute("INSERT INTO frames (...) VALUES (...)", (...))
    """
    with _write_lock:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ─────────── Convenience helpers used across modules ───────────

def count_frames(db_path: Path = DB_PATH, **filters) -> int:
    """Total frames, optionally filtered by exact-match column values.

    Example: count_frames(game_version="fh5", label_status="labeled")
    """
    with read_conn(db_path) as c:
        q = "SELECT COUNT(*) FROM frames"
        args: list = []
        if filters:
            clauses = [f"{k} = ?" for k in filters.keys()]
            args.extend(filters.values())
            q += " WHERE " + " AND ".join(clauses)
        return c.execute(q, args).fetchone()[0]


def count_labels(task: Optional[str] = None, db_path: Path = DB_PATH) -> int:
    """Count labels. With `task`, count rows for that task. Without, count
    distinct frames that have at least one label."""
    with read_conn(db_path) as c:
        if task:
            return c.execute(
                "SELECT COUNT(*) FROM labels WHERE task = ?", (task,)
            ).fetchone()[0]
        return c.execute(
            "SELECT COUNT(DISTINCT frame_id) FROM labels"
        ).fetchone()[0]


def get_active_model(db_path: Path = DB_PATH) -> Optional[dict]:
    """Return the active perception model row as a dict, or None if no model
    has been trained yet."""
    with read_conn(db_path) as c:
        r = c.execute(
            "SELECT * FROM models WHERE is_active = 1 LIMIT 1"
        ).fetchone()
        return dict(r) if r else None


def set_active_model(model_id: int, db_path: Path = DB_PATH) -> None:
    """Mark exactly one model as active. Atomic — clears all is_active flags
    then sets the one we want."""
    with write_conn(db_path) as c:
        c.execute("UPDATE models SET is_active = 0")
        c.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))


def overall_stats(db_path: Path = DB_PATH) -> dict:
    """One-shot dashboard summary. Used by Module 1's dashboard page."""
    with read_conn(db_path) as c:
        total    = c.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        labeled  = c.execute(
            "SELECT COUNT(DISTINCT frame_id) FROM labels WHERE task IN ('seg','det')"
        ).fetchone()[0]
        proposed = c.execute(
            "SELECT COUNT(DISTINCT frame_id) FROM proposals"
        ).fetchone()[0]
        queue    = c.execute("SELECT COUNT(*) FROM active_queue").fetchone()[0]

        by_version: dict[str, int] = {}
        for row in c.execute(
            "SELECT game_version, COUNT(*) FROM frames GROUP BY game_version"
        ):
            by_version[row[0]] = row[1]

        active = c.execute(
            "SELECT * FROM models WHERE is_active = 1 LIMIT 1"
        ).fetchone()

        return {
            "total_frames":     total,
            "labeled_frames":   labeled,
            "proposed_frames":  proposed,
            "queue_size":       queue,
            "frames_by_version": by_version,
            "active_model":     dict(active) if active else None,
        }


# ─────────── CLI smoke test ───────────

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
    print(json.dumps(overall_stats(), indent=2, default=str))