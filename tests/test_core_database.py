"""
Tests for backend.core (Module 0 — foundation).

Run with:
    python -m tests.test_core_database
or:
    pytest tests/test_core_database.py

Covers:
    - schema applies cleanly on a fresh DB
    - meta row is set with the correct schema version
    - all expected tables exist
    - read/write context managers behave correctly
    - the write lock actually serializes concurrent writers
    - cross-module helpers return correct counts on a fresh DB
    - settings round-trip + defaults + unknown-key rejection
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

# Make the project importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core import database as db_mod
from backend.core import paths as paths_mod
from backend.core.database import (
    SCHEMA_VERSION,
    count_frames,
    count_labels,
    get_active_model,
    init_db,
    overall_stats,
    read_conn,
    set_active_model,
    write_conn,
)


EXPECTED_TABLES = {
    "meta",
    "frames",
    "labels",
    "proposals",
    "models",
    "sources",
    "active_queue",
    "world_map_cells",
    "hud_masks",
    "ppo_checkpoints",
}


# ─────────── fixture helpers ───────────

def _fresh_db() -> Path:
    """Make a temp DB file, init it, return its path."""
    tmp = Path(tempfile.mkdtemp(prefix="forzatek_test_")) / "test.db"
    init_db(tmp)
    return tmp


# ─────────── schema ───────────

def test_init_db_creates_file():
    p = _fresh_db()
    assert p.exists(), "init_db should create the database file"
    assert p.stat().st_size > 0, "database file should be non-empty"
    print("✓ init_db creates the database file")


def test_init_db_is_idempotent():
    """Calling init_db twice on the same path must not error or change anything."""
    p = _fresh_db()
    size_before = p.stat().st_size
    init_db(p)
    init_db(p)
    # Size may differ trivially but the file should still be valid.
    with read_conn(p) as c:
        v = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert v["value"] == str(SCHEMA_VERSION)
    print("✓ init_db is idempotent")


def test_all_expected_tables_exist():
    p = _fresh_db()
    with read_conn(p) as c:
        rows = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    found = {r["name"] for r in rows}
    missing = EXPECTED_TABLES - found
    assert not missing, f"missing tables: {missing}"
    print(f"✓ all {len(EXPECTED_TABLES)} expected tables exist")


def test_schema_version_recorded():
    p = _fresh_db()
    with read_conn(p) as c:
        r = c.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
    assert r is not None and r["value"] == str(SCHEMA_VERSION)
    print(f"✓ schema_version recorded as {SCHEMA_VERSION}")


# ─────────── connection context managers ───────────

def test_write_then_read():
    p = _fresh_db()
    with write_conn(p) as c:
        c.execute(
            """INSERT INTO frames
               (ts, source_type, game_version, phash, frame_jpeg, label_status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (time.time(), "live", "fh5", 12345, b"\x00\x01\x02", "unlabeled"),
        )
    with read_conn(p) as c:
        r = c.execute("SELECT * FROM frames").fetchone()
    assert r is not None
    assert r["game_version"] == "fh5"
    assert r["phash"] == 12345
    print("✓ write_conn writes, read_conn reads")


def test_write_conn_rolls_back_on_exception():
    p = _fresh_db()
    try:
        with write_conn(p) as c:
            c.execute(
                """INSERT INTO frames
                   (ts, source_type, game_version, phash, frame_jpeg)
                   VALUES (?, ?, ?, ?, ?)""",
                (time.time(), "live", "fh5", 1, b"x"),
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert count_frames(p) == 0, "rollback should have undone the insert"
    print("✓ write_conn rolls back on exception")


def test_write_lock_serializes_writers():
    """Spawn N threads that each try to write. The lock should serialize them
    such that no `database is locked` error ever escapes, and all writes land."""
    p = _fresh_db()
    n_threads = 8
    n_writes_each = 5
    errors: list[Exception] = []

    def worker(tid: int):
        try:
            for i in range(n_writes_each):
                with write_conn(p) as c:
                    c.execute(
                        """INSERT INTO frames
                           (ts, source_type, game_version, phash, frame_jpeg)
                           VALUES (?, ?, ?, ?, ?)""",
                        (time.time(), "live", "fh5", tid * 1000 + i, b"x"),
                    )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"write contention produced errors: {errors}"
    assert count_frames(p) == n_threads * n_writes_each
    print(f"✓ write lock serializes {n_threads} concurrent writers cleanly")


# ─────────── helpers ───────────

def test_count_frames_with_filters():
    p = _fresh_db()
    with write_conn(p) as c:
        for i, gv in enumerate(["fh4", "fh5", "fh5", "fh5", "fh6"]):
            c.execute(
                """INSERT INTO frames
                   (ts, source_type, game_version, phash, frame_jpeg, label_status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (time.time(), "live", gv, i, b"x", "unlabeled"),
            )
    assert count_frames(p) == 5
    assert count_frames(p, game_version="fh5") == 3
    assert count_frames(p, game_version="fh4") == 1
    assert count_frames(p, game_version="fh5", label_status="unlabeled") == 3
    assert count_frames(p, game_version="fh9") == 0
    print("✓ count_frames respects filters")


def test_count_labels():
    p = _fresh_db()
    now = time.time()
    with write_conn(p) as c:
        # two frames
        for i in range(2):
            c.execute(
                """INSERT INTO frames
                   (ts, source_type, game_version, phash, frame_jpeg)
                   VALUES (?, ?, ?, ?, ?)""",
                (now, "live", "fh5", i, b"x"),
            )
        # frame 1: seg + det labels. frame 2: seg only.
        c.execute(
            """INSERT INTO labels (frame_id, task, data_json, provenance, created_at)
               VALUES (1, 'seg', '{}', 'manual', ?)""", (now,))
        c.execute(
            """INSERT INTO labels (frame_id, task, data_json, provenance, created_at)
               VALUES (1, 'det', '{}', 'manual', ?)""", (now,))
        c.execute(
            """INSERT INTO labels (frame_id, task, data_json, provenance, created_at)
               VALUES (2, 'seg', '{}', 'manual', ?)""", (now,))

    assert count_labels(db_path=p) == 2, "2 distinct frames have labels"
    assert count_labels(task="seg", db_path=p) == 2
    assert count_labels(task="det", db_path=p) == 1
    print("✓ count_labels distinguishes 'all' vs per-task")


def test_active_model_helpers():
    p = _fresh_db()
    assert get_active_model(p) is None, "no models yet"
    with write_conn(p) as c:
        for r in (1, 2):
            c.execute(
                """INSERT INTO models
                   (name, round_num, path, trained_on, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (f"v{r}", r, f"models/v{r}.pt", 100 * r, time.time()),
            )
    set_active_model(2, p)
    am = get_active_model(p)
    assert am is not None and am["round_num"] == 2

    # switching active model: only one row should be active at a time
    set_active_model(1, p)
    am = get_active_model(p)
    assert am["round_num"] == 1
    with read_conn(p) as c:
        n_active = c.execute(
            "SELECT COUNT(*) FROM models WHERE is_active=1"
        ).fetchone()[0]
    assert n_active == 1, "exactly one model should be active at a time"
    print("✓ active model getter/setter is exclusive")


def test_overall_stats_on_empty_db():
    p = _fresh_db()
    s = overall_stats(p)
    assert s["total_frames"] == 0
    assert s["labeled_frames"] == 0
    assert s["proposed_frames"] == 0
    assert s["queue_size"] == 0
    assert s["active_model"] is None
    assert s["frames_by_version"] == {}
    print("✓ overall_stats on empty DB returns sensible zeros")


# ─────────── settings ───────────

def test_settings_defaults_when_no_file(monkeypatch_settings_path):
    from backend import settings as settings_mod
    settings_mod._cache = None
    s = settings_mod.get_settings()
    # Spot-check a few keys
    assert s["capture_fps"] == settings_mod.DEFAULTS["capture_fps"]
    assert s["default_game_version"] == settings_mod.DEFAULTS["default_game_version"]
    assert s["telemetry_port"] == settings_mod.DEFAULTS["telemetry_port"]
    print("✓ settings returns defaults when no settings.json exists")


def test_settings_round_trip(monkeypatch_settings_path):
    from backend import settings as settings_mod
    settings_mod._cache = None
    settings_mod.save_settings({"capture_fps": 30, "device": "cuda"})
    settings_mod._cache = None  # force reload from disk
    s = settings_mod.get_settings()
    assert s["capture_fps"] == 30
    assert s["device"] == "cuda"
    # Untouched keys still come from defaults
    assert s["jpeg_quality"] == settings_mod.DEFAULTS["jpeg_quality"]
    print("✓ settings save → reload round-trips correctly")


def test_settings_rejects_unknown_keys(monkeypatch_settings_path):
    from backend import settings as settings_mod
    settings_mod._cache = None
    settings_mod.save_settings({"capture_fps": 25, "totally_made_up_key": 999})
    s = settings_mod.get_settings()
    assert s["capture_fps"] == 25
    assert "totally_made_up_key" not in s
    raw = json.loads(settings_mod.SETTINGS_PATH.read_text())
    assert "totally_made_up_key" not in raw, "unknown keys should never hit disk"
    print("✓ settings silently drops unknown keys")


def test_settings_get_single_key(monkeypatch_settings_path):
    from backend import settings as settings_mod
    settings_mod._cache = None
    assert settings_mod.get("capture_fps") == settings_mod.DEFAULTS["capture_fps"]
    assert settings_mod.get("nope_not_a_real_key", "fallback") == "fallback"
    print("✓ settings.get(key) works for known and unknown keys")


# ─────────── tiny test runner ───────────

class _Monkeypatch:
    """Just enough monkeypatch to redirect SETTINGS_PATH to a temp dir
    without bringing in pytest's full machinery."""

    def __init__(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="forzatek_settings_"))
        self._original = paths_mod.SETTINGS_PATH

    def apply(self):
        from backend import settings as settings_mod
        settings_mod.SETTINGS_PATH = self._tmp / "settings.json"
        settings_mod._cache = None

    def revert(self):
        from backend import settings as settings_mod
        settings_mod.SETTINGS_PATH = self._original
        settings_mod._cache = None


def _run_with_settings_patch(fn):
    mp = _Monkeypatch()
    mp.apply()
    try:
        fn(monkeypatch_settings_path=None)
    finally:
        mp.revert()


if __name__ == "__main__":
    # No-arg tests
    test_init_db_creates_file()
    test_init_db_is_idempotent()
    test_all_expected_tables_exist()
    test_schema_version_recorded()
    test_write_then_read()
    test_write_conn_rolls_back_on_exception()
    test_write_lock_serializes_writers()
    test_count_frames_with_filters()
    test_count_labels()
    test_active_model_helpers()
    test_overall_stats_on_empty_db()

    # Settings tests need SETTINGS_PATH redirected to a temp file
    _run_with_settings_patch(test_settings_defaults_when_no_file)
    _run_with_settings_patch(test_settings_round_trip)
    _run_with_settings_patch(test_settings_rejects_unknown_keys)
    _run_with_settings_patch(test_settings_get_single_key)

    print("\nAll Module 0 tests passed.")