"""
ForzaTek AI v2 — Tests for backend/system/stats.py

Run with:  python -m tests.test_system_stats

Like the Module 0 tests, this is a plain script — each function asserts and
prints a tick on success. The runner at the bottom calls them in order and
prints a summary at the end.

We use a fresh temp DB per test so order-of-execution doesn't matter and
we never touch the real `data/forzatek.db`.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

# Allow running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core import database
from backend.system import stats


# ─── Test fixtures ─────────────────────────────────────────────────────────
def _fresh_db() -> Path:
    """Return a path to a brand-new initialized DB in a temp dir."""
    tmp = Path(tempfile.mkdtemp(prefix="forzatek_test_")) / "forzatek.db"
    database.init_db(tmp)
    return tmp


def _seed_frame(db_path: Path, *, version="fh5", labeled=False) -> int:
    """Insert one frame, optionally with a confirmed label. Returns frame id."""
    with database.write_conn(db_path) as c:
        cur = c.execute(
            "INSERT INTO frames (ts, source_type, game_version, phash, frame_jpeg) "
            "VALUES (?, 'live', ?, ?, ?)",
            (time.time(), version, 0, b"\x00\x01\x02\x03"),
        )
        fid = cur.lastrowid
        if labeled:
            c.execute(
                "INSERT INTO labels (frame_id, task, data_json, provenance, created_at) "
                "VALUES (?, 'seg', ?, 'manual', ?)",
                (fid, "{}", time.time()),
            )
    return fid


def _patch_db_path(db_path: Path):
    """Point the database module at our temp DB for the duration of a test.

    Subtle: Python evaluates default arguments at *function definition time*,
    so `database.overall_stats(db_path=DB_PATH)` has captured the original
    Path object. Rebinding `database.DB_PATH` does NOT affect those defaults.
    We instead monkey-patch the `overall_stats` callable itself to pass our
    temp path through. This keeps the stats module untouched.
    """
    class _Ctx:
        def __enter__(self):
            self._orig_db_path        = database.DB_PATH
            self._orig_overall_stats  = database.overall_stats
            self._orig_count_frames   = database.count_frames
            self._orig_count_labels   = database.count_labels
            self._orig_get_active     = database.get_active_model

            database.DB_PATH = db_path
            database.overall_stats   = lambda dp=db_path: self._orig_overall_stats(dp)
            database.count_frames    = lambda dp=db_path, **f: self._orig_count_frames(dp, **f)
            database.count_labels    = lambda task=None, dp=db_path: self._orig_count_labels(task, dp)
            database.get_active_model = lambda dp=db_path: self._orig_get_active(dp)
            return self

        def __exit__(self, *exc):
            database.DB_PATH         = self._orig_db_path
            database.overall_stats   = self._orig_overall_stats
            database.count_frames    = self._orig_count_frames
            database.count_labels    = self._orig_count_labels
            database.get_active_model = self._orig_get_active
    return _Ctx()


# ─── Tests ────────────────────────────────────────────────────────────────
def test_health_payload_shape():
    h = stats.health()
    assert h["ok"] is True
    assert h["service"] == "forzatek-v2"
    assert isinstance(h["ts"], float)
    print("✓ health() returns expected shape")


def test_snapshot_on_empty_db():
    db = _fresh_db()
    with _patch_db_path(db):
        snap = stats.dashboard_snapshot()
    assert snap["total_frames"]   == 0
    assert snap["labeled_frames"] == 0
    assert snap["proposed_frames"]== 0
    assert snap["queue_size"]     == 0
    assert snap["frames_by_version"] == {}
    assert snap["active_model"]   is None
    assert "server_ts" in snap
    print("✓ dashboard_snapshot returns zeros for empty DB")


def test_snapshot_counts_frames_and_labels():
    db = _fresh_db()
    _seed_frame(db, version="fh5", labeled=True)
    _seed_frame(db, version="fh5", labeled=False)
    _seed_frame(db, version="fh4", labeled=True)
    with _patch_db_path(db):
        snap = stats.dashboard_snapshot()
    assert snap["total_frames"] == 3, snap
    assert snap["labeled_frames"] == 2, snap
    assert snap["frames_by_version"] == {"fh4": 1, "fh5": 2}, snap["frames_by_version"]
    print("✓ snapshot reports correct frame and label counts")


def test_snapshot_includes_runtime_defaults():
    db = _fresh_db()
    # Make sure no test left a registered provider behind.
    stats._STATE_PROVIDERS.clear()
    with _patch_db_path(db):
        snap = stats.dashboard_snapshot()
    rt = snap["runtime"]
    assert rt["capture"]["state"]    == "idle"
    assert rt["gamepad"]["state"]    == "disconnected"
    assert rt["telemetry"]["state"]  == "idle"
    print("✓ runtime block has Module 1 default placeholders")


def test_register_state_overrides_default():
    db = _fresh_db()
    stats._STATE_PROVIDERS.clear()

    def fake_capture():
        return {"state": "recording", "frames": 42}

    stats.register_state("capture", fake_capture)
    try:
        with _patch_db_path(db):
            snap = stats.dashboard_snapshot()
        assert snap["runtime"]["capture"]["state"] == "recording"
        assert snap["runtime"]["capture"]["frames"] == 42
        # Other namespaces still get defaults
        assert snap["runtime"]["gamepad"]["state"] == "disconnected"
    finally:
        stats._STATE_PROVIDERS.clear()
    print("✓ register_state() lets a module override its namespace")


def test_register_state_rejects_duplicate_namespace():
    stats._STATE_PROVIDERS.clear()
    stats.register_state("capture", lambda: {"state": "idle"})
    try:
        try:
            stats.register_state("capture", lambda: {"state": "idle"})
        except ValueError:
            print("✓ register_state() rejects duplicate namespace")
        else:
            raise AssertionError("expected ValueError on duplicate namespace")
    finally:
        stats._STATE_PROVIDERS.clear()


def test_broken_provider_does_not_break_snapshot():
    """A faulty provider should yield {'error': ...} but not raise."""
    db = _fresh_db()
    stats._STATE_PROVIDERS.clear()

    def explode():
        raise RuntimeError("boom")

    stats.register_state("gamepad", explode)
    try:
        with _patch_db_path(db):
            snap = stats.dashboard_snapshot()
        # The provider was called; it raised; we converted to an error dict.
        assert "error" in snap["runtime"]["gamepad"]
        assert "boom" in snap["runtime"]["gamepad"]["error"]
        # Total frames still computed
        assert snap["total_frames"] == 0
    finally:
        stats._STATE_PROVIDERS.clear()
    print("✓ broken provider becomes {error: ...}, snapshot still succeeds")


def test_provider_returning_non_dict_is_neutralized():
    db = _fresh_db()
    stats._STATE_PROVIDERS.clear()
    stats.register_state("telemetry", lambda: "not a dict")
    try:
        with _patch_db_path(db):
            snap = stats.dashboard_snapshot()
        assert "error" in snap["runtime"]["telemetry"]
    finally:
        stats._STATE_PROVIDERS.clear()
    print("✓ non-dict provider output is converted to an error")


def test_snapshot_is_json_serializable():
    """The dashboard ships this over the wire — must round-trip through JSON."""
    db = _fresh_db()
    with _patch_db_path(db):
        snap = stats.dashboard_snapshot()
    s = json.dumps(snap, default=str)
    assert isinstance(s, str) and len(s) > 10
    again = json.loads(s)
    assert again["total_frames"] == 0
    print("✓ snapshot is JSON-serializable")


# ─── Runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_health_payload_shape()
    test_snapshot_on_empty_db()
    test_snapshot_counts_frames_and_labels()
    test_snapshot_includes_runtime_defaults()
    test_register_state_overrides_default()
    test_register_state_rejects_duplicate_namespace()
    test_broken_provider_does_not_break_snapshot()
    test_provider_returning_non_dict_is_neutralized()
    test_snapshot_is_json_serializable()
    print("\nAll Module 1 system tests passed.")