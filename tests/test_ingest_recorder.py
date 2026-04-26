"""
Tests for backend.ingest.recorder (Module 2 — live recorder).

Run with:
    python -m tests.test_ingest_recorder
or:
    pytest tests/test_ingest_recorder.py

Covers:
    - phash64 / hamming64 behave on identical and different frames
    - the dedup ring rejects near-duplicates within the configured distance
    - state shape conforms to the dashboard's runtime-state contract
    - register_with_system() is idempotent and registers under "capture"
    - start() with a mocked screen-grabber writes frames to a temp DB
    - stop() drains the worker thread cleanly

The screen-capture layer is mocked: we monkey-patch `_grab_screen` on the
RECORDER instance so we never touch dxcam / mss / a real display. That
keeps these tests CI-safe.
"""
from __future__ import annotations

import sys
import tempfile
import time
import threading
from pathlib import Path

# Make the project importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa

from backend.core import database
from backend.core import paths as paths_mod
from backend.ingest import recorder as rec_mod
from backend.ingest.recorder import (
    RECORDER, Recorder, phash64, hamming64,
)


# ─── fixture helpers ───────────────────────────────────────────────────────

def _fresh_db() -> Path:
    """Create a temp DB and route every database.write_conn / database.read_conn
    call to it. Replacing module-level DB_PATH alone isn't enough — Python
    default-argument binding captures the original value at function-def time,
    so we wrap the context managers to inject the temp path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="forzatek_rec_test_"))
    tmp_db = tmp_dir / "test.db"
    database.init_db(tmp_db)

    database.DB_PATH = tmp_db
    paths_mod.DB_PATH = tmp_db

    # Wrap the context managers. We only do this once per process — if the
    # original is already wrapped, leave it.
    if not getattr(database.write_conn, "_ftk_test_wrapped", False):
        _orig_write = database.write_conn
        _orig_read  = database.read_conn

        def _wrapped_write(db_path: Path = None):
            return _orig_write(db_path if db_path is not None else database.DB_PATH)

        def _wrapped_read(db_path: Path = None):
            return _orig_read(db_path if db_path is not None else database.DB_PATH)

        _wrapped_write._ftk_test_wrapped = True  # type: ignore[attr-defined]
        _wrapped_read._ftk_test_wrapped  = True  # type: ignore[attr-defined]
        database.write_conn = _wrapped_write
        database.read_conn  = _wrapped_read
    return tmp_db


def _solid_frame(value: int, w: int = 320, h: int = 180) -> np.ndarray:
    """A perfectly uniform BGR frame. Useful as a 'null' input."""
    f = np.full((h, w, 3), value, dtype=np.uint8)
    return f


def _noise_frame(seed: int, w: int = 320, h: int = 180) -> np.ndarray:
    """Random BGR frame — every pixel a different value, so phashes differ."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


# ─── pHash tests ───────────────────────────────────────────────────────────

def test_phash_identical_frames_match():
    f = _noise_frame(seed=42)
    h1 = phash64(f)
    h2 = phash64(f.copy())
    assert h1 == h2, f"identical frames should produce identical hashes: {h1} vs {h2}"
    assert hamming64(h1, h2) == 0
    print("✓ phash64 is deterministic on identical frames")


def test_phash_different_frames_diverge():
    h1 = phash64(_noise_frame(seed=1))
    h2 = phash64(_noise_frame(seed=99))
    d = hamming64(h1, h2)
    # On 64-bit phashes, different random images have ~32 bits expected
    # distance. Anything > 8 is well outside the "duplicate" zone.
    assert d > 8, f"different noise frames should differ a lot, got hamming={d}"
    print(f"✓ phash64 separates different frames (hamming={d})")


# ─── dedup ring tests ──────────────────────────────────────────────────────

def test_dedup_ring_rejects_recent_match():
    r = Recorder()
    r._phash_ring_max = 5
    r._phash_distance = 6

    f1 = _noise_frame(seed=7)
    h1 = phash64(f1)
    r._phash_ring.append(h1)

    # The same frame should be flagged as duplicate.
    assert r._is_duplicate(h1) is True, "exact match should be a duplicate"
    print("✓ dedup ring flags exact phash match")


def test_dedup_ring_passes_distinct_frame():
    r = Recorder()
    r._phash_ring_max = 5
    r._phash_distance = 6

    h1 = phash64(_noise_frame(seed=11))
    h2 = phash64(_noise_frame(seed=12))
    r._phash_ring.append(h1)

    # Two random noise frames should be far apart.
    assert r._is_duplicate(h2) is False, "distinct frame must not match"
    print("✓ dedup ring lets distinct frames through")


# ─── state shape tests ─────────────────────────────────────────────────────

def test_initial_state_shape():
    r = Recorder()
    s = r.get_state()
    # The dashboard's `register_state("capture", ...)` provider contract
    # requires at minimum a "state" key. We promise a few more.
    for k in ("state", "running", "frames_written", "frames_skipped_dup",
              "last_fps", "backend", "buckets"):
        assert k in s, f"state should expose '{k}', got keys {list(s.keys())}"
    assert s["state"] == "idle"
    assert s["running"] is False
    assert isinstance(s["buckets"], list)
    print("✓ idle state shape conforms to runtime-state contract")


def test_register_with_system_is_idempotent():
    # Use a fresh import of stats so we don't collide with prior runs.
    from backend.system import stats as stats_mod
    stats_mod._STATE_PROVIDERS.clear()

    rec_mod.register_with_system()
    assert "capture" in stats_mod._STATE_PROVIDERS

    # Calling twice must NOT raise.
    rec_mod.register_with_system()
    assert "capture" in stats_mod._STATE_PROVIDERS

    # Calling get_state via the provider returns a dict with "state".
    snap = stats_mod._STATE_PROVIDERS["capture"]()
    assert isinstance(snap, dict) and "state" in snap

    stats_mod._STATE_PROVIDERS.clear()
    print("✓ register_with_system() is idempotent and exposes 'capture'")


# ─── end-to-end with mocked grabber ────────────────────────────────────────

class _FakeGrabber:
    """Stand-in for dxcam/mss. Yields a sequence of frames on .grab()."""
    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def grab(self):
        if self._i >= len(self._frames):
            return None
        f = self._frames[self._i]
        self._i += 1
        return f


def test_start_writes_frames_and_skips_dups():
    """End-to-end: feed a sequence of frames; verify writes + skips + DB rows."""
    _fresh_db()

    # 2 distinct random frames, then 3 copies of the second one.
    f_a = _noise_frame(seed=101)
    f_b = _noise_frame(seed=202)
    seq = [f_a, f_b, f_b.copy(), f_b.copy(), f_b.copy()]

    r = Recorder()

    # Bypass _setup_capture: pretend dxcam succeeded and inject our fake.
    fake = _FakeGrabber(seq)

    def fake_setup(monitor_index: int) -> str:
        # Install our fake into the dxcam slot so _grab_screen routes through it.
        r._dxcam_cam = fake
        return "fake"

    # Skip the real telemetry HTTP call entirely.
    def no_telemetry(_default_game):
        return {
            "game_version": "fh5",
            "biome": None,
            "weather": None,
            "time_of_day": None,
            "telemetry_json": None,
        }

    r._setup_capture = fake_setup  # type: ignore[assignment]
    r._fetch_telemetry_tags = no_telemetry  # type: ignore[assignment]

    # _grab_screen as written assumes dxcam returns RGB and flips channels.
    # Our fake returns BGR-shaped arrays already, so override to return them
    # directly (the 1:1 channel swap is irrelevant for phash testing).
    def grab():
        return fake.grab()
    r._grab_screen = grab  # type: ignore[assignment]

    r.start(monitor_index=0)
    # Wait for the queue to drain — 5 frames at 15 FPS target ≈ <1s.
    deadline = time.time() + 4.0
    while time.time() < deadline:
        st = r.get_state()
        if (st["frames_written"] + st["frames_skipped_dup"]) >= len(seq):
            break
        time.sleep(0.05)
    r.stop()

    st = r.get_state()
    assert st["frames_written"] == 2, f"expected 2 unique writes, got {st['frames_written']}"
    assert st["frames_skipped_dup"] >= 2, f"expected dup-skip ≥2, got {st['frames_skipped_dup']}"

    # Verify the DB really got 2 rows.
    with database.read_conn() as c:
        n = c.execute("SELECT COUNT(*) FROM frames WHERE source_type='live'").fetchone()[0]
    assert n == 2, f"expected 2 live frames in DB, got {n}"

    # Bucket counter should reflect the unique writes.
    total_buckets = sum(b["count"] for b in st["buckets"])
    assert total_buckets == 2, f"bucket counter mismatch: {st['buckets']}"

    print("✓ end-to-end: 2 unique frames written, 3 dups skipped, DB consistent")


def test_stop_when_already_idle_is_safe():
    r = Recorder()
    s = r.stop()
    assert s["running"] is False
    assert s["state"] == "idle"
    print("✓ stop() on idle recorder is a no-op")


def test_double_start_is_idempotent():
    """A second start() while running shouldn't spawn a second thread."""
    _fresh_db()
    r = Recorder()

    fake = _FakeGrabber([_noise_frame(seed=1) for _ in range(50)])
    r._setup_capture = lambda mi: (setattr(r, "_dxcam_cam", fake) or "fake")  # type: ignore
    r._fetch_telemetry_tags = lambda dg: {
        "game_version": "fh5", "biome": None, "weather": None,
        "time_of_day": None, "telemetry_json": None,
    }
    r._grab_screen = fake.grab  # type: ignore

    r.start()
    t1 = r._thread
    r.start()  # second call should not replace the thread
    t2 = r._thread
    assert t1 is t2, "second start() should NOT spawn a new thread"
    r.stop()
    print("✓ double-start is idempotent")


# ─── runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_phash_identical_frames_match()
    test_phash_different_frames_diverge()
    test_dedup_ring_rejects_recent_match()
    test_dedup_ring_passes_distinct_frame()
    test_initial_state_shape()
    test_register_with_system_is_idempotent()
    test_start_writes_frames_and_skips_dups()
    test_stop_when_already_idle_is_safe()
    test_double_start_is_idempotent()
    print("\nAll Module 2 recorder tests passed.")