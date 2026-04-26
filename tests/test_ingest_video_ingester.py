"""
Tests for backend.ingest.video_ingester (Module 2 — video pipeline).

Run with:
    python -m tests.test_ingest_video_ingester
or:
    pytest tests/test_ingest_video_ingester.py

Covers:
    - is_youtube_url() recognizes valid forms and rejects garbage
    - is_menu_frame() flags solid black / solid white / hue-dominated frames
      and lets a textured "driving-like" frame through
    - register_youtube + register_local insert into the `sources` table
      with correct fields
    - register_local raises FileNotFoundError on missing files
    - list_sources / progress return well-shaped dicts
    - cancel() on an unknown id is safe
    - end-to-end: a small synthetic video (cv2.VideoWriter) walks cleanly
      and produces frames in the DB (skipped if cv2 isn't installed)
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa

try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:
    _CV2_OK = False
    cv2 = None  # type: ignore

from backend.core import database
from backend.core import paths as paths_mod
from backend.ingest import video_ingester as vi


# ─── fixtures ──────────────────────────────────────────────────────────────

def _fresh_db() -> Path:
    """See test_ingest_recorder._fresh_db — same pattern."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="forzatek_vi_test_"))
    tmp_db = tmp_dir / "test.db"
    database.init_db(tmp_db)
    database.DB_PATH = tmp_db
    paths_mod.DB_PATH = tmp_db

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

    # Reset module-level registries so progress dicts don't leak between tests.
    vi._progress.clear()
    vi._threads.clear()
    vi._cancel.clear()
    return tmp_db


# ─── youtube URL detection ─────────────────────────────────────────────────

def test_is_youtube_url_accepts_common_forms():
    cases = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=abcdef12345",
        "https://youtu.be/abcdef12345",
        "https://www.youtube.com/shorts/abcdef12345",
        "https://www.youtube.com/live/abcdef12345",
    ]
    for u in cases:
        assert vi.is_youtube_url(u), f"should accept: {u}"
    print("✓ is_youtube_url accepts watch/shorts/live/youtu.be")


def test_is_youtube_url_rejects_garbage():
    for u in ("", "not a url", "https://example.com", "https://youtube.com/",
              "https://vimeo.com/12345"):
        assert not vi.is_youtube_url(u), f"should reject: {u!r}"
    print("✓ is_youtube_url rejects non-YouTube strings")


# ─── menu detector ─────────────────────────────────────────────────────────

def test_menu_detector_flags_solid_black():
    if not _CV2_OK:
        print("⚠ skipping menu black test (no cv2)"); return
    f = np.zeros((180, 320, 3), dtype=np.uint8)
    assert vi.is_menu_frame(f) is True
    print("✓ menu detector flags solid black")


def test_menu_detector_flags_solid_white():
    if not _CV2_OK:
        print("⚠ skipping menu white test (no cv2)"); return
    f = np.full((180, 320, 3), 255, dtype=np.uint8)
    assert vi.is_menu_frame(f) is True
    print("✓ menu detector flags solid white")


def test_menu_detector_passes_textured_frame():
    if not _CV2_OK:
        print("⚠ skipping menu textured test (no cv2)"); return
    rng = np.random.default_rng(7)
    f = rng.integers(0, 255, size=(180, 320, 3), dtype=np.uint8)
    # Random noise has tons of edges and even hue distribution.
    assert vi.is_menu_frame(f) is False
    print("✓ menu detector lets textured frames through")


# ─── register / list ───────────────────────────────────────────────────────

def test_register_youtube_inserts_row():
    _fresh_db()
    row = vi.register_youtube(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        game_version="fh5",
        biome_override="desert",
    )
    assert row["id"] >= 1
    assert row["kind"] == "youtube_url"
    assert row["status"] == "pending"
    assert row["game_version"] == "fh5"
    assert row["biome_override"] == "desert"

    listed = vi.list_sources()
    assert len(listed) == 1
    assert listed[0]["id"] == row["id"]
    assert listed[0]["kind"] == "youtube_url"
    print("✓ register_youtube inserts a sources row with correct fields")


def test_register_youtube_rejects_bad_url():
    _fresh_db()
    try:
        vi.register_youtube("not a url")
    except ValueError:
        print("✓ register_youtube rejects non-YouTube URL"); return
    raise AssertionError("expected ValueError")


def test_register_local_inserts_row():
    _fresh_db()
    tmp_video = Path(tempfile.mkdtemp()) / "a.mp4"
    tmp_video.write_bytes(b"x")  # not a real video, but it exists
    row = vi.register_local(str(tmp_video), game_version="fh4")
    assert row["kind"] == "video_file"
    assert row["uri"] == str(tmp_video.resolve())
    assert row["game_version"] == "fh4"
    print("✓ register_local inserts a sources row")


def test_register_local_missing_file_raises():
    _fresh_db()
    try:
        vi.register_local("/nonexistent/path.mp4")
    except FileNotFoundError:
        print("✓ register_local raises on missing file"); return
    raise AssertionError("expected FileNotFoundError")


# ─── progress / cancel ────────────────────────────────────────────────────

def test_progress_for_registered_source():
    _fresh_db()
    row = vi.register_youtube("https://youtu.be/abcdef12345")
    p = vi.progress(row["id"])
    assert p is not None
    assert p["source_id"] == row["id"]
    assert p["status"] == "registered"
    print("✓ progress() reports 'registered' for un-started sources")


def test_progress_for_unknown_id():
    _fresh_db()
    p = vi.progress(99999)
    assert p is None
    print("✓ progress() returns None for unknown id")


def test_cancel_unknown_id_is_safe():
    _fresh_db()
    out = vi.cancel(99999)
    assert isinstance(out, dict)
    print("✓ cancel() on unknown id is a no-op")


# ─── end-to-end walk ──────────────────────────────────────────────────────

def _make_synthetic_video(path: Path, n_frames: int = 30, fps: float = 10.0,
                           w: int = 320, h: int = 180) -> bool:
    """Write a tiny mp4 with `n_frames` random frames. Skip cleanly if the
    OpenCV build can't write mp4v (e.g. headless containers without ffmpeg
    sometimes drop the codec)."""
    if not _CV2_OK:
        return False
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not vw.isOpened():
        return False
    rng = np.random.default_rng(2026)
    try:
        for i in range(n_frames):
            f = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
            vw.write(f)
    finally:
        vw.release()
    return path.exists() and path.stat().st_size > 0


def test_end_to_end_local_walk():
    if not _CV2_OK:
        print("⚠ skipping end-to-end walk (no cv2)"); return

    _fresh_db()
    tmp_dir = Path(tempfile.mkdtemp(prefix="vi_e2e_"))
    vid_path = tmp_dir / "tiny.mp4"
    if not _make_synthetic_video(vid_path, n_frames=30, fps=10.0):
        print("⚠ skipping end-to-end walk (cv2 cannot write mp4 here)"); return

    row = vi.register_local(str(vid_path), game_version="fh5", biome_override="city")
    sid = row["id"]
    vi.start(sid)

    # Wait for the worker to either finish or reach steady state.
    deadline = time.time() + 12.0
    while time.time() < deadline:
        p = vi.progress(sid)
        if p and p["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.1)

    p = vi.progress(sid)
    assert p is not None
    assert p["status"] == "done", f"expected status=done, got {p['status']} (err={p.get('error')})"
    assert p["frames_written"] >= 1, "should have written at least one frame"

    # DB confirms.
    with database.read_conn() as c:
        n = c.execute(
            "SELECT COUNT(*) FROM frames WHERE source_id=? AND source_type='video'",
            (sid,),
        ).fetchone()[0]
        biome = c.execute(
            "SELECT biome FROM frames WHERE source_id=? LIMIT 1", (sid,),
        ).fetchone()[0]
    assert n == p["frames_written"], "DB row count must match progress"
    assert biome == "city", "biome_override should propagate to frames"

    print(f"✓ end-to-end local walk: wrote {n} frames, biome propagated")


# ─── runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_is_youtube_url_accepts_common_forms()
    test_is_youtube_url_rejects_garbage()
    test_menu_detector_flags_solid_black()
    test_menu_detector_flags_solid_white()
    test_menu_detector_passes_textured_frame()
    test_register_youtube_inserts_row()
    test_register_youtube_rejects_bad_url()
    test_register_local_inserts_row()
    test_register_local_missing_file_raises()
    test_progress_for_registered_source()
    test_progress_for_unknown_id()
    test_cancel_unknown_id_is_safe()
    test_end_to_end_local_walk()
    print("\nAll Module 2 video-ingester tests passed.")