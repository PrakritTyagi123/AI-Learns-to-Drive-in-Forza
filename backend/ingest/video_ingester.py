"""
ForzaTek AI v2 — Video Ingester
================================
Two-step pipeline per source:

    register   →  inserts row in `sources`, status='registered'
    start      →  background thread:
                    - if YouTube URL: yt-dlp into data/videos/
                    - cv2.VideoCapture walks the file at a configurable stride
                    - menu/loading-screen heuristic skips non-gameplay
                    - inserts each kept frame into `frames` (source_type='video')
                  status transitions: downloading → extracting → done | error | cancelled

Why the two-step
----------------
The UI wants to register a queue of videos quickly (paste 5 URLs),
inspect them, and only then kick off the heavy work. Splitting register
from start gives the user that affordance and lets us cancel cleanly.

Independence
------------
Reads no other module's internals. Writes to `frames` and `sources`
through `backend.core.database` — same as the recorder. A HUD mask is
NOT applied at ingest time; raw frames are stored intact and Module 3's
mask is applied at training/inference time. This means today's ingested
frames stay valid even if the user re-paints the HUD mask tomorrow.

Threading
---------
- One background thread per source that's currently `start()`ed.
- A small `_lock` guards `_threads` and `_progress` dicts.
- Cancellation is cooperative: the thread checks `_cancel_events[source_id]`
  between frames.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from backend.core import database
from backend.core.paths import VIDEOS_DIR, ensure_dirs
from backend import settings as settings_mod
from backend.ingest.recorder import phash64, hamming64

log = logging.getLogger("forzatek.ingest.video")

# ─── Optional deps ─────────────────────────────────────────────────────────
try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:  # pragma: no cover
    _CV2_OK = False
    cv2 = None  # type: ignore


# ─── Source kinds ──────────────────────────────────────────────────────────

# Canonical values used in the `sources.kind` column (schema.sql):
KIND_YOUTUBE = "youtube_url"
KIND_LOCAL   = "video_file"

_YT_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|shorts/|live/)|youtu\.be/)([A-Za-z0-9_-]{6,})"
)


def is_youtube_url(s: str) -> bool:
    return bool(_YT_RE.search(s or ""))


# ─── Per-source progress ───────────────────────────────────────────────────

@dataclass
class _Progress:
    source_id: int
    kind: str
    status: str = "registered"          # registered|downloading|extracting|done|error|cancelled
    download_pct: float = 0.0
    extract_pct: float = 0.0
    frames_written: int = 0
    frames_skipped_dup: int = 0
    frames_skipped_menu: int = 0
    total_frames: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0
    error: Optional[str] = None
    local_path: Optional[str] = None    # filesystem path actually being walked
    title: Optional[str] = None

    def snapshot(self) -> dict:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "status": self.status,
            "download_pct": round(self.download_pct, 2),
            "extract_pct": round(self.extract_pct, 2),
            "frames_written": self.frames_written,
            "frames_skipped_dup": self.frames_skipped_dup,
            "frames_skipped_menu": self.frames_skipped_menu,
            "total_frames": self.total_frames,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_sec": (
                (self.ended_at or time.time()) - self.started_at
                if self.started_at else 0.0
            ),
            "error": self.error,
            "local_path": self.local_path,
            "title": self.title,
        }


# ─── Menu-frame detector ───────────────────────────────────────────────────

def is_menu_frame(bgr: np.ndarray) -> bool:
    """Cheap heuristic: menu/loading screens are typically very low entropy
    OR dominated by a single hue. Driving frames have lots of edges and a
    distribution of colors.

    Tuned for false-negative (let some menus through) over false-positive
    (don't kill real driving frames). Cheap is the operative word — this
    runs once per kept candidate.
    """
    if not _CV2_OK or bgr is None or bgr.size == 0:
        return False
    h, w = bgr.shape[:2]
    if h < 16 or w < 16:
        return True

    # Edge density: Canny then mean
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 80, 160)
    edge_frac = float(np.count_nonzero(edges)) / float(edges.size)

    # Dominant color: HSV hue concentration
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].ravel()
    hist, _ = np.histogram(hue, bins=18, range=(0, 180))
    dom_hue_frac = hist.max() / max(hist.sum(), 1)

    # Brightness extremes (loading screens often pure black or pure white)
    mean_v = float(hsv[:, :, 2].mean())
    very_dark  = mean_v < 12.0
    very_light = mean_v > 240.0

    if very_dark or very_light:
        return True
    if edge_frac < 0.012 and dom_hue_frac > 0.55:
        return True
    return False


# ─── Module state ──────────────────────────────────────────────────────────

_lock = threading.Lock()
_progress: dict[int, _Progress] = {}
_threads:  dict[int, threading.Thread] = {}
_cancel:   dict[int, threading.Event] = {}


# ─── Public API ────────────────────────────────────────────────────────────

def register_youtube(
    url: str,
    game_version: Optional[str] = None,
    biome_override: Optional[str] = None,
) -> dict:
    """Insert a row into `sources` for a YouTube video. Does NOT download yet.

    Returns the new sources row as a dict.
    """
    if not is_youtube_url(url):
        raise ValueError("not a recognized YouTube URL")
    return _register_source(
        kind=KIND_YOUTUBE,
        uri=url,
        game_version=game_version,
        biome_override=biome_override,
    )


def register_local(
    file_path: str,
    game_version: Optional[str] = None,
    biome_override: Optional[str] = None,
) -> dict:
    """Insert a row into `sources` for a local file. Does NOT walk it yet."""
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"no such file: {file_path}")
    return _register_source(
        kind=KIND_LOCAL,
        uri=str(p.resolve()),
        game_version=game_version,
        biome_override=biome_override,
    )


def list_sources() -> list[dict]:
    """All registered sources, newest first, with their live progress merged in."""
    with database.read_conn() as c:
        rows = c.execute(
            "SELECT * FROM sources ORDER BY id DESC"
        ).fetchall()
    out = []
    with _lock:
        for r in rows:
            row = dict(r)
            prog = _progress.get(row["id"])
            if prog is not None:
                row["progress"] = prog.snapshot()
            else:
                db_status = row.get("status") or "pending"
                ui_status = {
                    "pending": "registered", "processing": "extracting",
                    "done": "done", "failed": "error",
                }.get(db_status, db_status)
                row["progress"] = {
                    "source_id": row["id"],
                    "status": ui_status,
                    "download_pct": 0.0,
                    "extract_pct": 100.0 if ui_status == "done" else 0.0,
                    "frames_written": row.get("frames_accepted") or 0,
                }
            out.append(row)
    return out


def progress(source_id: int) -> Optional[dict]:
    with _lock:
        p = _progress.get(source_id)
        if p is not None:
            return p.snapshot()
    # Fallback: if the thread finished and was cleaned up, read from DB.
    with database.read_conn() as c:
        r = c.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
    if r is None:
        return None
    row = dict(r)
    db_status = row.get("status") or "pending"
    # Reverse-map DB status back to UI-friendly state.
    ui_status = {"pending": "registered", "processing": "extracting",
                 "done": "done", "failed": "error"}.get(db_status, db_status)
    return {
        "source_id": source_id,
        "kind": row.get("kind"),
        "status": ui_status,
        "download_pct": 100.0 if ui_status in ("extracting", "done") else 0.0,
        "extract_pct": 100.0 if ui_status == "done" else 0.0,
        "frames_written": row.get("frames_accepted") or 0,
        "frames_skipped_dup": 0,
        "frames_skipped_menu": 0,
        "total_frames": 0,
        "started_at": 0.0,
        "ended_at": 0.0,
        "elapsed_sec": 0.0,
        "error": None,
        "local_path": None,
        "title": row.get("title"),
    }


def start(source_id: int) -> dict:
    """Spawn the background ingest thread for a source. Idempotent."""
    with _lock:
        if source_id in _threads and _threads[source_id].is_alive():
            return _progress[source_id].snapshot()

    with database.read_conn() as c:
        row = c.execute(
            "SELECT * FROM sources WHERE id=?", (source_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"no such source: {source_id}")
    src = dict(row)

    prog = _Progress(
        source_id=source_id,
        kind=src["kind"],
        status="downloading" if src["kind"] == KIND_YOUTUBE else "extracting",
        started_at=time.time(),
        local_path=src.get("local_path"),
        title=src.get("title"),
    )
    cancel_evt = threading.Event()
    th = threading.Thread(
        target=_run_source,
        args=(src, prog, cancel_evt),
        name=f"ingest-source-{source_id}",
        daemon=True,
    )
    with _lock:
        _progress[source_id] = prog
        _cancel[source_id] = cancel_evt
        _threads[source_id] = th
    th.start()
    return prog.snapshot()


def cancel(source_id: int) -> dict:
    with _lock:
        evt = _cancel.get(source_id)
        prog = _progress.get(source_id)
    if evt is not None:
        evt.set()
    if prog is None:
        return {"source_id": source_id, "status": "unknown"}
    return prog.snapshot()


# ─── Internal: registration ────────────────────────────────────────────────

def _register_source(
    kind: str,
    uri: str,
    game_version: Optional[str],
    biome_override: Optional[str],
) -> dict:
    cfg = settings_mod.get_settings()
    gv = game_version or cfg.get("default_game_version", "fh5")
    with database.write_conn() as c:
        cur = c.execute(
            """INSERT INTO sources
               (kind, uri, title, game_version, biome_override,
                frames_sampled, frames_accepted, status, created_at)
               VALUES (?, ?, ?, ?, ?, 0, 0, 'pending', ?)""",
            (kind, uri, _derive_title(kind, uri), gv, biome_override, time.time()),
        )
        sid = cur.lastrowid
        row = c.execute("SELECT * FROM sources WHERE id=?", (sid,)).fetchone()
    return dict(row)


def _derive_title(kind: str, uri: str) -> str:
    if kind == KIND_LOCAL:
        return Path(uri).name
    m = _YT_RE.search(uri)
    if m:
        return f"YouTube: {m.group(1)}"
    return uri


# ─── Internal: per-source worker ───────────────────────────────────────────

def _run_source(src: dict, prog: _Progress, cancel_evt: threading.Event) -> None:
    sid = src["id"]
    try:
        local_path: Optional[Path] = None

        if src["kind"] == KIND_YOUTUBE:
            local_path = _download_youtube(src, prog, cancel_evt)
            if cancel_evt.is_set():
                _finalize(sid, prog, "cancelled")
                return
            prog.local_path = str(local_path) if local_path else None
            _set_db_status(sid, "extracting")
        else:
            local_path = Path(src["uri"])
            prog.local_path = str(local_path)

        prog.status = "extracting"
        _walk_video(local_path, src, prog, cancel_evt)
        if cancel_evt.is_set():
            _finalize(sid, prog, "cancelled")
            return

        _finalize(sid, prog, "done")

    except Exception as e:
        log.exception("source %d failed", sid)
        prog.error = str(e)
        _finalize(sid, prog, "error")


def _finalize(source_id: int, prog: _Progress, status: str) -> None:
    prog.status = status
    prog.ended_at = time.time()
    _set_db_status(
        source_id, status,
        frames_sampled=prog.frames_skipped_dup + prog.frames_skipped_menu + prog.frames_written,
        frames_accepted=prog.frames_written,
    )


# Map our richer in-memory states onto the four DB-allowed values.
_DB_STATUS = {
    "registered":  "pending",
    "downloading": "processing",
    "extracting":  "processing",
    "done":        "done",
    "error":       "failed",
    "cancelled":   "failed",
}


def _set_db_status(
    source_id: int,
    status: str,
    *,
    frames_sampled: Optional[int] = None,
    frames_accepted: Optional[int] = None,
) -> None:
    db_status = _DB_STATUS.get(status, status)
    sets, args = ["status = ?"], [db_status]
    if frames_sampled is not None:
        sets.append("frames_sampled = ?"); args.append(frames_sampled)
    if frames_accepted is not None:
        sets.append("frames_accepted = ?"); args.append(frames_accepted)
    args.append(source_id)
    with database.write_conn() as c:
        c.execute(f"UPDATE sources SET {', '.join(sets)} WHERE id = ?", args)


# ─── Internal: yt-dlp download ─────────────────────────────────────────────

def _download_youtube(
    src: dict, prog: _Progress, cancel_evt: threading.Event
) -> Optional[Path]:
    ensure_dirs()
    out_tpl = str(VIDEOS_DIR / f"yt_{src['id']}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "--newline",
        "-o", out_tpl,
        src["uri"],
    ]

    if shutil.which("yt-dlp") is None:
        raise RuntimeError(
            "yt-dlp not found on PATH. `pip install yt-dlp` "
            "or install the binary."
        )

    log.info("downloading source %d via yt-dlp", src["id"])
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    pct_re = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")
    assert proc.stdout is not None
    for line in proc.stdout:
        if cancel_evt.is_set():
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            return None
        m = pct_re.search(line)
        if m:
            try:
                prog.download_pct = float(m.group(1))
            except ValueError:
                pass
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp exited {proc.returncode}")

    # Find the resulting file
    for ext in ("mp4", "mkv", "webm"):
        candidate = VIDEOS_DIR / f"yt_{src['id']}.{ext}"
        if candidate.exists():
            prog.download_pct = 100.0
            return candidate
    raise RuntimeError("yt-dlp finished but no output file found")


# ─── Internal: video walking ───────────────────────────────────────────────

def _walk_video(
    path: Path, src: dict, prog: _Progress, cancel_evt: threading.Event
) -> None:
    if not _CV2_OK:
        raise RuntimeError("opencv-python not installed; cannot walk videos")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    try:
        cfg = settings_mod.get_settings()
        target_fps_out = max(1, int(cfg.get("capture_fps", 15)))
        target_h = int(cfg.get("frame_resize_height", 720))
        jpeg_q = int(cfg.get("jpeg_quality", 85))
        gv = src.get("game_version") or cfg.get("default_game_version", "fh5")
        biome_override = src.get("biome_override")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        prog.total_frames = total

        # Frame stride such that effective output rate ≈ target_fps_out
        stride = max(1, int(round(src_fps / target_fps_out)))
        log.info(
            "walking %s: src_fps=%.1f stride=%d est_total=%d",
            path.name, src_fps, stride, total,
        )

        ring: list[int] = []
        ring_max = 60        # bigger ring than live recorder — videos are slower-changing
        dist = 6

        idx = 0
        kept = 0
        while True:
            if cancel_evt.is_set():
                return
            ok = cap.grab()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                idx += 1
                continue

            bgr = _resize(frame, target_h)
            if is_menu_frame(bgr):
                prog.frames_skipped_menu += 1
                idx += 1
                continue

            ph = phash64(bgr)
            if any(hamming64(ph, p) <= dist for p in ring):
                prog.frames_skipped_dup += 1
                idx += 1
                continue
            ring.append(ph)
            if len(ring) > ring_max:
                ring.pop(0)

            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            if not ok:
                idx += 1
                continue
            jpeg = buf.tobytes()
            h, w = bgr.shape[:2]
            video_t = idx / src_fps if src_fps > 0 else 0.0
            ph_signed = ph - (1 << 64) if ph >= (1 << 63) else ph

            with database.write_conn() as c:
                c.execute(
                    """INSERT INTO frames
                       (ts, source_id, source_type, game_version, biome,
                        weather, time_of_day, phash, frame_jpeg,
                        width, height, video_time_sec)
                       VALUES (?, ?, 'video', ?, ?, NULL, NULL, ?, ?, ?, ?, ?)""",
                    (
                        time.time(), src["id"], gv, biome_override,
                        ph_signed, jpeg, w, h, video_t,
                    ),
                )

            kept += 1
            prog.frames_written = kept
            if total > 0:
                prog.extract_pct = 100.0 * idx / max(total, 1)
            idx += 1
    finally:
        cap.release()
    prog.extract_pct = 100.0


def _resize(bgr: np.ndarray, target_h: int) -> np.ndarray:
    if not _CV2_OK or bgr is None:
        return bgr
    h, w = bgr.shape[:2]
    if h == target_h:
        return bgr
    scale = target_h / h
    return cv2.resize(bgr, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


__all__ = [
    "register_youtube", "register_local",
    "list_sources", "progress", "start", "cancel",
    "is_youtube_url", "is_menu_frame",
    "KIND_YOUTUBE", "KIND_LOCAL",
]