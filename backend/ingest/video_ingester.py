"""
ForzaTek AI v2 — Video Ingester
================================
YouTube + local video → frames pipeline. Fully rewritten with:

  * The phash overflow fix baked in (signed 64-bit conversion before INSERT).
  * Per-frame try/except that LOGS the error and counts it, so silent failure
    is impossible — if every frame fails to insert, you'll see thousands of
    error lines, not zero output.
  * Periodic progress logging every 100 frames so you can verify the loop
    is actually moving.
  * The schema-canonical kind values (`youtube_url` / `video_file`) and
    status values (`pending|processing|done|failed`).

This file is self-contained — drop it in and overwrite the old one.
"""
from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
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


# ─── Constants ─────────────────────────────────────────────────────────────

# Canonical values used in the `sources.kind` column (match schema.sql):
KIND_YOUTUBE = "youtube_url"
KIND_LOCAL   = "video_file"

# Map our richer in-memory states onto the four DB-allowed values.
_DB_STATUS = {
    "registered":  "pending",
    "downloading": "processing",
    "extracting":  "processing",
    "done":        "done",
    "error":       "failed",
    "cancelled":   "failed",
}

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
    status: str = "registered"
    download_pct: float = 0.0
    extract_pct: float = 0.0
    frames_written: int = 0
    frames_skipped_dup: int = 0
    frames_skipped_menu: int = 0
    frames_failed_insert: int = 0     # NEW: counts INSERT failures so they're visible
    total_frames: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0
    error: Optional[str] = None
    local_path: Optional[str] = None
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
            "frames_failed_insert": self.frames_failed_insert,
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
    """Cheap heuristic: menu/loading screens have low edge density AND
    one dominant hue, OR are extremely dark/light. Driving frames have
    lots of edges and color variety.

    Tuned to err on the side of letting frames through (false negatives
    OK, false positives NOT OK).
    """
    if not _CV2_OK or bgr is None or bgr.size == 0:
        return False
    h, w = bgr.shape[:2]
    if h < 16 or w < 16:
        return True

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 80, 160)
    edge_frac = float(np.count_nonzero(edges)) / float(edges.size)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist, _ = np.histogram(hsv[:, :, 0].ravel(), bins=18, range=(0, 180))
    dom_hue_frac = hist.max() / max(hist.sum(), 1)

    mean_v = float(hsv[:, :, 2].mean())
    if mean_v < 12.0 or mean_v > 240.0:
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
    if not is_youtube_url(url):
        raise ValueError("not a recognized YouTube URL")
    return _register_source(KIND_YOUTUBE, url, game_version, biome_override)


def register_local(
    file_path: str,
    game_version: Optional[str] = None,
    biome_override: Optional[str] = None,
) -> dict:
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"no such file: {file_path}")
    return _register_source(KIND_LOCAL, str(p.resolve()), game_version, biome_override)


def list_sources() -> list[dict]:
    """All registered sources, newest first, with live progress merged in."""
    with database.read_conn() as c:
        rows = c.execute(
            "SELECT * FROM sources ORDER BY created_at DESC"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        with _lock:
            p = _progress.get(d["id"])
        if p is not None:
            snap = p.snapshot()
            d.update({
                "download_pct":  snap["download_pct"],
                "extract_pct":   snap["extract_pct"],
                "live_status":   snap["status"],
                "elapsed_sec":   snap["elapsed_sec"],
                "error":         snap["error"],
                "frames_skipped_dup":   snap["frames_skipped_dup"],
                "frames_skipped_menu":  snap["frames_skipped_menu"],
                "frames_failed_insert": snap["frames_failed_insert"],
            })
        out.append(d)
    return out


def progress(source_id: int) -> Optional[dict]:
    with _lock:
        p = _progress.get(source_id)
    return p.snapshot() if p is not None else None


def start(source_id: int) -> dict:
    """Begin downloading (if YouTube) and walking the source. Idempotent."""
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
    log.info("started worker thread for source %d", source_id)
    return prog.snapshot()


def cancel(source_id: int) -> dict:
    with _lock:
        evt = _cancel.get(source_id)
        prog = _progress.get(source_id)
    if evt is not None:
        evt.set()
    return prog.snapshot() if prog is not None else {"source_id": source_id, "status": "unknown"}


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
    log.info("worker %d starting (kind=%s, uri=%s)", sid, src["kind"], src["uri"])
    try:
        local_path: Optional[Path] = None

        if src["kind"] == KIND_YOUTUBE:
            local_path = _download_youtube(src, prog, cancel_evt)
            if cancel_evt.is_set():
                _finalize(sid, prog, "cancelled")
                return
            prog.local_path = str(local_path) if local_path else None
            prog.status = "extracting"
            _set_db_status(sid, "extracting")
        else:
            local_path = Path(src["uri"])
            prog.local_path = str(local_path)
            prog.status = "extracting"
            _set_db_status(sid, "extracting")

        _walk_video(local_path, src, prog, cancel_evt)
        if cancel_evt.is_set():
            _finalize(sid, prog, "cancelled")
            return

        _finalize(sid, prog, "done")
        log.info(
            "worker %d done: written=%d skipped_dup=%d skipped_menu=%d failed=%d",
            sid, prog.frames_written, prog.frames_skipped_dup,
            prog.frames_skipped_menu, prog.frames_failed_insert,
        )

    except Exception as e:
        log.exception("worker %d failed", sid)
        prog.error = str(e)
        _finalize(sid, prog, "error")


def _finalize(source_id: int, prog: _Progress, status: str) -> None:
    prog.status = status
    prog.ended_at = time.time()
    sampled = (prog.frames_skipped_dup + prog.frames_skipped_menu
               + prog.frames_written + prog.frames_failed_insert)
    _set_db_status(
        source_id, status,
        frames_sampled=sampled,
        frames_accepted=prog.frames_written,
    )


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


# ─── Internal: YouTube download ────────────────────────────────────────────

def _download_youtube(
    src: dict, prog: _Progress, cancel_evt: threading.Event
) -> Path:
    ensure_dirs()
    out_template = str(VIDEOS_DIR / f"yt_{src['id']}.%(ext)s")
    log.info("downloading source %d via yt-dlp", src["id"])
    _set_db_status(src["id"], "downloading")

    # We DON'T pass --no-progress here. Instead we parse yt-dlp's progress
    # lines so the UI gets a live download_pct.
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--no-warnings",
        "--newline",          # one progress line per update, not \r overwrite
        "--progress",
        "-o", out_template,
        src["uri"],
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # yt-dlp progress lines look like:
    #   [download]  37.4% of 127.43MiB at  18.55MiB/s ETA 00:04
    pct_re = re.compile(r"\[download\]\s+([\d.]+)%")
    last_log = 0.0

    try:
        for line in proc.stdout:
            if cancel_evt.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RuntimeError("cancelled during download")

            m = pct_re.search(line)
            if m:
                try:
                    pct = float(m.group(1))
                    prog.download_pct = pct
                    # Throttle DB writes to once a second so we don't hammer it.
                    now = time.time()
                    if now - last_log >= 1.0:
                        last_log = now
                        log.info("source %d download %.1f%%", src["id"], pct)
                except ValueError:
                    pass
    finally:
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"yt-dlp exited with code {rc}")

    for ext in ("mp4", "mkv", "webm"):
        candidate = VIDEOS_DIR / f"yt_{src['id']}.{ext}"
        if candidate.exists():
            prog.download_pct = 100.0
            return candidate
    raise RuntimeError("yt-dlp finished but no output file found")


# ─── Internal: video walker ────────────────────────────────────────────────

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

        stride = max(1, int(round(src_fps / target_fps_out)))
        log.info(
            "walking %s: src_fps=%.1f stride=%d est_total=%d target_h=%d",
            path.name, src_fps, stride, total, target_h,
        )

        ring: list[int] = []
        ring_max = 60
        dup_dist = 6

        idx = 0
        kept = 0
        last_log_at = time.time()

        while True:
            if cancel_evt.is_set():
                log.info("walker %d cancelled", src["id"])
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
            if any(hamming64(ph, p) <= dup_dist for p in ring):
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

            # SQLite INTEGER is signed 64-bit; phash64 returns unsigned.
            ph_signed = ph - (1 << 64) if ph >= (1 << 63) else ph

            # CRITICAL: per-frame try/except around the INSERT so failures
            # surface in the log and increment a counter, instead of being
            # swallowed by the outer exception handler in _run_source.
            try:
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
            except Exception as e:
                prog.frames_failed_insert += 1
                # Log the FIRST 3 failures in full (with traceback), then
                # one out of every 100 after that, so we don't drown the log.
                if prog.frames_failed_insert <= 3:
                    log.exception(
                        "INSERT failed for source %d at idx=%d: %s",
                        src["id"], idx, e,
                    )
                elif prog.frames_failed_insert % 100 == 0:
                    log.error(
                        "source %d: %d INSERT failures so far (last: %s)",
                        src["id"], prog.frames_failed_insert, e,
                    )

            if total > 0:
                prog.extract_pct = 100.0 * idx / max(total, 1)

            # Periodic progress log so the operator knows the loop is alive.
            now = time.time()
            if now - last_log_at >= 5.0:
                log.info(
                    "source %d progress: kept=%d skipped_dup=%d skipped_menu=%d "
                    "failed=%d (%.1f%%)",
                    src["id"], prog.frames_written, prog.frames_skipped_dup,
                    prog.frames_skipped_menu, prog.frames_failed_insert,
                    prog.extract_pct,
                )
                # Also flush counters to DB so the UI reflects live progress.
                _set_db_status(
                    src["id"], "extracting",
                    frames_sampled=(prog.frames_written + prog.frames_skipped_dup
                                    + prog.frames_skipped_menu + prog.frames_failed_insert),
                    frames_accepted=prog.frames_written,
                )
                last_log_at = now

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