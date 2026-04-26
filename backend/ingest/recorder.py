"""
ForzaTek AI v2 — Live Recorder
==============================
Background-thread screen-capture loop. Writes one row to `frames` per
captured frame, with `source_type='live'`.

Design constraints
------------------
- One singleton recorder. You can't start two simultaneous captures of
  the same screen — that would just contend on dxcam.
- pHash-based dedup against the last N frames. Driving in a straight
  line at low telemetry rate would otherwise produce hundreds of
  near-identical frames per minute.
- Frame tagging (biome / weather / time-of-day / game_version) is
  read from telemetry over HTTP. We do NOT import the telemetry module —
  module independence is the whole point of v2. If telemetry isn't up
  yet (Module 7 builds it later) we just leave the tags NULL and move on.
- The capture backend is dxcam on Windows, mss everywhere else /
  fallback. Both wrapped behind a tiny `_grab_screen()` helper.
- The latest captured frame (BGR np.ndarray, 720p JPEG bytes) is held
  in memory for the live preview MJPEG stream in `routes.py`.

Threading
---------
- Main thread calls `start()` / `stop()`.
- A worker thread runs `_loop()` until `_stop_event` is set.
- A small `_lock` guards the latest-frame snapshot the route reads.

Frame storage shape
-------------------
- Captured frame is resized so its height equals
  `settings["frame_resize_height"]` (default 720), preserving aspect.
- Encoded as JPEG quality `settings["jpeg_quality"]` (default 85).
- Stored as a BLOB in `frames.frame_jpeg`.
"""
from __future__ import annotations

import io
import logging
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from backend.core import database
from backend import settings as settings_mod

log = logging.getLogger("forzatek.ingest.recorder")

# ─── Optional deps ─────────────────────────────────────────────────────────
# We keep these as soft imports so a developer on a non-Windows box can
# at least import the module and run unit tests with mocked frames.
try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:  # pragma: no cover
    _CV2_OK = False
    cv2 = None  # type: ignore

try:
    import dxcam  # type: ignore
    _DXCAM_OK = True
except Exception:
    _DXCAM_OK = False
    dxcam = None  # type: ignore

try:
    import mss  # type: ignore
    _MSS_OK = True
except Exception:
    _MSS_OK = False
    mss = None  # type: ignore

try:
    import requests  # type: ignore
    _REQUESTS_OK = True
except Exception:
    _REQUESTS_OK = False
    requests = None  # type: ignore


# ─── pHash (perceptual hash) ───────────────────────────────────────────────

def phash64(bgr: np.ndarray) -> int:
    """Cheap 64-bit perceptual hash. 8x8 DCT of the low-frequency block.

    We avoid scipy / imagehash to keep deps minimal — this is the same
    approach but ~30 lines.
    """
    if not _CV2_OK:
        # Without cv2 we degenerate to a deterministic "hash" of the
        # mean — useless for dedup but lets the module import.
        return int(bgr.mean() * 1000) & 0xFFFFFFFFFFFFFFFF

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(g)
    block = dct[:8, :8].copy()
    block[0, 0] = 0  # drop DC term — it's just average brightness
    median = np.median(block)
    bits = (block > median).flatten().astype(np.uint64)
    h = 0
    for i, b in enumerate(bits):
        if b:
            h |= (1 << i)
    return int(h)


def hamming64(a: int, b: int) -> int:
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count("1")


# ─── Recorder state ────────────────────────────────────────────────────────

@dataclass
class _RecState:
    running: bool = False
    started_at: float = 0.0
    frames_written: int = 0
    frames_skipped_dup: int = 0
    last_fps: float = 0.0
    last_error: Optional[str] = None
    backend: str = "none"
    monitor_index: int = 0
    width: int = 0
    height: int = 0

    # bucket counters — (game, biome, weather, time_of_day) -> count
    # Held in memory because the UI grid needs to update at recording rate;
    # the DB has the same info but a query per tick would be wasteful.
    buckets: dict[tuple, int] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "state": "recording" if self.running else "idle",
            "running": self.running,
            "started_at": self.started_at,
            "uptime_sec": (time.time() - self.started_at) if self.running else 0.0,
            "frames_written": self.frames_written,
            "frames_skipped_dup": self.frames_skipped_dup,
            "last_fps": round(self.last_fps, 2),
            "backend": self.backend,
            "monitor_index": self.monitor_index,
            "width": self.width,
            "height": self.height,
            "last_error": self.last_error,
            "buckets": [
                {
                    "game": k[0], "biome": k[1],
                    "weather": k[2], "time_of_day": k[3],
                    "count": v,
                }
                for k, v in self.buckets.items()
            ],
        }


# ─── The actual recorder ───────────────────────────────────────────────────

class Recorder:
    """Singleton-ish. Use the module-level `RECORDER` instance."""

    def __init__(self) -> None:
        self._state = _RecState()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Latest captured frame (for live preview)
        self._latest_jpeg: Optional[bytes] = None

        # pHash dedup ring
        self._phash_ring: list[int] = []
        self._phash_ring_max = 30
        self._phash_distance = 6  # bits — tune if dedup is too eager / lazy

        # dxcam camera object (None when not running)
        self._dxcam_cam = None
        self._mss_inst = None

    # ─── lifecycle ─────────────────────────────────────────────────────

    def start(self, monitor_index: int = 0) -> dict:
        """Start a capture session. Idempotent: a second start while running
        returns the current state without restarting."""
        with self._lock:
            if self._state.running:
                return self._state.snapshot()

            self._stop_event.clear()
            self._state = _RecState(
                running=True,
                started_at=time.time(),
                monitor_index=monitor_index,
            )
            self._phash_ring.clear()

            self._thread = threading.Thread(
                target=self._loop, name="ingest-recorder", daemon=True
            )
            self._thread.start()
            log.info("recorder started on monitor %d", monitor_index)
            return self._state.snapshot()

    def stop(self) -> dict:
        with self._lock:
            if not self._state.running:
                return self._state.snapshot()
            self._stop_event.set()
        # Wait outside the lock so the loop can finish its tick.
        if self._thread:
            self._thread.join(timeout=3.0)
        with self._lock:
            self._state.running = False
            self._teardown_capture()
            log.info(
                "recorder stopped (wrote=%d, dup_skipped=%d)",
                self._state.frames_written,
                self._state.frames_skipped_dup,
            )
            return self._state.snapshot()

    def get_state(self) -> dict:
        with self._lock:
            return self._state.snapshot()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    # ─── capture backend ───────────────────────────────────────────────

    def _setup_capture(self, monitor_index: int) -> str:
        """Initialize dxcam (preferred) or mss. Returns the backend name."""
        if _DXCAM_OK and platform.system() == "Windows":
            try:
                self._dxcam_cam = dxcam.create(output_idx=monitor_index)
                if self._dxcam_cam is not None:
                    return "dxcam"
            except Exception as e:
                log.warning("dxcam init failed (%s); falling back to mss", e)

        if _MSS_OK:
            self._mss_inst = mss.mss()
            return "mss"

        raise RuntimeError(
            "no screen-capture backend available "
            "(install dxcam on Windows or mss for fallback)"
        )

    def _teardown_capture(self) -> None:
        if self._dxcam_cam is not None:
            try:
                self._dxcam_cam.release()
            except Exception:
                pass
            self._dxcam_cam = None
        if self._mss_inst is not None:
            try:
                self._mss_inst.close()
            except Exception:
                pass
            self._mss_inst = None

    def _grab_screen(self) -> Optional[np.ndarray]:
        """Return latest frame as BGR ndarray, or None if not ready."""
        if self._dxcam_cam is not None:
            arr = self._dxcam_cam.grab()
            if arr is None:
                return None
            # dxcam returns RGB; cv2 wants BGR.
            return arr[:, :, ::-1].copy() if _CV2_OK else arr
        if self._mss_inst is not None:
            mons = self._mss_inst.monitors
            idx = self._state.monitor_index + 1  # mss is 1-indexed (0 is "all")
            if idx >= len(mons):
                idx = 1
            shot = self._mss_inst.grab(mons[idx])
            arr = np.asarray(shot, dtype=np.uint8)
            # mss gives BGRA; drop alpha.
            return arr[:, :, :3]
        return None

    # ─── capture loop ──────────────────────────────────────────────────

    def _loop(self) -> None:
        try:
            self._state.backend = self._setup_capture(self._state.monitor_index)
        except Exception as e:
            self._state.last_error = str(e)
            self._state.running = False
            log.error("recorder failed to start: %s", e)
            return

        cfg = settings_mod.get_settings()
        target_fps = max(1, int(cfg.get("capture_fps", 15)))
        target_dt = 1.0 / target_fps
        target_h = int(cfg.get("frame_resize_height", 720))
        jpeg_q = int(cfg.get("jpeg_quality", 85))
        default_game = cfg.get("default_game_version", "fh5")

        last_tick = time.time()
        ema_fps = 0.0

        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                frame = self._grab_screen()
                if frame is None:
                    time.sleep(0.01)
                    continue

                bgr = self._resize(frame, target_h)
                h, w = bgr.shape[:2]
                self._state.width = w
                self._state.height = h

                # Dedup
                ph = phash64(bgr)
                if self._is_duplicate(ph):
                    self._state.frames_skipped_dup += 1
                else:
                    jpeg = self._encode_jpeg(bgr, jpeg_q)
                    if jpeg is not None:
                        tags = self._fetch_telemetry_tags(default_game)
                        self._insert_frame(jpeg, ph, w, h, tags, frame_orig=frame)
                        self._update_buckets(tags)
                        self._state.frames_written += 1

                        # Cache for live preview.
                        with self._lock:
                            self._latest_jpeg = jpeg

                        self._phash_ring.append(ph)
                        if len(self._phash_ring) > self._phash_ring_max:
                            self._phash_ring.pop(0)

                # FPS EMA
                dt_real = max(time.time() - last_tick, 1e-6)
                last_tick = time.time()
                inst_fps = 1.0 / dt_real
                ema_fps = 0.9 * ema_fps + 0.1 * inst_fps if ema_fps > 0 else inst_fps
                self._state.last_fps = ema_fps

                # Pace to target FPS
                spent = time.time() - t0
                slack = target_dt - spent
                if slack > 0:
                    self._stop_event.wait(timeout=slack)

            except Exception as e:
                log.exception("recorder tick failed")
                self._state.last_error = str(e)
                self._stop_event.wait(timeout=0.2)

        self._teardown_capture()

    # ─── helpers ───────────────────────────────────────────────────────

    def _resize(self, bgr: np.ndarray, target_h: int) -> np.ndarray:
        h, w = bgr.shape[:2]
        if h == target_h or not _CV2_OK:
            return bgr
        scale = target_h / h
        return cv2.resize(bgr, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)

    def _encode_jpeg(self, bgr: np.ndarray, quality: int) -> Optional[bytes]:
        if _CV2_OK:
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes() if ok else None
        # Last-ditch fallback: PNG via numpy isn't worth the dep — drop frame.
        return None

    def _is_duplicate(self, ph: int) -> bool:
        for prev in self._phash_ring:
            if hamming64(ph, prev) <= self._phash_distance:
                return True
        return False

    def _fetch_telemetry_tags(self, default_game: str) -> dict:
        """Try to read live telemetry from Module 7's HTTP route. If it's
        not up, return defaults. Never raises.

        We resolve the side-port from `backend.main` if present, else 8001.
        """
        url = f"http://127.0.0.1:8001/api/telemetry/live"
        out = {
            "game_version": default_game,
            "biome":        None,
            "weather":      None,
            "time_of_day":  None,
            "telemetry_json": None,
        }
        if not _REQUESTS_OK:
            return out
        try:
            r = requests.get(url, timeout=0.05)
            if r.status_code == 200:
                tel = r.json() or {}
                out["biome"]          = tel.get("biome")
                out["weather"]        = tel.get("weather")
                out["time_of_day"]    = tel.get("time_of_day")
                out["game_version"]   = tel.get("game_version", default_game)
                out["telemetry_json"] = tel
        except Exception:
            pass  # no telemetry up yet — fine.
        return out

    def _insert_frame(
        self,
        jpeg_bytes: bytes,
        ph: int,
        w: int,
        h: int,
        tags: dict,
        frame_orig: np.ndarray,
    ) -> None:
        import json as _json
        tel_json = (
            _json.dumps(tags["telemetry_json"])
            if tags.get("telemetry_json") else None
        )
        # SQLite INTEGER is signed 64-bit (max 2^63-1). phash64 returns
        # an unsigned 64-bit value; fold the top bit so it fits.
        ph_signed = ph - (1 << 64) if ph >= (1 << 63) else ph
        with database.write_conn() as c:
            c.execute(
                """INSERT INTO frames
                   (ts, source_id, source_type, game_version, biome, weather,
                    time_of_day, phash, frame_jpeg, width, height, telemetry_json)
                   VALUES (?, NULL, 'live', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    tags["game_version"],
                    tags["biome"],
                    tags["weather"],
                    tags["time_of_day"],
                    ph_signed,
                    jpeg_bytes,
                    w, h,
                    tel_json,
                ),
            )

    def _update_buckets(self, tags: dict) -> None:
        key = (
            tags.get("game_version") or "?",
            tags.get("biome") or "—",
            tags.get("weather") or "—",
            tags.get("time_of_day") or "—",
        )
        self._state.buckets[key] = self._state.buckets.get(key, 0) + 1


# ─── Module-level singleton ────────────────────────────────────────────────

RECORDER = Recorder()


def get_state() -> dict:
    """Used by Module 1's dashboard via stats.register_state('capture', ...)."""
    return RECORDER.get_state()


def register_with_system() -> None:
    """Hook the recorder into the dashboard's runtime-state registry.
    Called once from `backend/ingest/eel_api.py::register_eel`."""
    try:
        from backend.system import stats as _stats
        _stats.register_state("capture", get_state)
    except ValueError:
        # Already registered — idempotent.
        pass
    except Exception as e:
        log.warning("could not register capture state: %s", e)


__all__ = ["RECORDER", "Recorder", "phash64", "hamming64", "get_state",
           "register_with_system"]