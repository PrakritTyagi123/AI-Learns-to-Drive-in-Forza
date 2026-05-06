"""
Module 5 — Memmap dataset cache
================================
Bakes the labeled subset of `frames` + `labels` into three flat files:
  data/perception_cache/
    images.bin   — uint8, shape (N, INPUT_H, INPUT_W, 3)   ≈ 0.4 MB / sample
    masks.bin    — uint8, shape (N, INPUT_H, INPUT_W)      ≈ 0.15 MB / sample
    meta.json    — index: frame_id, game_version, det boxes per row

Why this exists:
  Training reads the same data dozens of times across epochs. Reading from
  SQLite + decoding JPEG + applying HUD mask + resizing on every step blocks
  the GPU. With this cache, each step is a numpy.memmap read (kernel page
  cache, parallel-safe across worker processes) plus augmentation.

Key invariants:
  * The cache is always in the SAME schema as the live training pipeline:
    image is BGR uint8 at INPUT_H×INPUT_W with HUD mask already applied;
    mask is class ids 0–3 + 255 for ignore.
  * `meta.json` holds the source of truth for ordering. The .bin files are
    indexed positionally in the same order as the meta entries.
  * Staleness signal lives in meta.json: counts and max(labels.created_at)
    against the DB. If anything moved, rebuild.

Public surface:
  build(progress_callback=None, cancel_check=None) -> dict      # blocking, builds cache
  status() -> dict                                              # cheap, for the UI
  is_stale() -> bool                                            # cheap
  load_meta() -> dict | None                                    # for the dataset
  CachedPerceptionDataset(ids, train, augment_strength=1.0)     # drop-in replacement
  make_splits_from_cache(seed, val_frac, game_versions=None)
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from backend.core import database, paths
from backend.perception.classes import (
    DET_GRID_H,
    DET_GRID_W,
    EXPECTED_DET_FORMAT,
    EXPECTED_SEG_FORMAT,
    IGNORE_INDEX,
    INPUT_H,
    INPUT_W,
    NUM_DET_CLASSES,
    NUM_SEG_CLASSES,
)
from backend.perception.model import encode_det_targets

log = logging.getLogger("forzatek.perception.cache")

# Optional decoders — same fallback chain as dataset.py.
try:
    import cv2  # type: ignore
except Exception:                                  # pragma: no cover
    cv2 = None
try:
    from PIL import Image
except Exception:                                  # pragma: no cover
    Image = None
try:
    from backend.hud_mask import auto_propagate as _hud_mask
except Exception:                                  # pragma: no cover
    _hud_mask = None


# ─── Paths ─────────────────────────────────────────────────────────────────
def _cache_dir() -> Path:
    p = paths.DATA_DIR / "perception_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p

# v1 (raw uint8) layout:
def _images_path() -> Path: return _cache_dir() / "images.bin"
def _masks_path()  -> Path: return _cache_dir() / "masks.bin"

# v2 (compressed) layout — variable-length JPEGs + PNGs with offset tables:
def _images_jpeg_path() -> Path:    return _cache_dir() / "images_jpeg.bin"
def _images_jpeg_idx_path() -> Path: return _cache_dir() / "images_jpeg.idx"
def _masks_png_path()  -> Path:     return _cache_dir() / "masks_png.bin"
def _masks_png_idx_path() -> Path:  return _cache_dir() / "masks_png.idx"

# Both versions share these:
def _det_targets_path() -> Path: return _cache_dir() / "det_targets.bin"
def _meta_path()   -> Path: return _cache_dir() / "meta.json"
def _build_progress_path() -> Path: return _cache_dir() / "_build_progress.json"
def _build_log_path() -> Path: return _cache_dir() / "_build_log.txt"
def _cancel_flag_path() -> Path: return _cache_dir() / "_build_cancel.flag"

# Format identifiers used in meta.json["format"].
# v1 = raw uint8 fixed-size (43 GB for 72k frames at 512x288). Deprecated
#      because it doesn't fit in RAM, causing disk thrashing during training.
# v2 = JPEG-compressed images + PNG-compressed masks + flat det targets file.
#      ~5 GB for 72k frames, fits comfortably in RAM page cache.
FORMAT_V1_RAW = "raw_uint8_v1"
FORMAT_V2_COMPRESSED = "compressed_v1"

# JPEG quality used for the cache. 90 is essentially indistinguishable from
# the source while still giving ~17:1 compression on natural frames.
_CACHE_JPEG_QUALITY = 90


# ─── Decoder helpers (mirror dataset.py exactly) ───────────────────────────
def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    if cv2 is not None:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img
    if Image is None:                              # pragma: no cover
        raise RuntimeError("Need OpenCV or Pillow to decode JPEGs.")
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    arr = np.asarray(img)[:, :, ::-1]
    return np.ascontiguousarray(arr)


def _decode_seg_mask(b64: str) -> np.ndarray:
    if not b64:
        raise ValueError("empty seg mask")
    raw = base64.b64decode(b64)
    if Image is not None:
        img = Image.open(io.BytesIO(raw))
        if img.mode != "L":
            img = img.convert("L")
        return np.array(img, dtype=np.uint8)
    if cv2 is not None:                            # pragma: no cover
        arr = np.frombuffer(raw, dtype=np.uint8)
        m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError("cv2 failed to decode seg PNG")
        return m
    raise RuntimeError("Need Pillow or OpenCV to decode seg PNG.")


def _resize_image(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    if Image is None:                              # pragma: no cover
        raise RuntimeError("Need OpenCV or Pillow to resize.")
    pil = Image.fromarray(img[:, :, ::-1])
    pil = pil.resize((w, h), Image.BILINEAR)
    return np.asarray(pil)[:, :, ::-1]


def _resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if Image is None:                              # pragma: no cover
        raise RuntimeError("Need OpenCV or Pillow to resize mask.")
    pil = Image.fromarray(mask)
    pil = pil.resize((w, h), Image.NEAREST)
    return np.array(pil, dtype=np.uint8)


def _validate_seg_classes(arr: np.ndarray) -> np.ndarray:
    valid = (arr < NUM_SEG_CLASSES) | (arr == IGNORE_INDEX)
    if not valid.all():
        out = arr.copy()
        out[~valid] = IGNORE_INDEX
        return out
    return arr


# ─── Cache encoders (v2 compressed format) ────────────────────────────────
def _encode_jpeg_for_cache(img_bgr: np.ndarray, quality: int = _CACHE_JPEG_QUALITY) -> bytes:
    """Encode a 512×288 BGR uint8 frame as JPEG bytes."""
    if cv2 is not None:
        ok, buf = cv2.imencode(".jpg", img_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode JPEG failed")
        return buf.tobytes()
    if Image is None:                              # pragma: no cover
        raise RuntimeError("Need OpenCV or Pillow to encode JPEG.")
    pil = Image.fromarray(img_bgr[:, :, ::-1])      # BGR -> RGB
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def _encode_png_for_cache(mask_u8: np.ndarray) -> bytes:
    """Encode a uint8 (H, W) mask as PNG bytes. Lossless."""
    if Image is not None:
        pil = Image.fromarray(mask_u8.astype(np.uint8), mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=False, compress_level=3)
        return buf.getvalue()
    if cv2 is not None:                            # pragma: no cover
        ok, buf = cv2.imencode(".png", mask_u8.astype(np.uint8),
                               [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if not ok:
            raise RuntimeError("cv2.imencode PNG failed")
        return buf.tobytes()
    raise RuntimeError("Need Pillow or OpenCV to encode PNG.")


# ─── DB-side fingerprint for staleness ─────────────────────────────────────
def _db_fingerprint() -> Dict[str, Any]:
    """Cheap signature used to detect cache staleness without re-decoding."""
    with database.read_conn() as conn:
        row = conn.execute(
            """SELECT COUNT(DISTINCT f.id) AS n_frames,
                      COUNT(l.id)          AS n_labels,
                      COALESCE(MAX(l.created_at), 0.0) AS max_label_ts
               FROM frames f
               LEFT JOIN labels l ON l.frame_id = f.id
               WHERE f.label_status = 'labeled'
                 AND EXISTS (SELECT 1 FROM labels l2 WHERE l2.frame_id = f.id)"""
        ).fetchone()
    if row is None:
        return {"n_frames": 0, "n_labels": 0, "max_label_ts": 0.0}
    return {
        "n_frames":     int(row["n_frames"] or 0),
        "n_labels":     int(row["n_labels"] or 0),
        "max_label_ts": float(row["max_label_ts"] or 0.0),
    }


# ─── Status / staleness ────────────────────────────────────────────────────
def status() -> Dict[str, Any]:
    """Lightweight readout for the UI. Never raises."""
    out: Dict[str, Any] = {
        "exists":          False,
        "fresh":           False,
        "n_frames":        0,
        "size_bytes":      0,
        "built_at":        None,
        "format":          None,
        "current_db":      _db_fingerprint(),
        "build_in_progress": _build_in_progress(),
    }
    try:
        if not _meta_path().exists():
            return out
        meta = json.loads(_meta_path().read_text())
        out["exists"]    = True
        out["n_frames"]  = int(meta.get("n_frames", 0))
        out["built_at"]  = meta.get("built_at")
        out["cache_db"]  = meta.get("db_fingerprint", {})
        out["format"]    = meta.get("format", FORMAT_V1_RAW)

        # Sum sizes of all .bin files we know about (both formats).
        total = 0
        for p in (_images_path(), _masks_path(),
                  _images_jpeg_path(), _images_jpeg_idx_path(),
                  _masks_png_path(), _masks_png_idx_path(),
                  _det_targets_path(), _meta_path()):
            if p.exists():
                total += p.stat().st_size
        out["size_bytes"] = total

        # Freshness: db fingerprint matches AND the right files exist for the format.
        if out["format"] == FORMAT_V2_COMPRESSED:
            files_present = (_images_jpeg_path().exists()
                             and _images_jpeg_idx_path().exists()
                             and _masks_png_path().exists()
                             and _masks_png_idx_path().exists()
                             and _det_targets_path().exists())
        else:
            files_present = _images_path().exists() and _masks_path().exists()
        out["fresh"] = (out["cache_db"] == out["current_db"] and files_present)
    except Exception as e:
        log.warning("cache status read failed: %s", e)
    return out


def is_stale() -> bool:
    s = status()
    return not s.get("fresh", False)


def _build_in_progress() -> bool:
    p = _build_progress_path()
    if not p.exists():
        return False
    try:
        prog = json.loads(p.read_text())
        return prog.get("status") == "running"
    except Exception:
        return False


def build_progress() -> Dict[str, Any]:
    """Read the build's live progress JSON. Returns {'status':'idle'} if none."""
    p = _build_progress_path()
    if not p.exists():
        return {"status": "idle"}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"status": "unknown"}


def build_log_tail(n_lines: int = 200) -> List[str]:
    p = _build_log_path()
    if not p.exists():
        return []
    try:
        return [l.rstrip("\n") for l in p.read_text().splitlines()[-int(n_lines):]]
    except Exception:
        return []


def request_cancel() -> None:
    """Set the cancel flag; the running build polls for this."""
    try:
        _cancel_flag_path().write_text(str(time.time()))
    except Exception:
        pass


# ─── Build ─────────────────────────────────────────────────────────────────
def build(
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Read all labeled frames from SQLite and write the memmap cache.

    Always writes a fresh cache from scratch — does not try to incrementally
    update. Cheaper to recompute than to reason about partial invalidation.
    """
    started = time.time()
    cancel_check = cancel_check or (lambda: False)

    # Truncate log + cancel flag.
    try:
        _build_log_path().write_text("")
        if _cancel_flag_path().exists():
            _cancel_flag_path().unlink()
    except Exception:
        pass

    def _emit_log(line: str) -> None:
        try:
            with _build_log_path().open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        log.info(line)

    def _publish(payload: Dict[str, Any]) -> None:
        try:
            tmp = _build_progress_path().with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(_build_progress_path())
        except Exception:
            pass
        if progress_callback:
            try: progress_callback(payload)
            except Exception: pass

    _emit_log(f"[cache] start — building memmap cache at {_cache_dir()}")
    _publish({"status": "running", "phase": "scanning",
              "scanned": 0, "total": 0, "ts": time.time(), "started_at": started})

    # 1) List labeled frame ids in deterministic order.
    with database.read_conn() as conn:
        rows = conn.execute(
            """SELECT DISTINCT f.id, f.game_version
               FROM frames f
               WHERE f.label_status = 'labeled'
                 AND EXISTS (SELECT 1 FROM labels l WHERE l.frame_id = f.id)
               ORDER BY f.id ASC"""
        ).fetchall()

    n_total = len(rows)
    if n_total == 0:
        msg = "no labeled frames available — label some in Module 4 first"
        _emit_log(f"[cache] ERROR {msg}")
        _publish({"status": "error", "message": msg, "ts": time.time()})
        return {"ok": False, "error": msg}

    _emit_log(f"[cache] {n_total} labeled frames will be cached (compressed v2)")

    # 2) Open the cache files for streaming write.
    # v2 layout — variable-length JPEGs and PNGs with offset tables, plus
    # a fixed-stride det targets file:
    #   images_jpeg.bin  — concat of all frames' JPEG bytes
    #   images_jpeg.idx  — uint64 array of byte offsets, length n_kept+1
    #   masks_png.bin    — concat of all frames' PNG bytes
    #   masks_png.idx    — uint64 array of byte offsets, length n_kept+1
    #   det_targets.bin  — fixed-stride float32 (4032 bytes per frame)
    # Total ~5 GB for 72k frames vs the legacy 43 GB raw layout.
    det_target_bytes_per_frame = DET_GRID_H * DET_GRID_W * (5 + NUM_DET_CLASSES) * 4
    _emit_log(f"[cache] target layout: jpeg+idx + png+idx + det_targets "
              f"({det_target_bytes_per_frame} bytes/frame for det)")

    images_f = open(_images_jpeg_path(), "wb", buffering=8 * 1024 * 1024)
    masks_f  = open(_masks_png_path(),   "wb", buffering=8 * 1024 * 1024)
    # Det target file is fixed-size and small, can pre-allocate.
    det_total_bytes = n_total * det_target_bytes_per_frame
    with open(_det_targets_path(), "wb") as f:
        if det_total_bytes > 0:
            f.seek(det_total_bytes - 1)
            f.write(b"\0")
    det_f = open(_det_targets_path(), "r+b", buffering=8 * 1024 * 1024)

    # Offset tables — built up as we write each chunk's results.
    # Each idx file ends up as length n_kept+1 (the last entry is the total
    # file size, so file_i can be read as bytes[idx[i]:idx[i+1]]).
    img_offsets: List[int] = [0]
    mask_offsets: List[int] = [0]

    meta_entries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    flush_every = 5000          # flush + fsync every N rows. We have RAM headroom.

    # Decide on worker count. Override via FORZATEK_CACHE_WORKERS env var if set.
    try:
        env_workers = int(os.environ.get("FORZATEK_CACHE_WORKERS", "0"))
    except ValueError:
        env_workers = 0
    n_workers = env_workers if env_workers > 0 else _default_worker_count()
    _emit_log(f"[cache] decode workers: {n_workers}")

    # Pack the work list. Smaller chunks = better backpressure granularity.
    # 25 frames per chunk × ~590 KB = ~15 MB per result. With backpressure
    # capping us at 2*n_workers in-flight, peak in-flight memory is bounded
    # to ~240 MB at n_workers=8 — well under any RAM pressure threshold.
    full_work: List[Tuple[int, int, str]] = [
        (i, int(row["id"]), row["game_version"] or "_unknown")
        for i, row in enumerate(rows)
    ]
    JOB_CHUNK = 25
    job_chunks: List[List[Tuple[int, int, str]]] = [
        full_work[i:i + JOB_CHUNK] for i in range(0, len(full_work), JOB_CHUNK)
    ]
    total_chunks = len(job_chunks)

    # Backpressure cap. Keep at most this many chunks queued in the
    # ProcessPoolExecutor at once. Workers idle briefly between chunks
    # rather than racing ahead and filling RAM with unconsumed results.
    in_flight_cap = max(2, n_workers * 2)

    received = 0       # frames returned from workers (success or fail)
    cancelled = False

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            next_chunk_idx = 0
            in_flight = {}       # future -> (chunk_idx, n_frames_in_chunk)

            # Prime: submit up to in_flight_cap chunks to fill the worker pool.
            while next_chunk_idx < total_chunks and len(in_flight) < in_flight_cap:
                chunk = job_chunks[next_chunk_idx]
                fut = pool.submit(_worker_decode_chunk, chunk)
                in_flight[fut] = (next_chunk_idx, len(chunk))
                next_chunk_idx += 1

            # Main consumption loop. Each iteration waits for *any* one future
            # to complete, processes it, then submits one more chunk. This
            # keeps exactly in_flight_cap chunks queued at all times.
            while in_flight:
                # Cancellation check between completions.
                if cancel_check() or _cancel_flag_path().exists():
                    if not cancelled:
                        _emit_log("[cache] cancelled by user — draining workers")
                        cancelled = True

                # Wait for any one future to finish. We can't use as_completed
                # here directly because we need to mutate `in_flight` and add
                # new futures dynamically. concurrent.futures.wait with
                # FIRST_COMPLETED gives us exactly that.
                from concurrent.futures import wait, FIRST_COMPLETED
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED,
                               timeout=1.0)
                if not done:
                    # Timed out without any completion — loop and re-check
                    # cancellation. Keeps cancel responsive even if a worker
                    # is mid-decode on a slow frame.
                    continue

                for fut in done:
                    chunk_idx, n_in_chunk = in_flight.pop(fut)

                    if cancelled:
                        # Drop the result without processing.
                        received += n_in_chunk
                        try: fut.result()
                        except Exception: pass
                        continue

                    try:
                        results = fut.result()
                    except Exception as e:
                        _emit_log(f"[cache] WARN worker chunk {chunk_idx} crashed: {e}")
                        received += n_in_chunk
                        continue

                    for r in results:
                        received += 1
                        if "error" in r:
                            skipped.append({"frame_id": r["frame_id"],
                                            "reason":   r["error"]})
                            continue

                        kept_idx = len(meta_entries)
                        # Append JPEG to images_jpeg.bin and update offset.
                        # We're writing sequentially so no seek needed; the
                        # 8 MB buffered file handle batches small writes.
                        images_f.write(r["jpeg_bytes"])
                        img_offsets.append(img_offsets[-1] + len(r["jpeg_bytes"]))
                        # Same for the PNG mask.
                        masks_f.write(r["png_bytes"])
                        mask_offsets.append(mask_offsets[-1] + len(r["png_bytes"]))
                        # Det targets are fixed-size, write at known offset.
                        det_f.seek(kept_idx * det_target_bytes_per_frame)
                        det_f.write(r["det_bytes"])

                        meta_entries.append({
                            "frame_id":     r["frame_id"],
                            "game_version": r["game_version"],
                        })

                    # Publish progress after each chunk completion.
                    _publish({
                        "status":     "running",
                        "phase":      "encoding (parallel, compressed)",
                        "scanned":    received,
                        "total":      n_total,
                        "kept":       len(meta_entries),
                        "skipped":    len(skipped),
                        "started_at": started,
                        "ts":         time.time(),
                        "workers":    n_workers,
                        "in_flight":  len(in_flight),
                    })

                    # Periodic flush to release dirty pages back to disk.
                    if received % flush_every < JOB_CHUNK:
                        images_f.flush()
                        masks_f.flush()
                        det_f.flush()
                        try:
                            os.fsync(images_f.fileno())
                            os.fsync(masks_f.fileno())
                            os.fsync(det_f.fileno())
                        except OSError:
                            pass

                    # Backpressure refill: each completed chunk earns one
                    # new submission (until we run out of work). This is
                    # what keeps in-flight bounded.
                    if not cancelled and next_chunk_idx < total_chunks:
                        chunk = job_chunks[next_chunk_idx]
                        new_fut = pool.submit(_worker_decode_chunk, chunk)
                        in_flight[new_fut] = (next_chunk_idx, len(chunk))
                        next_chunk_idx += 1

        if cancelled:
            raise KeyboardInterrupt("cancelled")

        # 3) Final flush + close.
        images_f.flush(); masks_f.flush(); det_f.flush()
        try:
            os.fsync(images_f.fileno())
            os.fsync(masks_f.fileno())
            os.fsync(det_f.fileno())
        except OSError:
            pass
        images_f.close(); masks_f.close(); det_f.close()
        images_f = masks_f = det_f = None

        n_kept = len(meta_entries)

        # Truncate det_targets.bin if some frames were skipped.
        if n_kept != n_total:
            _emit_log(f"[cache] kept {n_kept}/{n_total} (skipped {len(skipped)}); finalizing…")
            with open(_det_targets_path(), "r+b") as f:
                f.truncate(n_kept * det_target_bytes_per_frame)

        # Write the offset index files for variable-length payloads.
        # Each idx file is a uint64 array of length n_kept+1, where the
        # last entry equals the total .bin file size.
        img_idx_arr  = np.asarray(img_offsets,  dtype=np.uint64)
        mask_idx_arr = np.asarray(mask_offsets, dtype=np.uint64)
        # Sanity: arrays should match n_kept+1 because we appended once per kept frame.
        if len(img_idx_arr) != n_kept + 1:
            raise RuntimeError(
                f"img offsets length mismatch: {len(img_idx_arr)} vs n_kept+1={n_kept+1}")
        if len(mask_idx_arr) != n_kept + 1:
            raise RuntimeError(
                f"mask offsets length mismatch: {len(mask_idx_arr)} vs n_kept+1={n_kept+1}")
        _images_jpeg_idx_path().write_bytes(img_idx_arr.tobytes())
        _masks_png_idx_path().write_bytes(mask_idx_arr.tobytes())

        # 4) Write meta.json.
        total_compressed = (
            _images_jpeg_path().stat().st_size +
            _masks_png_path().stat().st_size +
            _det_targets_path().stat().st_size +
            _images_jpeg_idx_path().stat().st_size +
            _masks_png_idx_path().stat().st_size
        )
        meta_doc = {
            "version":         2,
            "format":          FORMAT_V2_COMPRESSED,
            "n_frames":        n_kept,
            "input_h":         INPUT_H,
            "input_w":         INPUT_W,
            "det_grid_h":      DET_GRID_H,
            "det_grid_w":      DET_GRID_W,
            "det_channels":    5 + NUM_DET_CLASSES,
            "jpeg_quality":    _CACHE_JPEG_QUALITY,
            "built_at":        time.time(),
            "build_seconds":   time.time() - started,
            "db_fingerprint":  _db_fingerprint(),
            "skipped":         skipped,
            "entries":         meta_entries,
        }
        _meta_path().write_text(json.dumps(meta_doc))

        elapsed = time.time() - started
        _emit_log(f"[cache] done — {n_kept} frames in {elapsed:.1f}s "
                  f"({total_compressed/1e9:.2f} GB total compressed)")
        _publish({
            "status":         "done",
            "kept":           n_kept,
            "skipped":        len(skipped),
            "total":          n_total,
            "elapsed_sec":    elapsed,
            "size_bytes":     int(total_compressed),
            "started_at":     started,
            "finished_at":    time.time(),
            "ts":             time.time(),
        })
        return {"ok": True, "n_frames": n_kept, "skipped": len(skipped),
                "elapsed_sec": elapsed}

    except KeyboardInterrupt:
        # Best-effort cleanup of partial files so a future build starts fresh.
        for fh in (images_f, masks_f, det_f):
            try:
                if fh is not None:
                    fh.close()
            except Exception:
                pass
        for p in (_images_jpeg_path(), _images_jpeg_idx_path(),
                  _masks_png_path(), _masks_png_idx_path(),
                  _det_targets_path(), _meta_path()):
            try:
                if p.exists(): p.unlink()
            except Exception: pass
        _publish({"status": "cancelled", "ts": time.time()})
        return {"ok": False, "cancelled": True}

    except Exception as e:
        log.exception("cache build crashed: %s", e)
        for fh in (images_f, masks_f, det_f):
            try:
                if fh is not None:
                    fh.close()
            except Exception:
                pass
        _emit_log(f"[cache] ERROR {e}")
        _publish({"status": "error", "message": str(e), "ts": time.time()})
        return {"ok": False, "error": str(e)}


def _process_one_frame(frame_id: int) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Decode one frame into (image_uint8, mask_uint8, det_boxes_list).

    Image: BGR uint8, INPUT_H × INPUT_W × 3, HUD-mask applied if available.
    Mask:  uint8, INPUT_H × INPUT_W, class ids 0–3 + 255.
    Boxes: list of normalized {cls,x,y,w,h,confidence} dicts.
    """
    with database.read_conn() as conn:
        frame = conn.execute(
            "SELECT id, frame_jpeg, game_version, width, height "
            "FROM frames WHERE id = ?",
            (frame_id,),
        ).fetchone()
        if frame is None:
            raise ValueError(f"frame {frame_id} not in DB")
        label_rows = conn.execute(
            "SELECT task, data_json FROM labels "
            "WHERE frame_id = ? ORDER BY id ASC",
            (frame_id,),
        ).fetchall()

    seg_payload: Optional[Dict[str, Any]] = None
    det_payload: Optional[Dict[str, Any]] = None
    for r in label_rows:
        try: payload = json.loads(r["data_json"])
        except Exception: continue
        if r["task"] == "seg":
            if payload.get("format") != EXPECTED_SEG_FORMAT:
                log.warning("frame %d seg format=%s", frame_id, payload.get("format"))
            seg_payload = payload
        elif r["task"] == "det":
            if payload.get("format") != EXPECTED_DET_FORMAT:
                log.warning("frame %d det format=%s", frame_id, payload.get("format"))
            det_payload = payload

    img = _decode_jpeg(bytes(frame["frame_jpeg"]))
    h0, w0 = img.shape[:2]
    if _hud_mask is not None and frame["game_version"]:
        img = _hud_mask.apply_mask(img, frame["game_version"])

    if seg_payload is None or "mask_png_b64" not in seg_payload:
        seg = np.full((h0, w0), IGNORE_INDEX, dtype=np.uint8)
    else:
        seg = _decode_seg_mask(seg_payload["mask_png_b64"])
        seg = _validate_seg_classes(seg)
        if seg.shape[:2] != (h0, w0):
            seg = _resize_mask(seg, w0, h0)

    img = _resize_image(img, INPUT_W, INPUT_H)
    seg = _resize_mask(seg, INPUT_W, INPUT_H)

    boxes = (det_payload or {}).get("boxes", []) or []
    return img, seg, boxes


# ─── Parallel worker (top-level for Windows spawn) ─────────────────────────
def _worker_decode_chunk(frame_specs: List[Tuple[int, int, str]]
                          ) -> List[Dict[str, Any]]:
    """Decode + compress a chunk of frames in a child process.

    Args:
        frame_specs: list of (request_idx, frame_id, game_version).

    Returns: list of dicts. On success:
        {"request_idx", "frame_id", "game_version",
         "jpeg_bytes",   # JPEG-encoded BGR image (~17:1 compression)
         "png_bytes",    # PNG-encoded uint8 seg mask (lossless, ~30:1)
         "det_bytes",    # raw bytes of the (G_h, G_w, 5+C) float32 det target
         "det_boxes"}    # original list of dicts, kept for stale-debug only
      On failure:
        {"request_idx", "frame_id", "error"}

    Why bytes everywhere: pickling bytes across the multiprocessing pipe is
    ~3x faster than pickling numpy arrays of equal size. The parent writes
    them directly into the cache .bin files with one write() call each.
    """
    # Local import in worker to avoid pickling concerns.
    from backend.perception.model import encode_det_targets

    out: List[Dict[str, Any]] = []
    for request_idx, frame_id, game_version in frame_specs:
        try:
            img_arr, mask_arr, det_boxes = _process_one_frame(int(frame_id))

            # Sanity check before encoding.
            if img_arr.shape != (INPUT_H, INPUT_W, 3) or img_arr.dtype != np.uint8:
                out.append({"request_idx": request_idx, "frame_id": frame_id,
                            "error": f"bad img shape/dtype {img_arr.shape}/{img_arr.dtype}"})
                continue
            if mask_arr.shape != (INPUT_H, INPUT_W) or mask_arr.dtype != np.uint8:
                out.append({"request_idx": request_idx, "frame_id": frame_id,
                            "error": f"bad mask shape/dtype {mask_arr.shape}/{mask_arr.dtype}"})
                continue

            jpeg_bytes = _encode_jpeg_for_cache(img_arr)
            png_bytes  = _encode_png_for_cache(mask_arr)

            # Pre-encode det targets at build time so workers don't have to
            # do it at training time. This was the source of the earlier
            # ~1.2 GB/worker memory blow-up. Always 4032 bytes (16 × 9 × 7
            # float32 = 4032) so we can read them with a fixed-stride memmap.
            det_t = encode_det_targets(det_boxes)             # (G_h, G_w, 5+C) f32
            det_bytes = det_t.contiguous().numpy().tobytes()  # 4032 bytes

            out.append({
                "request_idx":  request_idx,
                "frame_id":     int(frame_id),
                "game_version": game_version or "_unknown",
                "jpeg_bytes":   jpeg_bytes,
                "png_bytes":    png_bytes,
                "det_bytes":    det_bytes,
                "det_boxes":    det_boxes,
            })
        except Exception as e:
            out.append({
                "request_idx": request_idx,
                "frame_id":    int(frame_id),
                "error":       str(e),
            })
    return out


def _default_worker_count() -> int:
    """Cap at 8 — SQLite WAL contention caps useful parallelism around there.
    Also leave 2 cores for the main process + OS scheduling.
    """
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu - 2))


def _chunk_evenly(items: List[Any], n_chunks: int) -> List[List[Any]]:
    """Split items into ~n_chunks roughly equal-sized lists."""
    if n_chunks <= 0:
        return [items]
    n_chunks = min(n_chunks, len(items))
    if n_chunks <= 1:
        return [list(items)]
    chunk_size = (len(items) + n_chunks - 1) // n_chunks
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# ─── Cached dataset ────────────────────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_meta() -> Optional[Dict[str, Any]]:
    if not _meta_path().exists():
        return None
    try:
        return json.loads(_meta_path().read_text())
    except Exception as e:
        log.warning("load_meta failed: %s", e)
        return None


def make_splits_from_cache(
    seed: int = 0,
    val_frac: float = 0.15,
    game_versions: Optional[List[str]] = None,
) -> Tuple[List[int], List[int]]:
    """Same algorithm as dataset.make_splits but operates on cached entries.

    Returns *cache indices*, not frame_ids — those are what the cached
    dataset takes as `ids`.
    """
    meta = load_meta()
    if meta is None:
        raise RuntimeError("cache not built — call build() first")
    entries = meta["entries"]
    by_version: Dict[str, List[int]] = {}
    for idx, e in enumerate(entries):
        if game_versions and e["game_version"] not in game_versions:
            continue
        by_version.setdefault(e["game_version"], []).append(idx)

    rng = random.Random(seed)
    train_ids: List[int] = []; val_ids: List[int] = []
    for ver, ids in sorted(by_version.items()):
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * val_frac)) if len(ids) >= 4 else 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])
    rng.shuffle(train_ids); rng.shuffle(val_ids)
    return train_ids, val_ids


class CachedPerceptionDataset(Dataset):
    """Drop-in replacement for PerceptionDataset that reads from the cache.

    `ids` are *cache indices* (0 .. n_frames-1), not frame_ids — get them
    from make_splits_from_cache.

    Supports two cache formats transparently:

      v1 (FORMAT_V1_RAW): legacy fixed-stride uint8 memmap files. Reads are
          essentially free but the cache is ~43 GB for 72k frames at 512×288,
          which exceeds typical system RAM and causes disk thrashing.

      v2 (FORMAT_V2_COMPRESSED): JPEG images + PNG masks + flat det targets
          file. Cache is ~5 GB and fits easily in OS page cache. Per-sample
          cost includes a JPEG decode (~1 ms on a modern core) which overlaps
          with the GPU forward pass and so doesn't reduce throughput.

    OUTPUT FORMAT (same for both v1 and v2):
        image:  uint8 BGR tensor of shape (3, H, W).
        seg:    uint8 tensor of shape (H, W) — class ids 0..3 + 255.
        det:    pre-encoded float32 grid (G_h, G_w, 5+C) WITHOUT flip applied.

    All augmentation runs on the GPU in train.py's _GPUAugment.

    Each worker process re-opens the memmaps in __init__ via lazy property,
    so we don't share file handles across forks.
    """

    def __init__(
        self,
        ids: List[int],
        train: bool = True,
        augment_strength: float = 1.0,    # kept for API compatibility
    ):
        self.ids: List[int] = list(ids)
        self.train: bool = bool(train)
        self.aug: float = float(augment_strength)
        self._format: Optional[str] = None
        self._meta:   Optional[Dict[str, Any]] = None
        # v1 handles
        self._v1_images: Optional[np.memmap] = None
        self._v1_masks:  Optional[np.memmap] = None
        # v2 handles
        self._v2_jpeg_blob:  Optional[np.memmap] = None
        self._v2_png_blob:   Optional[np.memmap] = None
        self._v2_jpeg_idx:   Optional[np.ndarray] = None
        self._v2_png_idx:    Optional[np.ndarray] = None
        self._v2_det_mm:     Optional[np.memmap] = None

    def _ensure_open(self) -> None:
        if self._format is not None:
            return
        meta = load_meta()
        if meta is None:
            raise RuntimeError("cache not built — call cache.build() first")
        self._meta = meta
        self._format = meta.get("format", FORMAT_V1_RAW)
        n = int(meta["n_frames"])

        if self._format == FORMAT_V2_COMPRESSED:
            # Memmap each blob; idx files are small, we just read them in.
            jpeg_total = _images_jpeg_path().stat().st_size
            png_total  = _masks_png_path().stat().st_size
            self._v2_jpeg_blob = np.memmap(_images_jpeg_path(), dtype=np.uint8,
                                            mode="r", shape=(jpeg_total,))
            self._v2_png_blob  = np.memmap(_masks_png_path(),  dtype=np.uint8,
                                            mode="r", shape=(png_total,))
            self._v2_jpeg_idx = np.frombuffer(_images_jpeg_idx_path().read_bytes(),
                                              dtype=np.uint64)
            self._v2_png_idx  = np.frombuffer(_masks_png_idx_path().read_bytes(),
                                              dtype=np.uint64)
            det_channels = int(meta.get("det_channels", 5 + NUM_DET_CLASSES))
            det_grid_h   = int(meta.get("det_grid_h",   DET_GRID_H))
            det_grid_w   = int(meta.get("det_grid_w",   DET_GRID_W))
            self._v2_det_mm = np.memmap(_det_targets_path(), dtype=np.float32,
                                         mode="r",
                                         shape=(n, det_grid_h, det_grid_w, det_channels))
        else:
            # v1 raw uint8.
            self._v1_images = np.memmap(_images_path(), dtype=np.uint8, mode="r",
                                         shape=(n, INPUT_H, INPUT_W, 3))
            self._v1_masks  = np.memmap(_masks_path(),  dtype=np.uint8, mode="r",
                                         shape=(n, INPUT_H, INPUT_W))

    def __len__(self) -> int:
        return len(self.ids)

    def _read_v2_sample(self, cache_idx: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """Decode JPEG + PNG for one sample, plus read the det target."""
        # Image — slice the memmap to the JPEG bytes for this frame, decode.
        j0 = int(self._v2_jpeg_idx[cache_idx])
        j1 = int(self._v2_jpeg_idx[cache_idx + 1])
        jpeg_bytes = bytes(self._v2_jpeg_blob[j0:j1])
        if cv2 is not None:
            img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8),
                                cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"jpeg decode failed at cache_idx={cache_idx}")
        else:
            from PIL import Image as _PIL
            pil = _PIL.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            img = np.asarray(pil)[:, :, ::-1]
            img = np.ascontiguousarray(img)

        # Mask — same pattern, decode the PNG.
        m0 = int(self._v2_png_idx[cache_idx])
        m1 = int(self._v2_png_idx[cache_idx + 1])
        png_bytes = bytes(self._v2_png_blob[m0:m1])
        if cv2 is not None:
            seg = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            if seg is None:
                raise ValueError(f"png decode failed at cache_idx={cache_idx}")
        else:
            from PIL import Image as _PIL
            seg = np.array(_PIL.open(io.BytesIO(png_bytes)).convert("L"),
                            dtype=np.uint8)

        # Det target — direct memmap read, no decode.
        det_t = torch.from_numpy(np.array(self._v2_det_mm[cache_idx],
                                          dtype=np.float32, copy=True))
        return img, seg, det_t

    def _read_v1_sample(self, cache_idx: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """Read uint8 arrays from v1 fixed-stride memmaps; encode det on the fly."""
        img = np.array(self._v1_images[cache_idx], dtype=np.uint8, copy=True)
        seg = np.array(self._v1_masks[cache_idx],  dtype=np.uint8, copy=True)
        # v1 didn't pre-compute det targets — encode on the fly.
        boxes = self._meta["entries"][cache_idx].get("det_boxes") or []
        det_t = encode_det_targets(boxes)
        return img, seg, det_t

    def __getitem__(self, position: int) -> dict:
        self._ensure_open()
        cache_idx = self.ids[position]
        if self._format == FORMAT_V2_COMPRESSED:
            img, seg, det_t = self._read_v2_sample(cache_idx)
        else:
            img, seg, det_t = self._read_v1_sample(cache_idx)

        # uint8 BGR (3, H, W) — GPU does normalize + flip + jitter.
        img_chw = np.ascontiguousarray(img.transpose(2, 0, 1))
        img_t = torch.from_numpy(img_chw)
        seg_t = torch.from_numpy(np.ascontiguousarray(seg))

        meta_entry = self._meta["entries"][cache_idx]
        return {
            "image":    img_t,
            "seg":      seg_t,
            "det":      det_t,
            "frame_id": int(meta_entry["frame_id"]),
        }


# ─── Cleanup ───────────────────────────────────────────────────────────────
def clear() -> bool:
    """Delete all cache files. Returns True if anything was removed.

    Cleans both the legacy raw-uint8 cache (v1) and the new compressed
    cache (v2) — useful when migrating from one format to the other.
    """
    removed = False
    for p in (_images_path(), _masks_path(),                  # v1
              _images_jpeg_path(), _images_jpeg_idx_path(),   # v2 images
              _masks_png_path(), _masks_png_idx_path(),       # v2 masks
              _det_targets_path(),                            # both formats
              _meta_path(),
              _build_progress_path(), _build_log_path()):
        try:
            if p.exists():
                p.unlink()
                removed = True
        except Exception as e:
            log.warning("clear failed for %s: %s", p, e)
    return removed