"""
Module 6 — Compare service
==========================
Pure DB + compute logic. No Eel, no FastAPI.

Architecture notes for the fix
------------------------------
1. ALL inference runs in a dedicated worker thread (a single
   ThreadPoolExecutor with one slot). This is critical because Eel + Flask
   + gevent run on a single cooperative event loop — if we run GPU
   inference on that loop, the websocket reply to JS can't be sent until
   the GPU returns, which on a fresh load can look like "stuck forever".
2. Every public entry point has aggressive INFO-level logging at each
   stage so the terminal tells you exactly where time is going.
3. Frames get downscaled to a sane max-dim (default 1280) before going
   into SegFormer. SegFormer at 1024² is already the bottleneck; feeding
   it 4K just makes it slower with no quality benefit.
4. `_run_yolo` and `_run_ours` are wrapped in per-call timeouts so a
   single broken frame can't hang the UI.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.core import database
from backend.perception import metrics as perception_metrics
from backend.perception.classes import (
    IGNORE_INDEX,
    NUM_SEG_CLASSES,
    SEG_CLASSES,
    SEG_PALETTE,
)

log = logging.getLogger("forzatek.compare.service")

try:
    import cv2  # type: ignore
except Exception:                                  # pragma: no cover
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception:                                  # pragma: no cover
    Image = None  # type: ignore

# ─── HF cache hints ────────────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ─── Tunables ──────────────────────────────────────────────────────────────
# Downscale frames before inference. SegFormer's processor resizes to
# 1024² internally anyway, so feeding it a 4K frame is pure waste.
INFERENCE_MAX_DIM = 1280
# Per-call timeouts so a stuck inference can't freeze the UI forever.
INFERENCE_TIMEOUT_S = 60.0
# Disagreement scan pool — smaller = faster strategy switch.
DEFAULT_SCAN_POOL = 16

# ─── Single dedicated inference worker ─────────────────────────────────────
# One worker, max one in-flight call. This is the whole point of the fix —
# inference happens HERE, not on the gevent thread that Eel is using.
_inference_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="compare-inf")

# ─── Lazy holders ──────────────────────────────────────────────────────────
_perception_runtime = None        # type: ignore
_prelabeler_loaded: bool = False
_load_lock = threading.Lock()

# ─── Warmup state ──────────────────────────────────────────────────────────
_warmup_state: Dict[str, Any] = {
    "phase": "idle",
    "started": None,
    "finished": None,
    "yolo": "pending",
    "ours": "pending",
    "error": None,
}
_warmup_lock = threading.Lock()
_warmup_thread: Optional[threading.Thread] = None

STRATEGIES = ("random", "recent", "disagreement")


@dataclass
class CompareResult:
    frame_id: int
    width: int
    height: int
    game_version: Optional[str]
    gt_jpeg:    Optional[bytes]
    yolo_jpeg:  Optional[bytes]
    ours_jpeg:  Optional[bytes]
    yolo_metrics: Dict[str, Any]
    ours_metrics: Dict[str, Any]
    disagreement: float
    have_yolo: bool
    have_ours: bool


# ───────────────────────────────────────────────────────────────────────────
# Warmup
# ───────────────────────────────────────────────────────────────────────────
def _warmup_set(**kw) -> None:
    with _warmup_lock:
        _warmup_state.update(kw)


def warmup_status() -> Dict[str, Any]:
    with _warmup_lock:
        return dict(_warmup_state)


def warmup(force: bool = False) -> Dict[str, Any]:
    global _warmup_thread
    with _warmup_lock:
        phase = _warmup_state["phase"]
        if phase == "running":
            return dict(_warmup_state)
        if phase == "done" and not force:
            return dict(_warmup_state)
        _warmup_state.update({
            "phase": "running", "started": time.time(),
            "finished": None, "yolo": "pending",
            "ours": "pending", "error": None,
        })

    def _run() -> None:
        log.info("compare.warmup: starting")
        try:
            from backend.labeling import prelabeler
            ok, msg = prelabeler.is_available()
            if not ok:
                log.warning("compare.warmup: prelabeler not available: %s", msg)
                _warmup_set(yolo="error")
            else:
                t0 = time.time()
                if hasattr(prelabeler, "_load_yolo"):
                    prelabeler._load_yolo()
                if hasattr(prelabeler, "_load_seg"):
                    prelabeler._load_seg()
                global _prelabeler_loaded
                _prelabeler_loaded = True
                log.info("compare.warmup: prelabeler loaded in %.1fs",
                         time.time() - t0)
                _warmup_set(yolo="ok")
        except Exception as e:
            log.exception("compare.warmup: prelabeler load failed")
            _warmup_set(yolo="error", error=str(e))

        try:
            from backend.perception.runtime import load_active
            t0 = time.time()
            rt = load_active()
            global _perception_runtime
            _perception_runtime = rt
            if rt is None:
                log.info("compare.warmup: no active perception model")
                _warmup_set(ours="error")
            else:
                log.info("compare.warmup: perception runtime ready in %.1fs",
                         time.time() - t0)
                _warmup_set(ours="ok")
        except Exception as e:
            log.exception("compare.warmup: perception runtime load failed")
            _warmup_set(ours="error", error=str(e))

        _warmup_set(phase="done", finished=time.time())
        log.info("compare.warmup: complete")

    _warmup_thread = threading.Thread(
        target=_run, name="compare-warmup", daemon=True,
    )
    _warmup_thread.start()
    return warmup_status()


def _ensure_loaded() -> Tuple[bool, bool]:
    with _warmup_lock:
        if _warmup_state["phase"] == "done":
            return (_warmup_state["yolo"] == "ok",
                    _warmup_state["ours"] == "ok")
        running = _warmup_state["phase"] == "running"

    if running and _warmup_thread is not None:
        _warmup_thread.join(timeout=5.0)
        with _warmup_lock:
            if _warmup_state["phase"] == "done":
                return (_warmup_state["yolo"] == "ok",
                        _warmup_state["ours"] == "ok")

    with _load_lock:
        global _prelabeler_loaded, _perception_runtime
        if not _prelabeler_loaded:
            try:
                from backend.labeling import prelabeler
                ok, _ = prelabeler.is_available()
                if ok:
                    if hasattr(prelabeler, "_load_yolo"):
                        prelabeler._load_yolo()
                    if hasattr(prelabeler, "_load_seg"):
                        prelabeler._load_seg()
                    _prelabeler_loaded = True
            except Exception as e:
                log.warning("synchronous prelabeler load failed: %s", e)

        if _perception_runtime is None:
            try:
                from backend.perception.runtime import load_active
                _perception_runtime = load_active()
            except Exception as e:
                log.warning("synchronous perception runtime load failed: %s", e)

    return (_prelabeler_loaded, _perception_runtime is not None)


def reload_perception() -> bool:
    global _perception_runtime
    with _load_lock:
        try:
            from backend.perception.runtime import load_active
            _perception_runtime = load_active()
        except Exception as e:
            log.warning("reload_perception failed: %s", e)
            _perception_runtime = None
            return False
    ok = _perception_runtime is not None
    _warmup_set(ours=("ok" if ok else "error"))
    return ok


# ───────────────────────────────────────────────────────────────────────────
# DB helpers
# ───────────────────────────────────────────────────────────────────────────
def _list_labeled_ids(limit: Optional[int] = None) -> List[int]:
    sql = "SELECT id FROM frames WHERE label_status='labeled' ORDER BY id DESC"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with database.read_conn() as conn:
        rows = conn.execute(sql).fetchall()
    return [int(r["id"]) for r in rows]


def _load_frame_row(frame_id: int) -> Optional[dict]:
    with database.read_conn() as conn:
        row = conn.execute(
            "SELECT id, frame_jpeg, width, height, game_version, "
            "label_status FROM frames WHERE id = ?",
            (int(frame_id),),
        ).fetchone()
    return dict(row) if row else None


def _load_gt_for_frame(frame_id: int) -> Tuple[Optional[np.ndarray], List[dict]]:
    seg_mask: Optional[np.ndarray] = None
    boxes: List[dict] = []
    with database.read_conn() as conn:
        rows = conn.execute(
            "SELECT task, data_json FROM labels WHERE frame_id = ? "
            "ORDER BY id DESC",
            (int(frame_id),),
        ).fetchall()
    seen_seg = False
    seen_det = False
    for r in rows:
        if seen_seg and seen_det:
            break
        try:
            payload = json.loads(r["data_json"])
        except Exception:
            continue
        if r["task"] == "seg" and not seen_seg:
            b64 = payload.get("mask_png_b64")
            if b64:
                seg_mask = _decode_png_b64_mask(b64)
            seen_seg = True
        elif r["task"] == "det" and not seen_det:
            raw = payload.get("boxes") or []
            for b in raw:
                try:
                    boxes.append({
                        "cls": str(b.get("cls", "")),
                        "x": float(b["x"]), "y": float(b["y"]),
                        "w": float(b["w"]), "h": float(b["h"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            seen_det = True
    return seg_mask, boxes


# ───────────────────────────────────────────────────────────────────────────
# Decode helpers
# ───────────────────────────────────────────────────────────────────────────
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


def _decode_png_b64_mask(b64: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    if cv2 is not None:
        arr = np.frombuffer(raw, dtype=np.uint8)
        m = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if m is None:
            return None
        if m.ndim == 3:
            m = m[..., 0]
        return m.astype(np.uint8)
    if Image is None:                              # pragma: no cover
        return None
    img = Image.open(io.BytesIO(raw))
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.uint8)


def _encode_jpeg(bgr: np.ndarray, quality: int = 80) -> bytes:
    if cv2 is not None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()
    if Image is None:                              # pragma: no cover
        raise RuntimeError("Need OpenCV or Pillow to encode JPEGs.")
    rgb = bgr[:, :, ::-1]
    img = Image.fromarray(np.ascontiguousarray(rgb))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=int(quality))
    return out.getvalue()


def _maybe_downscale(bgr: np.ndarray, max_dim: int = INFERENCE_MAX_DIM
                      ) -> Tuple[np.ndarray, float]:
    """Downscale a frame so its longest side ≤ max_dim. Returns (img, scale).
    scale = (new / orig). 1.0 means no resize.
    """
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return bgr, 1.0
    s = max_dim / float(longest)
    new_w = int(round(w * s))
    new_h = int(round(h * s))
    if cv2 is not None:
        small = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        rgb = bgr[:, :, ::-1]
        small = np.asarray(
            Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)
        )[:, :, ::-1]
        small = np.ascontiguousarray(small)
    return small, s


# ───────────────────────────────────────────────────────────────────────────
# Overlay rendering
# ───────────────────────────────────────────────────────────────────────────
def _resize_mask_nearest(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    if mask.shape[1] == w and mask.shape[0] == h:
        return mask
    if cv2 is not None:
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if Image is None:                              # pragma: no cover
        return mask
    img = Image.fromarray(mask, mode="L")
    img = img.resize((w, h), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def _palette_overlay(bgr: np.ndarray, mask: np.ndarray,
                      alpha: float = 0.45) -> np.ndarray:
    h, w = bgr.shape[:2]
    if mask.shape != (h, w):
        mask = _resize_mask_nearest(mask, w, h)
    color_rgb = np.zeros_like(bgr)
    for cls_id, rgb in SEG_PALETTE.items():
        if cls_id == IGNORE_INDEX:
            continue
        sel = (mask == cls_id)
        if not sel.any():
            continue
        color_rgb[sel] = (rgb[2], rgb[1], rgb[0])
    valid = (mask != IGNORE_INDEX) & (mask < NUM_SEG_CLASSES)
    if not valid.any():
        return bgr.copy()
    out = bgr.copy()
    a = float(alpha)
    out[valid] = (out[valid].astype(np.float32) * (1 - a)
                  + color_rgb[valid].astype(np.float32) * a).astype(np.uint8)
    return out


def _draw_boxes(bgr: np.ndarray, boxes: List[dict],
                box_color: Tuple[int, int, int] = (255, 255, 255),
                with_confidence: bool = False) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    for b in boxes:
        try:
            cx, cy, bw, bh = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
        except (KeyError, TypeError, ValueError):
            continue
        x0 = int(max(0, (cx - bw / 2) * w))
        y0 = int(max(0, (cy - bh / 2) * h))
        x1 = int(min(w - 1, (cx + bw / 2) * w))
        y1 = int(min(h - 1, (cy + bh / 2) * h))
        if x1 <= x0 or y1 <= y0:
            continue
        if cv2 is not None:
            cv2.rectangle(out, (x0, y0), (x1, y1), box_color, 2)
            label = str(b.get("cls", ""))
            if with_confidence and "confidence" in b:
                label = f"{label} {float(b['confidence']):.2f}"
            if label:
                cv2.putText(out, label, (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1,
                            lineType=cv2.LINE_AA)
        else:
            out[y0:y0+2,  x0:x1] = box_color
            out[y1-2:y1,  x0:x1] = box_color
            out[y0:y1,    x0:x0+2] = box_color
            out[y0:y1,    x1-2:x1] = box_color
    return out


def _render_overlay(bgr_orig: np.ndarray, seg_mask: Optional[np.ndarray],
                    boxes: List[dict],
                    box_color: Tuple[int, int, int],
                    with_confidence: bool = False) -> bytes:
    out = bgr_orig if seg_mask is None else _palette_overlay(bgr_orig, seg_mask)
    out = _draw_boxes(out, boxes, box_color, with_confidence=with_confidence)
    return _encode_jpeg(out, quality=82)


# ───────────────────────────────────────────────────────────────────────────
# Inference helpers — run on the small inference frame; map results back
# to the original frame size when needed.
# ───────────────────────────────────────────────────────────────────────────
def _run_yolo(bgr_small: np.ndarray, game_version: Optional[str]
              ) -> Optional[dict]:
    try:
        from backend.labeling import prelabeler
        t0 = time.time()
        out = prelabeler.prelabel_frame(bgr_small, game_version)
        log.info("compare: yolo+segformer infer took %.2fs", time.time() - t0)
    except Exception as e:
        log.exception("prelabel_frame failed: %s", e)
        return None

    seg_b64 = (out.get("seg") or {}).get("mask_png_b64")
    seg_map = _decode_png_b64_mask(seg_b64) if seg_b64 else None
    if seg_map is not None and seg_map.shape != bgr_small.shape[:2]:
        seg_map = _resize_mask_nearest(seg_map, bgr_small.shape[1], bgr_small.shape[0])

    boxes = list((out.get("det") or {}).get("boxes") or [])
    return {"seg_map": seg_map, "boxes": boxes}


def _run_ours(bgr_small: np.ndarray, game_version: Optional[str]
              ) -> Optional[dict]:
    rt = _perception_runtime
    if rt is None:
        return None
    try:
        t0 = time.time()
        out = rt.infer(bgr_small, game_version=game_version)
        log.info("compare: ours infer took %.2fs", time.time() - t0)
    except Exception as e:
        log.exception("perception.infer failed: %s", e)
        return None
    return {
        "seg_map": np.asarray(out["seg_map"], dtype=np.uint8),
        "boxes":   list(out.get("boxes") or []),
    }


def _run_in_worker(fn, *args, timeout: float = INFERENCE_TIMEOUT_S,
                    label: str = "infer"):
    """Submit fn to the dedicated inference thread with a hard timeout.

    Returns the result or None on timeout/exception (which it also logs).
    The whole point: by running fn in a non-gevent thread, the gevent event
    loop is free to send our reply back to the websocket the moment we're
    done. Without this, the websocket reply waits until inference returns.
    """
    fut = _inference_pool.submit(fn, *args)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout:
        log.error("compare: %s timed out after %.0fs", label, timeout)
        return None
    except Exception as e:
        log.exception("compare: %s crashed: %s", label, e)
        return None


# ───────────────────────────────────────────────────────────────────────────
# Metric computation per model
# ───────────────────────────────────────────────────────────────────────────
def _seg_metrics_one(pred_mask: Optional[np.ndarray],
                      gt_mask: Optional[np.ndarray]) -> Dict[str, float]:
    if pred_mask is None or gt_mask is None:
        return {}
    if pred_mask.shape != gt_mask.shape:
        pred_mask = _resize_mask_nearest(pred_mask, gt_mask.shape[1], gt_mask.shape[0])
    cm = perception_metrics.confusion_matrix(
        pred_mask, gt_mask, NUM_SEG_CLASSES, IGNORE_INDEX,
    )
    sm = perception_metrics.seg_metrics(cm, SEG_CLASSES)
    sm["_n_pixels"] = int((gt_mask != IGNORE_INDEX).sum())
    return sm


def _det_metrics_one(pred_boxes: List[dict],
                      gt_boxes: List[dict]) -> Dict[str, Any]:
    return perception_metrics.detection_pr(
        [pred_boxes], [gt_boxes], iou_thresh=0.5,
    )


def _disagreement(yolo_out: Optional[dict],
                   ours_out: Optional[dict]) -> float:
    if yolo_out is None or ours_out is None:
        return 0.0
    seg_a, seg_b = yolo_out.get("seg_map"), ours_out.get("seg_map")
    seg_disagree = 0.0
    if seg_a is not None and seg_b is not None:
        if seg_a.shape != seg_b.shape:
            seg_a = _resize_mask_nearest(seg_a, seg_b.shape[1], seg_b.shape[0])
        valid = (seg_a != IGNORE_INDEX) | (seg_b != IGNORE_INDEX)
        denom = int(valid.sum())
        if denom > 0:
            seg_disagree = float((seg_a != seg_b).astype(np.bool_)[valid].sum() / denom)
    box_a = len(yolo_out.get("boxes") or [])
    box_b = len(ours_out.get("boxes") or [])
    box_disagree = abs(box_a - box_b) / max(1, max(box_a, box_b))
    return float(0.7 * seg_disagree + 0.3 * box_disagree)


# ───────────────────────────────────────────────────────────────────────────
# The actual compare body — runs in the inference worker thread.
# ───────────────────────────────────────────────────────────────────────────
def _compare_frame_body(frame_id: int, render_overlays: bool
                         ) -> Optional[CompareResult]:
    t_start = time.time()

    row = _load_frame_row(frame_id)
    if row is None:
        log.warning("compare: frame %d not found", frame_id)
        return None
    if row.get("label_status") != "labeled":
        log.warning("compare: frame %d is not labeled (status=%s)",
                    frame_id, row.get("label_status"))
        return None

    bgr = _decode_jpeg(bytes(row["frame_jpeg"]))
    h, w = bgr.shape[:2]
    width = int(row.get("width") or w)
    height = int(row.get("height") or h)
    game_version = row.get("game_version")
    log.info("compare: frame %d %dx%d gv=%s — decoded in %.2fs",
             frame_id, width, height, game_version, time.time() - t_start)

    t = time.time()
    gt_mask, gt_boxes = _load_gt_for_frame(frame_id)
    log.info("compare: gt loaded in %.2fs (seg=%s boxes=%d)",
             time.time() - t, gt_mask is not None, len(gt_boxes))

    yolo_ok, ours_ok = _ensure_loaded()

    # Downscale ONCE — both models infer on the small copy.
    bgr_small, scale = _maybe_downscale(bgr, INFERENCE_MAX_DIM)
    if scale < 1.0:
        log.info("compare: downscaled %dx%d -> %dx%d (s=%.3f) for inference",
                 w, h, bgr_small.shape[1], bgr_small.shape[0], scale)

    yolo_out: Optional[dict] = None
    if yolo_ok:
        yolo_out = _run_yolo(bgr_small, game_version)

    ours_out: Optional[dict] = None
    if ours_ok:
        ours_out = _run_ours(bgr_small, game_version)

    # Upscale predicted seg maps back to the original frame size so they
    # match the GT we'll compare against.
    if yolo_out is not None and yolo_out.get("seg_map") is not None:
        if yolo_out["seg_map"].shape != (h, w):
            yolo_out["seg_map"] = _resize_mask_nearest(yolo_out["seg_map"], w, h)
    if ours_out is not None and ours_out.get("seg_map") is not None:
        if ours_out["seg_map"].shape != (h, w):
            ours_out["seg_map"] = _resize_mask_nearest(ours_out["seg_map"], w, h)

    # Metrics
    t = time.time()
    yolo_metrics: Dict[str, Any] = {}
    if yolo_out is not None:
        yolo_metrics = {
            "seg": _seg_metrics_one(yolo_out.get("seg_map"), gt_mask),
            "det": _det_metrics_one(yolo_out.get("boxes") or [], gt_boxes),
            "n_boxes": len(yolo_out.get("boxes") or []),
        }
    ours_metrics: Dict[str, Any] = {}
    if ours_out is not None:
        ours_metrics = {
            "seg": _seg_metrics_one(ours_out.get("seg_map"), gt_mask),
            "det": _det_metrics_one(ours_out.get("boxes") or [], gt_boxes),
            "n_boxes": len(ours_out.get("boxes") or []),
        }
    log.info("compare: metrics computed in %.2fs", time.time() - t)

    # Overlays
    gt_jpeg = yolo_jpeg = ours_jpeg = None
    if render_overlays:
        t = time.time()
        try:
            gt_jpeg = _render_overlay(bgr, gt_mask, gt_boxes,
                                       box_color=(120, 220, 120))
        except Exception as e:
            log.warning("frame %d gt overlay failed: %s", frame_id, e)
        if yolo_out is not None:
            try:
                yolo_jpeg = _render_overlay(
                    bgr, yolo_out.get("seg_map"),
                    yolo_out.get("boxes") or [],
                    box_color=(80, 160, 230), with_confidence=True,
                )
            except Exception as e:
                log.warning("frame %d yolo overlay failed: %s", frame_id, e)
        if ours_out is not None:
            try:
                ours_jpeg = _render_overlay(
                    bgr, ours_out.get("seg_map"),
                    ours_out.get("boxes") or [],
                    box_color=(230, 200, 80), with_confidence=True,
                )
            except Exception as e:
                log.warning("frame %d ours overlay failed: %s", frame_id, e)
        log.info("compare: overlays rendered in %.2fs", time.time() - t)

    log.info("compare: frame %d total %.2fs", frame_id, time.time() - t_start)

    return CompareResult(
        frame_id=int(frame_id),
        width=width, height=height,
        game_version=game_version,
        gt_jpeg=gt_jpeg, yolo_jpeg=yolo_jpeg, ours_jpeg=ours_jpeg,
        yolo_metrics=yolo_metrics, ours_metrics=ours_metrics,
        disagreement=_disagreement(yolo_out, ours_out),
        have_yolo=yolo_out is not None,
        have_ours=ours_out is not None,
    )


def compare_frame(frame_id: int, render_overlays: bool = True
                  ) -> Optional[CompareResult]:
    """Public entry point. Submits the work to the inference worker thread
    so the calling gevent thread can yield the websocket reply promptly.
    """
    log.info("compare: compare_frame(%d, overlays=%s) submitted",
             frame_id, render_overlays)
    return _run_in_worker(
        _compare_frame_body, frame_id, render_overlays,
        label=f"compare_frame({frame_id})",
    )


# ───────────────────────────────────────────────────────────────────────────
# Frame picker
# ───────────────────────────────────────────────────────────────────────────
def _disagreement_pick_body(pool: List[int]) -> Optional[int]:
    best_id: Optional[int] = None
    best_score = -1.0
    for fid in pool:
        row = _load_frame_row(fid)
        if row is None:
            continue
        try:
            bgr = _decode_jpeg(bytes(row["frame_jpeg"]))
        except Exception:
            continue
        bgr_small, _ = _maybe_downscale(bgr, INFERENCE_MAX_DIM)
        gv = row.get("game_version")
        y = _run_yolo(bgr_small, gv)
        o = _run_ours(bgr_small, gv)
        s = _disagreement(y, o)
        if s > best_score:
            best_score = s
            best_id = fid
    return best_id


def pick_next_frame(strategy: str = "random",
                     exclude: Optional[List[int]] = None,
                     scan_pool: int = DEFAULT_SCAN_POOL) -> Optional[int]:
    if strategy not in STRATEGIES:
        strategy = "random"

    log.info("compare: pick_next_frame strategy=%s exclude=%d",
             strategy, len(exclude or []))

    excluded = set(int(i) for i in (exclude or []))
    ids = [i for i in _list_labeled_ids(limit=None) if i not in excluded]
    if not ids:
        log.warning("compare: no labeled frames")
        return None

    if strategy == "recent":
        return ids[0]
    if strategy == "random":
        return random.choice(ids)

    # Disagreement — costly. Run scan in the worker thread with a long timeout.
    pool = ids[:max(1, int(scan_pool))]
    yolo_ok, ours_ok = _ensure_loaded()
    if not (yolo_ok and ours_ok):
        log.info("compare: disagreement strategy needs both models; "
                 "falling back to random")
        return random.choice(ids)

    timeout = max(60.0, INFERENCE_TIMEOUT_S * 2)
    picked = _run_in_worker(_disagreement_pick_body, pool,
                             timeout=timeout, label="disagreement_pick")
    return picked if picked is not None else random.choice(ids)


# ───────────────────────────────────────────────────────────────────────────
# Summary
# ───────────────────────────────────────────────────────────────────────────
def _summary_body(n: int, cancel_check) -> Dict[str, Any]:
    started = time.time()
    yolo_ok, ours_ok = _ensure_loaded()

    ids = _list_labeled_ids(limit=int(n))
    if not ids:
        return {"n_frames": 0, "n_skipped": 0,
                "have_yolo": yolo_ok, "have_ours": ours_ok,
                "rows": [], "elapsed_s": 0.0,
                "message": "no labeled frames"}

    seg_acc = {
        "yolo": {f"iou_{c}": [] for c in SEG_CLASSES},
        "ours": {f"iou_{c}": [] for c in SEG_CLASSES},
    }
    seg_acc["yolo"]["miou"] = []
    seg_acc["ours"]["miou"] = []
    seg_acc["yolo"]["pixel_acc"] = []
    seg_acc["ours"]["pixel_acc"] = []

    det_acc: Dict[str, Dict[str, Dict[str, list]]] = {"yolo": {}, "ours": {}}
    n_processed = 0
    n_skipped = 0

    for fid in ids:
        if cancel_check():
            break
        try:
            # Bypass the worker pool here — we're already running inside
            # the worker. Call the body directly.
            res = _compare_frame_body(int(fid), render_overlays=False)
        except Exception as e:
            log.warning("summary: frame %d errored: %s", fid, e)
            n_skipped += 1
            continue
        if res is None:
            n_skipped += 1
            continue
        n_processed += 1

        for tag, m in (("yolo", res.yolo_metrics), ("ours", res.ours_metrics)):
            sm = m.get("seg") or {}
            for k in list(seg_acc[tag].keys()):
                v = sm.get(k)
                if v is None:
                    continue
                if isinstance(v, float) and np.isfinite(v):
                    seg_acc[tag][k].append(float(v))

            dm = m.get("det") or {}
            for cls_name, vals in dm.items():
                if cls_name == "mAP" or not isinstance(vals, dict):
                    continue
                if vals.get("n_gt", 0) <= 0 and vals.get("n_pred", 0) <= 0:
                    continue
                bucket = det_acc[tag].setdefault(cls_name, {
                    "precision": [], "recall": [], "ap": [],
                })
                for k in ("precision", "recall", "ap"):
                    v = vals.get(k)
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        bucket[k].append(float(v))

    rows: List[Dict[str, Any]] = []

    def _add(metric: str, yolo_vals: List[float], ours_vals: List[float],
             higher_is_better: bool = True) -> None:
        if not yolo_vals and not ours_vals:
            return
        y_mean = float(np.mean(yolo_vals)) if yolo_vals else float("nan")
        o_mean = float(np.mean(ours_vals)) if ours_vals else float("nan")
        winner = _decide_winner(y_mean, o_mean, higher_is_better)
        rows.append({
            "metric": metric, "yolo": y_mean, "ours": o_mean,
            "winner": winner,
            "n_yolo": len(yolo_vals), "n_ours": len(ours_vals),
            "n": min(len(yolo_vals), len(ours_vals)),
        })

    for c in SEG_CLASSES:
        _add(f"{c} IoU",
             seg_acc["yolo"].get(f"iou_{c}", []),
             seg_acc["ours"].get(f"iou_{c}", []))
    _add("mean IoU", seg_acc["yolo"]["miou"], seg_acc["ours"]["miou"])
    _add("pixel acc", seg_acc["yolo"]["pixel_acc"], seg_acc["ours"]["pixel_acc"])

    cls_names = sorted(set(det_acc["yolo"].keys()) | set(det_acc["ours"].keys()))
    for cls in cls_names:
        for k in ("precision", "recall", "ap"):
            _add(f"{cls} {k}",
                 det_acc["yolo"].get(cls, {}).get(k, []),
                 det_acc["ours"].get(cls, {}).get(k, []))

    return {
        "n_frames":  int(n_processed),
        "n_skipped": int(n_skipped),
        "have_yolo": yolo_ok,
        "have_ours": ours_ok,
        "rows":      rows,
        "elapsed_s": round(time.time() - started, 2),
    }


def summary(n: int = 50, cancel_check=None) -> Dict[str, Any]:
    cancel_check = cancel_check or (lambda: False)
    timeout = max(120.0, n * 5.0)
    log.info("compare: summary(n=%d, timeout=%.0fs) submitted", n, timeout)
    result = _run_in_worker(
        _summary_body, int(n), cancel_check,
        timeout=timeout, label=f"summary({n})",
    )
    if result is None:
        return {"n_frames": 0, "n_skipped": 0,
                "have_yolo": False, "have_ours": False,
                "rows": [], "elapsed_s": 0.0,
                "message": "summary timed out or failed"}
    return result


def _decide_winner(y: float, o: float,
                    higher_is_better: bool, eps: float = 1e-3) -> str:
    yf = np.isfinite(y); of = np.isfinite(o)
    if not yf and not of:
        return "n/a"
    if not yf:
        return "ours"
    if not of:
        return "yolo"
    if abs(y - o) < eps:
        return "tie"
    if higher_is_better:
        return "ours" if o > y else "yolo"
    return "ours" if o < y else "yolo"


# ───────────────────────────────────────────────────────────────────────────
# Status — never blocks, never triggers loads.
# ───────────────────────────────────────────────────────────────────────────
def status() -> Dict[str, Any]:
    n_labeled = len(_list_labeled_ids(limit=None))
    ws = warmup_status()
    return {
        "n_labeled":    n_labeled,
        "have_yolo":    (ws["yolo"] == "ok"),
        "have_ours":    (ws["ours"] == "ok"),
        "warmup":       ws,
        "strategies":   list(STRATEGIES),
    }