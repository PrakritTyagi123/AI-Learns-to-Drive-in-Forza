"""
Module 4 — Prelabeler
=====================
Wraps the two pretrained models that seed our labels:

    * YOLOv8 / YOLOv11 (Ultralytics) → vehicle + sign bounding boxes
    * SegFormer-B0/B2 (HF)           → 4-class segmentation
                                        offroad / road / curb / wall

Critical contract: the HUD mask is no longer painted onto the input frame.
It's used post-inference to filter YOLO boxes that overlap the "your car"
screen region. SegFormer is left to figure out the scene on its own.

4-CLASS MODE (this file):
    Per-pixel confidence gating, threshold 0.6 (configurable).
    - Each pixel gets the class with highest softmax prob, IF that prob is
      ≥ threshold AND the argmax class maps to one of our 4 driving classes.
    - Otherwise the pixel stays 255 (unknown / ignore in training).
    Frame-level "decisive" metric = fraction of pixels that got a real class.

Output shape — `prelabel_frame(frame_bgr, game_version)` returns:
    {
        "seg": {
            "mask_png_b64":  str,   # H×W uint8: 0..3 + 255
            "mean_entropy":  float,
            "pct_real":      float, # fraction of pixels with class 0..3
            "pct_per_class": dict,  # {"offroad": .., "road": .., "curb": .., "wall": ..}
        },
        "det": {
            "boxes": [
                {"cls": str, "x": float, "y": float, "w": float, "h": float, "confidence": float},
                ...
            ],
            "min_confidence": float,
        },
    }
"""
from __future__ import annotations

import base64
import logging
import threading
from typing import Any, Optional

import cv2
import numpy as np

from backend.hud_mask import auto_propagate  # noqa: F401  # imported for back-compat

# Class ids match Module 5's perception/classes.py:
#   0 offroad   1 road   2 curb   3 wall
SEG_CLASS_NAMES = ["offroad", "road", "curb", "wall"]
NUM_SEG_CLASSES = 4
IGNORE_INDEX    = 255

log = logging.getLogger("forzatek.labeling.prelabeler")


# ─── Errors ─────────────────────────────────────────────────────────────────
class PrelabelerUnavailable(RuntimeError):
    """Raised when a required ML library isn't installed."""


# ─── Settings access ────────────────────────────────────────────────────────
def _get_device() -> str:
    try:
        from backend import settings
        s = settings.get_settings()
        dev = (s.get("ml_device") or "auto").lower()
    except Exception:
        dev = "auto"

    if dev in ("cuda", "cpu", "mps"):
        return dev
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_yolo_threshold() -> float:
    try:
        from backend import settings
        s = settings.get_settings()
        return float(s.get("yolo_conf_threshold", 0.25))
    except Exception:
        return 0.25


def _get_pixel_confidence_threshold() -> float:
    """Per-pixel softmax probability below which a pixel is marked
    255 (unknown). Higher = stricter, fewer auto-labeled pixels.
    Override via settings.json: "seg_pixel_confidence_threshold": 0.65
    """
    try:
        from backend import settings
        return float(settings.get_settings().get("seg_pixel_confidence_threshold", 0.6))
    except Exception:
        return 0.6


def _get_min_class_coverage() -> float:
    """Frames with fewer than this fraction of pixels in real classes go
    to the queue instead of auto-trust. Default 0.30 = 30% real labels.
    """
    try:
        from backend import settings
        return float(settings.get_settings().get("seg_min_class_coverage", 0.30))
    except Exception:
        return 0.30


# ─── COCO class id → our class name mapping (YOLO side) ────────────────────
_COCO_TO_OURS = {
    2:  "vehicle",  # car
    3:  "vehicle",  # motorcycle
    5:  "vehicle",  # bus
    7:  "vehicle",  # truck
    11: "sign",     # stop sign
}


# ─── Cityscapes class id → our 4-class space ───────────────────────────────
# Cityscapes 19-class list:
#   0 road, 1 sidewalk, 2 building, 3 wall, 4 fence, 5 pole, 6 traffic-light,
#   7 traffic-sign, 8 vegetation, 9 terrain, 10 sky, 11 person, 12 rider,
#   13 car, 14 truck, 15 bus, 16 train, 17 motorcycle, 18 bicycle
#
# Mapping rationale:
#   road       → road       (drivable)
#   sidewalk   → curb       (between drivable and not)
#   building   → wall       (impassable vertical)
#   wall       → wall
#   fence      → wall
#   vegetation → offroad    (often grass / bushes — not drivable surface)
#   terrain    → offroad    (dirt, grass, sand)
#   sky        → 255        (not a surface; ignore at training time)
#   person, rider, vehicles, signs, poles, traffic lights → 255
#       (the detection head handles these; they're not seg targets)
_CITYSCAPES_TO_OURS = {
    0: 1,  # road       → road
    1: 2,  # sidewalk   → curb
    2: 3,  # building   → wall
    3: 3,  # wall       → wall
    4: 3,  # fence      → wall
    8: 0,  # vegetation → offroad
    9: 0,  # terrain    → offroad
    # All other Cityscapes ids fall through and get marked 255.
}

# Lookup table form for fast vectorized remap. Default 255 for unmapped.
_CITY_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _src, _dst in _CITYSCAPES_TO_OURS.items():
    _CITY_LUT[_src] = _dst


# ─── Lazy model loading ─────────────────────────────────────────────────────
_YOLO_MODEL: Any = None
_YOLO_LOCK = threading.Lock()
_SEG_MODEL: Any = None
_SEG_PROC: Any = None
_SEG_LOCK = threading.Lock()


def _load_yolo() -> Any:
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    with _YOLO_LOCK:
        if _YOLO_MODEL is not None:
            return _YOLO_MODEL
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise PrelabelerUnavailable(
                "ultralytics not installed. `pip install ultralytics`"
            ) from e

        device = _get_device()
        try:
            from backend import settings
            weights = settings.get_settings().get("yolo_model") or "yolo11m.pt"
        except Exception:
            weights = "yolo11m.pt"

        log.info("loading YOLO weights '%s' on device='%s' (this happens once)...",
                 weights, device)
        _YOLO_MODEL = YOLO(weights)

        try:
            _YOLO_MODEL.to(device)
            actual = next(_YOLO_MODEL.model.parameters()).device
            log.info("YOLO loaded: %s on %s", weights, actual)
        except Exception as e:
            log.warning("could not move YOLO to %s (%s); will use auto", device, e)

        return _YOLO_MODEL


def _load_seg() -> tuple[Any, Any]:
    global _SEG_MODEL, _SEG_PROC
    if _SEG_MODEL is not None and _SEG_PROC is not None:
        return _SEG_MODEL, _SEG_PROC
    with _SEG_LOCK:
        if _SEG_MODEL is not None and _SEG_PROC is not None:
            return _SEG_MODEL, _SEG_PROC
        try:
            import torch  # noqa: F401
            from transformers import (  # type: ignore
                SegformerForSemanticSegmentation,
                SegformerImageProcessor,
            )
        except ImportError as e:
            raise PrelabelerUnavailable(
                "transformers/torch not installed. "
                "`pip install transformers torch`"
            ) from e

        try:
            from backend import settings
            ckpt = (settings.get_settings().get("segformer_model")
                    or "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
        except Exception:
            ckpt = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

        device = _get_device()
        log.info("loading SegFormer '%s' on device='%s' (this happens once, ~250MB download)...",
                 ckpt, device)
        _SEG_PROC  = SegformerImageProcessor.from_pretrained(ckpt)
        _SEG_MODEL = SegformerForSemanticSegmentation.from_pretrained(ckpt)
        try:
            _SEG_MODEL = _SEG_MODEL.to(device).eval()
            actual = next(_SEG_MODEL.parameters()).device
            log.info("SegFormer loaded: %s on %s", ckpt.split("/")[-1], actual)
        except Exception as e:
            log.warning("could not move SegFormer to %s (%s); falling back to cpu", device, e)
            _SEG_MODEL = _SEG_MODEL.to("cpu").eval()
        return _SEG_MODEL, _SEG_PROC


# ─── YOLO inference (single-frame; kept for compatibility) ─────────────────
def _run_yolo(image_bgr: np.ndarray) -> dict:
    return _run_yolo_batch([image_bgr])[0]


def _run_yolo_batch(images_bgr: list) -> list:
    if not images_bgr:
        return []
    model = _load_yolo()
    conf = _get_yolo_threshold()
    device = _get_device()
    try:
        results = model.predict(
            images_bgr,
            conf=conf,
            verbose=False,
            device=device if device != "auto" else None,
        )
    except Exception as e:
        log.warning("YOLO batch predict failed (%s); returning empty", e)
        return [{"boxes": [], "min_confidence": 1.0} for _ in images_bgr]

    out_per_img = []
    for r, img in zip(results, images_bgr):
        h, w = img.shape[:2]
        boxes_out: list[dict] = []
        if r.boxes is None or len(r.boxes) == 0:
            out_per_img.append({"boxes": [], "min_confidence": 1.0})
            continue
        xywh  = r.boxes.xywh.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cids  = r.boxes.cls.cpu().numpy().astype(int)
        for (cx, cy, bw, bh), c, cid in zip(xywh, confs, cids):
            name = _COCO_TO_OURS.get(int(cid))
            if name is None:
                continue
            boxes_out.append({
                "cls":        name,
                "x":          float(cx) / w,
                "y":          float(cy) / h,
                "w":          float(bw) / w,
                "h":          float(bh) / h,
                "confidence": float(c),
            })
        min_conf = min((b["confidence"] for b in boxes_out), default=1.0)
        out_per_img.append({"boxes": boxes_out, "min_confidence": float(min_conf)})
    return out_per_img


# ─── SegFormer inference (4-class, per-pixel gating) ───────────────────────
def _run_seg(image_bgr: np.ndarray) -> dict:
    return _run_seg_batch([image_bgr])[0]


def _run_seg_batch(images_bgr: list) -> list:
    """Batched seg with per-pixel confidence gating.

    For each frame, returns:
        mask_png_b64    PNG-encoded uint8 mask, ids 0..3 + 255
        mean_entropy    Mean per-pixel entropy of the softmax (for queue priority)
        pct_real        Fraction of pixels with id 0..3 (rest are 255)
        pct_per_class   Per-class pixel fractions (offroad/road/curb/wall)
    """
    if not images_bgr:
        return []

    model, proc = _load_seg()
    import torch

    rgbs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images_bgr]
    inputs = proc(images=rgbs, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits  # (B, C_city, H/4, W/4)

    pixel_thr = _get_pixel_confidence_threshold()

    results = []
    for i, img in enumerate(images_bgr):
        h, w = img.shape[:2]
        logit_i = logits[i:i+1]
        logit_up = torch.nn.functional.interpolate(
            logit_i, size=(h, w), mode="bilinear", align_corners=False,
        )
        probs = torch.softmax(logit_up, dim=1)[0]   # (C_city, H, W)

        # argmax + max-prob per pixel.
        max_probs, pred_ids = torch.max(probs, dim=0)  # (H, W) each
        pred_ids = pred_ids.cpu().numpy().astype(np.uint8)
        max_probs = max_probs.cpu().numpy()

        # Mean entropy across all pixels — used for queue priority.
        eps = 1e-8
        entropy = -(probs * (probs.clamp(min=eps)).log()).sum(dim=0)
        mean_entropy = float(entropy.mean().cpu().item())

        # Step 1: remap Cityscapes ids → our 4 classes (or 255 if unmapped).
        mask = _CITY_LUT[pred_ids]                          # (H, W) uint8

        # Step 2: per-pixel confidence gate. Pixels below threshold → 255.
        mask = np.where(max_probs >= pixel_thr, mask, IGNORE_INDEX).astype(np.uint8)

        # Stats.
        total = mask.size
        is_real = mask < NUM_SEG_CLASSES
        pct_real = float(is_real.sum()) / total
        pct_per_class = {
            SEG_CLASS_NAMES[c]: float((mask == c).sum()) / total
            for c in range(NUM_SEG_CLASSES)
        }

        ok, buf = cv2.imencode(".png", mask)
        if not ok:
            raise RuntimeError("failed to encode seg mask as PNG")
        mask_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        results.append({
            "mask_png_b64":  mask_b64,
            "mean_entropy":  mean_entropy,
            "pct_real":      pct_real,
            "pct_per_class": pct_per_class,
        })

    return results


# ─── Public entry points ────────────────────────────────────────────────────
def prelabel_frame(
    frame_bgr: np.ndarray,
    game_version: Optional[str],
) -> dict:
    if frame_bgr is None or frame_bgr.ndim != 3:
        raise ValueError("prelabel_frame requires a 3-channel BGR image")

    seg = _run_seg(frame_bgr)
    det = _run_yolo(frame_bgr)
    det = _filter_self_boxes(det, game_version)
    return {"seg": seg, "det": det}


def prelabel_batch(
    frames_bgr: list,
    game_versions: list,
) -> list:
    if len(frames_bgr) != len(game_versions):
        raise ValueError("frames and game_versions must be same length")
    if not frames_bgr:
        return []
    for img in frames_bgr:
        if img is None or img.ndim != 3:
            raise ValueError("each frame must be a 3-channel BGR image")

    seg_list = _run_seg_batch(frames_bgr)
    det_list = _run_yolo_batch(frames_bgr)
    det_list = [
        _filter_self_boxes(d, gv)
        for d, gv in zip(det_list, game_versions)
    ]
    return [{"seg": s, "det": d} for s, d in zip(seg_list, det_list)]


# ─── YOLO box filter (self-car suppression) ─────────────────────────────────
def _get_self_overlap_threshold() -> float:
    try:
        from backend import settings
        return float(settings.get_settings().get("self_box_overlap_threshold", 0.50))
    except Exception:
        return 0.50


def _filter_self_boxes(det: dict, game_version: Optional[str]) -> dict:
    if not game_version or not det.get("boxes"):
        return det

    try:
        from backend.hud_mask import service as hud_service
        m = hud_service.get_mask(game_version)
    except Exception as e:
        log.debug("self-filter: could not fetch mask for %s: %s", game_version, e)
        return det
    if m is None or not m.get("rects"):
        return det

    rects = m["rects"]
    thr = _get_self_overlap_threshold()

    kept: list = []
    for b in det["boxes"]:
        bx1 = b["x"] - b["w"] / 2.0
        by1 = b["y"] - b["h"] / 2.0
        bx2 = b["x"] + b["w"] / 2.0
        by2 = b["y"] + b["h"] / 2.0
        b_area = max(1e-8, b["w"] * b["h"])

        max_iob = 0.0
        for r in rects:
            ix1 = max(bx1, r["x"])
            iy1 = max(by1, r["y"])
            ix2 = min(bx2, r["x"] + r["w"])
            iy2 = min(by2, r["y"] + r["h"])
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            iob = inter / b_area
            if iob > max_iob:
                max_iob = iob

        if max_iob < thr:
            kept.append(b)

    new_min_conf = min((b["confidence"] for b in kept), default=1.0)
    return {"boxes": kept, "min_confidence": float(new_min_conf)}


def is_available() -> tuple[bool, str]:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False, "ultralytics not installed"
    try:
        import transformers  # noqa: F401
        import torch         # noqa: F401
    except ImportError:
        return False, "transformers/torch not installed"
    return True, "ok"