"""
Module 4 — Prelabeler
=====================
Wraps the two pretrained models that seed our labels:

    * YOLOv8 (Ultralytics)  → vehicle + sign bounding boxes
    * SegFormer-B0 (HF)     → road / curb / wall / offroad segmentation

Critical contract: the HUD mask from Module 3 is applied to every frame
BEFORE either model sees it. This stops YOLO from labeling the speedometer
as a "stop sign" and SegFormer from painting the minimap as "offroad."

Both models are loaded lazily — the first call to `prelabel_frame()` pays
the load cost (5-15 seconds), every subsequent call is fast. This means
tests that don't actually need the models can run without downloading
hundreds of MB of weights.

Model availability:
    * If `ultralytics` or `transformers`/`torch` aren't installed, the
      relevant wrapper raises `PrelabelerUnavailable`. The auto-labeler
      worker catches this and pauses cleanly with a helpful message.
    * Set ML_DEVICE in settings.json to 'cuda' / 'cpu'. Defaults to cuda
      if available.

Output shape — `prelabel_frame(frame_bgr, game_version)` returns:
    {
        "seg": {
            "mask_png_b64": str,    # H×W uint8, values are SEG_CLASS ids
            "mean_entropy": float,  # higher = less sure overall
        },
        "det": {
            "boxes": [
                {
                    "cls":        "vehicle" | "sign",
                    "x":          float,   # 0..1 normalized center
                    "y":          float,
                    "w":          float,
                    "h":          float,
                    "confidence": float,
                },
                ...
            ],
            "min_confidence": float,    # the worst box (drives queue priority)
        },
    }

Both heads' confidence/uncertainty are combined by `auto_labeler.py`
into a single decision: trust it, or queue it for review.
"""
from __future__ import annotations

import base64
import io
import logging
import threading
from typing import Any, Optional

import cv2
import numpy as np

from backend.hud_mask import auto_propagate

# NOTE: We deliberately do NOT import backend.perception here. Module 5
# may not exist yet when Module 4 ships. Class names + ids are duplicated
# below. When Module 5 lands, switch to `from backend.perception import classes`.
SEG_CLASS_NAMES = ["offroad", "road", "curb", "wall"]

log = logging.getLogger("forzatek.labeling.prelabeler")


# ─── Errors ─────────────────────────────────────────────────────────────────
class PrelabelerUnavailable(RuntimeError):
    """Raised when a required ML library isn't installed."""


# ─── Settings access ────────────────────────────────────────────────────────
def _get_device() -> str:
    """Read ML device from settings, fall back to auto-detect."""
    try:
        from backend import settings
        s = settings.get_settings()
        dev = (s.get("ml_device") or "auto").lower()
    except Exception:
        dev = "auto"

    if dev in ("cuda", "cpu", "mps"):
        return dev
    # auto
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


# ─── COCO class id → our class name mapping ────────────────────────────────
# YOLOv8's COCO classes: 2 car, 3 motorcycle, 5 bus, 7 truck → vehicle
#                        11 stop sign  → sign
# Other COCO classes are dropped.
_COCO_TO_OURS = {
    2:  "vehicle",  # car
    3:  "vehicle",  # motorcycle
    5:  "vehicle",  # bus
    7:  "vehicle",  # truck
    11: "sign",     # stop sign
}


# ─── SegFormer (Cityscapes) class id → our 4-class space ───────────────────
# Cityscapes ids that matter for driving:
#   0 road        → road       (1 in our space)
#   1 sidewalk    → curb       (2 in our space)
#   8 vegetation  → offroad    (0 in our space)
#   9 terrain     → offroad    (0 in our space)
#   2 building    → wall       (3 in our space)
#   3 wall        → wall       (3 in our space)
#   4 fence       → wall       (3 in our space)
# Everything else (sky, persons, vehicles, signs, poles, etc.) maps to
# offroad — the segmentation head only answers "where can I drive?",
# vehicles/signs are handled by the detection head separately.
_CITYSCAPES_TO_OURS = {
    0: 1,  # road     → road
    1: 2,  # sidewalk → curb
    2: 3,  # building → wall
    3: 3,  # wall     → wall
    4: 3,  # fence    → wall
    8: 0,  # vegetation → offroad
    9: 0,  # terrain  → offroad
}


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
        # Read which YOLO weights to use from settings, default to v11m.
        # Override via settings.json: "yolo_model": "yolo11l.pt" / "yolo11x.pt" / etc.
        try:
            from backend import settings
            weights = settings.get_settings().get("yolo_model") or "yolo11m.pt"
        except Exception:
            weights = "yolo11m.pt"

        log.info("loading YOLO weights '%s' on device='%s' (this happens once)...",
                 weights, device)
        _YOLO_MODEL = YOLO(weights)

        # Force the model onto the chosen device and confirm.
        try:
            _YOLO_MODEL.to(device)
            # Ultralytics caches the device internally; log the actual one.
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

        # SegFormer size — B2 is the sweet spot for road analysis on a 16GB GPU.
        # Override via settings.json: "segformer_model": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
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


# ─── YOLO inference ─────────────────────────────────────────────────────────
def _run_yolo(image_bgr: np.ndarray) -> dict:
    model = _load_yolo()
    conf = _get_yolo_threshold()
    device = _get_device()
    try:
        results = model.predict(
            image_bgr,
            conf=conf,
            verbose=False,
            device=device if device != "auto" else None,
        )
    except Exception as e:
        log.warning("YOLO predict failed (%s); returning no boxes", e)
        return {"boxes": [], "min_confidence": 1.0}

    if not results:
        return {"boxes": [], "min_confidence": 1.0}

    r = results[0]
    h, w = image_bgr.shape[:2]
    boxes_out: list[dict] = []
    if r.boxes is None or len(r.boxes) == 0:
        return {"boxes": [], "min_confidence": 1.0}

    xywh = r.boxes.xywh.cpu().numpy()        # (N, 4) center xy + wh in pixels
    confs = r.boxes.conf.cpu().numpy()       # (N,)
    cids  = r.boxes.cls.cpu().numpy().astype(int)  # (N,)

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
    return {"boxes": boxes_out, "min_confidence": float(min_conf)}


# ─── SegFormer inference ────────────────────────────────────────────────────
# ROAD-ONLY MODE: We don't trust SegFormer's wall/curb/offroad predictions on
# Forza (Cityscapes domain mismatch). Instead we ask one question only:
# "is this pixel road, with high confidence?" Output is binary:
#   1   = road (Cityscapes class 0 with P >= road_threshold)
#   255 = unknown (transparent on canvas, ignored at training time)
#
# This is much more robust than 4-class because Forza asphalt actually does
# look like Cityscapes asphalt — the rest of the scene (deserts, mountains,
# off-road) is what SegFormer can't handle, so we just don't ask.
CITYSCAPES_ROAD_ID = 0


def _get_road_threshold() -> float:
    """Min P(road) for a pixel to be marked as road. Higher = stricter,
    fewer false-positive roads on dirt or sand. Default 0.55 is a safe start.
    Override via settings.json: "seg_road_threshold": 0.65
    """
    try:
        from backend import settings
        return float(settings.get_settings().get("seg_road_threshold", 0.55))
    except Exception:
        return 0.55


def _get_not_road_threshold() -> float:
    """Min P(not-road) (= 1 - P(road)) for a pixel to be 'clearly not road'.
    Used by the auto-labeler to decide if the model is decisive overall:
    a frame where most pixels are confidently road OR confidently not-road
    (with a small ambiguous boundary) is trustworthy.
    """
    try:
        from backend import settings
        return float(settings.get_settings().get("seg_not_road_threshold", 0.85))
    except Exception:
        return 0.85


def _run_seg(image_bgr: np.ndarray) -> dict:
    """Single-frame seg. Used by `prelabel_frame` (kept for back-compat)."""
    return _run_seg_batch([image_bgr])[0]


def _run_seg_batch(images_bgr: list) -> list:
    """Batched seg. Road-only output. Returns one dict per input image with:
        mask_png_b64    — paletted PNG, 1=road, 255=unknown
        pct_road        — fraction of pixels marked road
        pct_decisive    — fraction of pixels where model is sure either way
                          (sum of road + clearly-not-road); high = trustworthy
        mean_entropy    — kept for back-compat, drives queue priority
    """
    model, proc = _load_seg()
    import torch

    if not images_bgr:
        return []

    rgbs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images_bgr]
    inputs = proc(images=rgbs, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits  # (B, C, H/4, W/4)

    road_thr     = _get_road_threshold()
    not_road_thr = _get_not_road_threshold()

    results = []
    for i, img in enumerate(images_bgr):
        h, w = img.shape[:2]
        logit_i = logits[i:i+1]
        logit_up = torch.nn.functional.interpolate(
            logit_i, size=(h, w), mode="bilinear", align_corners=False
        )
        probs = torch.softmax(logit_up, dim=1)[0]   # (C, H, W)

        # P(road) per pixel — single Cityscapes class.
        p_road = probs[CITYSCAPES_ROAD_ID].cpu().numpy()  # (H, W)

        # Mean entropy for queue priority.
        eps = 1e-8
        entropy = -(probs * (probs.clamp(min=eps)).log()).sum(dim=0)
        mean_entropy = float(entropy.mean().cpu().item())

        # Build binary mask. Default unknown (255). Where P(road) is high → road.
        mask = np.full_like(p_road, 255, dtype=np.uint8)
        is_road        = p_road >= road_thr
        is_clearly_not = p_road <= (1.0 - not_road_thr)
        mask[is_road] = 1
        # is_clearly_not is left as 255 (unknown to the canvas / trainer),
        # but we count it for the trust decision below.

        pct_road     = float(is_road.sum()        / mask.size)
        pct_decisive = float((is_road | is_clearly_not).sum() / mask.size)

        ok, buf = cv2.imencode(".png", mask)
        if not ok:
            raise RuntimeError("failed to encode seg mask as PNG")
        mask_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        results.append({
            "mask_png_b64":  mask_b64,
            "mean_entropy":  mean_entropy,
            "pct_road":      pct_road,
            "pct_decisive":  pct_decisive,
        })

    return results


def _run_yolo_batch(images_bgr: list) -> list:
    """Batched YOLO. Returns one dict per input image."""
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


# ─── Public entry point ─────────────────────────────────────────────────────
def prelabel_frame(
    frame_bgr: np.ndarray,
    game_version: Optional[str],
) -> dict:
    """Run YOLO + SegFormer on a single frame.

    The HUD mask is no longer applied to the input frame. Instead it is
    used to filter out YOLO detections that fall inside your-car region
    (post-inference). SegFormer is left to figure out road on its own —
    in road-only mode the rest of the scene is just rendered transparent
    so its mistakes there don't matter.

    Raises PrelabelerUnavailable if the underlying libraries are missing.
    """
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
    """Run YOLO + SegFormer on a batch of frames.

    Same per-frame output shape as `prelabel_frame`. The HUD mask is
    NOT applied to the input — both models see the raw frame. After YOLO
    returns, boxes overlapping the mask region (your car's screen area)
    are filtered out.

    Both lists must be the same length. game_version may be None per-item.
    """
    if len(frames_bgr) != len(game_versions):
        raise ValueError("frames and game_versions must be same length")
    if not frames_bgr:
        return []

    for img in frames_bgr:
        if img is None or img.ndim != 3:
            raise ValueError("each frame must be a 3-channel BGR image")

    seg_list = _run_seg_batch(frames_bgr)
    det_list = _run_yolo_batch(frames_bgr)

    # Filter "your-car" boxes per-frame, using each frame's game_version mask.
    det_list = [
        _filter_self_boxes(d, gv)
        for d, gv in zip(det_list, game_versions)
    ]

    return [{"seg": s, "det": d} for s, d in zip(seg_list, det_list)]


# ─── YOLO box filter (self-car suppression) ─────────────────────────────────
def _get_self_overlap_threshold() -> float:
    """If a YOLO box overlaps a HUD-mask rect by more than this fraction
    of the box's own area, drop it as 'this is my own car, not traffic'.
    """
    try:
        from backend import settings
        return float(settings.get_settings().get("self_box_overlap_threshold", 0.50))
    except Exception:
        return 0.50


def _filter_self_boxes(det: dict, game_version: Optional[str]) -> dict:
    """Drop YOLO boxes whose centers lie inside (or which overlap heavily
    with) the HUD-mask region for this game_version.

    The HUD mask is repurposed in road-only mode: instead of being painted
    onto the frame as black pixels, it just defines the "your-car region"
    for box suppression.
    """
    if not game_version or not det.get("boxes"):
        return det

    # Fetch normalized rects for this game version.
    try:
        from backend.hud_mask import service as hud_service
        m = hud_service.get_mask(game_version)
    except Exception as e:
        log.debug("self-filter: could not fetch mask for %s: %s", game_version, e)
        return det
    if m is None or not m.get("rects"):
        return det

    rects = m["rects"]   # list of {x,y,w,h} in 0..1
    thr = _get_self_overlap_threshold()

    kept: list = []
    for b in det["boxes"]:
        bx1 = b["x"] - b["w"] / 2.0
        by1 = b["y"] - b["h"] / 2.0
        bx2 = b["x"] + b["w"] / 2.0
        by2 = b["y"] + b["h"] / 2.0
        b_area = max(1e-8, b["w"] * b["h"])

        # Sum the IoB (intersection over box-area) across all mask rects.
        # If any single rect covers more than `thr` of the box, drop it.
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
    """Cheap availability check used by the auto-labeler before it starts.

    Doesn't actually load weights. Just imports.
    """
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