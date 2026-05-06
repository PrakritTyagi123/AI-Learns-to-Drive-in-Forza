"""
Module 5 — Class definitions and model shape constants
======================================================
Single source of truth. Anything that needs to know "what does the perception
model output" imports from here. Module 4 (auto_labeler/service), Module 6
(compare/service), Module 8 (drive/runtime, ppo/env) all read this file —
never re-define these constants anywhere else.

The label schema written by Module 4 is the contract:

    seg label payload:
        {
          "mask_png_b64": "...",
          "classes": {"0": "offroad", "1": "road",
                      "2": "curb", "3": "wall", "255": "unknown"},
          "format": "seg_v1"
        }

    det label payload:
        {
          "boxes": [
            {"cls": "vehicle"|"sign", "x":..., "y":..., "w":..., "h":...,
             "confidence": ...}, ...
          ],
          "format": "det_v1"
        }

The constants below MUST stay aligned with Module 4's writers.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# ─── Input geometry ────────────────────────────────────────────────────────
# Forza HD frames are 1920x1080 (16:9). The model trains on a downscaled
# letterbox-fit. 512x288 keeps the aspect ratio exact and is divisible by
# 32 (every layer in MobileNetV3 needs that).
INPUT_W: int = 512
INPUT_H: int = 288

# ─── Segmentation classes ──────────────────────────────────────────────────
# IDs MUST match the PNG class ids written by labeling/service.py.
# 255 is the canonical "ignore" sentinel — pixels with this id are excluded
# from the seg loss. Auto-trusted labels are mostly 255 outside the road
# region (we only trust SegFormer's road; everything else is "unknown").
NUM_SEG_CLASSES: int = 4
SEG_CLASSES: List[str] = ["offroad", "road", "curb", "wall"]
SEG_CLASS_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(SEG_CLASSES)}
IGNORE_INDEX: int = 255

# Palette for overlay rendering. RGB tuples.
SEG_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0:   (160,  82,  45),   # offroad — terracotta
    1:   ( 90, 120, 175),   # road — slate-blue (matches accent)
    2:   (210, 180,  90),   # curb — dull gold
    3:   (200,  60,  60),   # wall — muted red
    255: (  0,   0,   0),   # unknown — transparent in overlays
}

# ─── Detection classes ─────────────────────────────────────────────────────
# Two classes only. We deliberately keep the head small — Forza has many
# kinds of road signs but the agent only needs "is there a sign here" and
# "is there a vehicle here" for now.
NUM_DET_CLASSES: int = 2
DET_CLASSES: List[str] = ["vehicle", "sign"]
DET_CLASS_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(DET_CLASSES)}

DET_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (110, 191, 115),     # vehicle — green (matches --ok)
    1: (212, 160,  74),     # sign — gold (matches --warn)
}

# ─── Detection grid (anchor-free, single-scale) ────────────────────────────
# We use a SSD-lite-style head on the deepest backbone stage. With INPUT_W=512,
# the backbone's last stride-32 feature map is 16x9. Each grid cell predicts:
#     [obj, x, y, w, h, cls_logits...]   = 5 + NUM_DET_CLASSES floats
# x/y are sigmoid-relative-to-cell, w/h are in normalized image coords.
DET_STRIDE: int = 32
DET_GRID_W: int = INPUT_W // DET_STRIDE   # 16
DET_GRID_H: int = INPUT_H // DET_STRIDE   # 9

# ─── Schema version sanity ─────────────────────────────────────────────────
EXPECTED_SEG_FORMAT: str = "seg_v1"
EXPECTED_DET_FORMAT: str = "det_v1"


def describe() -> dict:
    """Used by perception_stats() and tests."""
    return {
        "input_w":       INPUT_W,
        "input_h":       INPUT_H,
        "num_seg":       NUM_SEG_CLASSES,
        "num_det":       NUM_DET_CLASSES,
        "seg_classes":   list(SEG_CLASSES),
        "det_classes":   list(DET_CLASSES),
        "ignore_index":  IGNORE_INDEX,
        "det_grid":      [DET_GRID_W, DET_GRID_H],
        "seg_format":    EXPECTED_SEG_FORMAT,
        "det_format":    EXPECTED_DET_FORMAT,
    }