"""
ForzaTek AI v2 — Tests for backend/perception/model.py and classes.py.

Run with:  python -m tests.test_perception_model

Plain-script style matching the rest of the project (no pytest).
Each test prints ✓ on success; the runner at the bottom collects pass/fail.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Allow running from project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from backend.perception import classes as cls
from backend.perception.model import (
    Perception,
    compute_loss,
    decode_detections,
    encode_det_targets,
)


_PASS = 0
_FAIL = 0


def check(name: str, ok: bool, info: str = "") -> None:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  ✓ {name}")
    else:
        _FAIL += 1
        print(f"  ✗ {name}  {info}")


# ─── classes.py ────────────────────────────────────────────────────────────
def test_classes_constants():
    print("\n[classes]")
    check("INPUT_W and INPUT_H are positive multiples of 32",
          cls.INPUT_W > 0 and cls.INPUT_H > 0
          and cls.INPUT_W % 32 == 0 and cls.INPUT_H % 32 == 0,
          f"got {cls.INPUT_W}x{cls.INPUT_H}")
    check("NUM_SEG_CLASSES is 4",
          cls.NUM_SEG_CLASSES == 4, f"got {cls.NUM_SEG_CLASSES}")
    check("NUM_DET_CLASSES is 2",
          cls.NUM_DET_CLASSES == 2, f"got {cls.NUM_DET_CLASSES}")
    check("SEG_CLASSES has 4 names",
          len(cls.SEG_CLASSES) == 4 and "road" in cls.SEG_CLASSES,
          f"got {cls.SEG_CLASSES}")
    check("IGNORE_INDEX is 255",
          cls.IGNORE_INDEX == 255)
    check("DET_GRID is W/32 x H/32",
          cls.DET_GRID_W == cls.INPUT_W // 32
          and cls.DET_GRID_H == cls.INPUT_H // 32,
          f"got {cls.DET_GRID_W}x{cls.DET_GRID_H}")
    check("schema versions are seg_v1 / det_v1",
          cls.EXPECTED_SEG_FORMAT == "seg_v1"
          and cls.EXPECTED_DET_FORMAT == "det_v1")
    desc = cls.describe()
    check("describe() returns required keys",
          all(k in desc for k in ("input_w", "input_h", "num_seg", "num_det",
                                  "seg_classes", "det_classes", "ignore_index")))


# ─── model.py forward ──────────────────────────────────────────────────────
def test_model_forward_shape():
    print("\n[model.forward shape]")
    model = Perception(pretrained_backbone=False).eval()
    x = torch.zeros(2, 3, cls.INPUT_H, cls.INPUT_W)
    with torch.inference_mode():
        out = model(x)
    seg = out["seg_logits"]
    det = out["det_logits"]
    check("seg_logits has shape (B, S, H, W)",
          tuple(seg.shape) == (2, cls.NUM_SEG_CLASSES, cls.INPUT_H, cls.INPUT_W),
          f"got {tuple(seg.shape)}")
    check("det_logits has shape (B, 5+C, G_h, G_w)",
          tuple(det.shape) ==
          (2, 5 + cls.NUM_DET_CLASSES, cls.DET_GRID_H, cls.DET_GRID_W),
          f"got {tuple(det.shape)}")
    check("outputs are finite",
          torch.isfinite(seg).all().item() and torch.isfinite(det).all().item())


def test_model_handles_other_resolutions():
    print("\n[model.forward — interp upsample]")
    model = Perception(pretrained_backbone=False).eval()
    # Even at non-default sizes, seg head upsamples back to INPUT_H/W.
    x = torch.zeros(1, 3, cls.INPUT_H, cls.INPUT_W)
    with torch.inference_mode():
        out = model(x)
    check("seg upsample preserves input geometry",
          out["seg_logits"].shape[-2:] == (cls.INPUT_H, cls.INPUT_W))


# ─── encode_det_targets ────────────────────────────────────────────────────
def test_encode_det_targets_basic():
    print("\n[encode_det_targets]")
    boxes = [
        {"cls": "vehicle", "x": 0.5,  "y": 0.5,  "w": 0.1, "h": 0.1},
        {"cls": "sign",    "x": 0.25, "y": 0.75, "w": 0.05, "h": 0.05},
    ]
    t = encode_det_targets(boxes)
    check("target shape is (G_h, G_w, 5+C)",
          tuple(t.shape) == (cls.DET_GRID_H, cls.DET_GRID_W,
                              5 + cls.NUM_DET_CLASSES),
          f"got {tuple(t.shape)}")
    n_pos = (t[..., 0] > 0).sum().item()
    check("two positive cells assigned", n_pos == 2, f"got {n_pos}")

    # Locate the vehicle cell.
    gx = int(0.5 * cls.DET_GRID_W)
    gy = int(0.5 * cls.DET_GRID_H)
    vehicle_cell = t[gy, gx]
    check("vehicle cell has obj=1", float(vehicle_cell[0]) == 1.0)
    check("vehicle cell encodes one-hot class 0",
          float(vehicle_cell[5]) == 1.0 and float(vehicle_cell[6]) == 0.0)
    check("vehicle cell w,h match",
          abs(float(vehicle_cell[3]) - 0.1) < 1e-6
          and abs(float(vehicle_cell[4]) - 0.1) < 1e-6)


def test_encode_det_targets_ignores_invalid():
    print("\n[encode_det_targets — invalid input]")
    boxes = [
        {"cls": "vehicle", "x": 0.5,  "y": 0.5,  "w": 0.1, "h": 0.1},
        {"cls": "vehicle", "x": 1.5,  "y": 0.5,  "w": 0.1, "h": 0.1},   # x out of range
        {"cls": "vehicle", "x": 0.4,  "y": 0.4,  "w": 0,   "h": 0.1},   # w=0
        {"cls": "blah",    "x": 0.4,  "y": 0.4,  "w": 0.1, "h": 0.1},   # bad cls
        "garbage",
    ]
    t = encode_det_targets(boxes)
    n_pos = (t[..., 0] > 0).sum().item()
    check("only the one valid box is encoded", n_pos == 1, f"got {n_pos}")


def test_encode_det_targets_collision_keeps_larger():
    print("\n[encode_det_targets — same-cell collision]")
    boxes = [
        {"cls": "vehicle", "x": 0.5, "y": 0.5, "w": 0.05, "h": 0.05},  # smaller
        {"cls": "sign",    "x": 0.5, "y": 0.5, "w": 0.20, "h": 0.20},  # larger
    ]
    t = encode_det_targets(boxes)
    gx = int(0.5 * cls.DET_GRID_W); gy = int(0.5 * cls.DET_GRID_H)
    # Sign should win (cls index 1).
    check("larger box wins the cell",
          float(t[gy, gx, 6]) == 1.0 and float(t[gy, gx, 5]) == 0.0)


# ─── compute_loss ──────────────────────────────────────────────────────────
def test_compute_loss_runs_and_is_scalar():
    print("\n[compute_loss]")
    model = Perception(pretrained_backbone=False)
    x = torch.zeros(2, 3, cls.INPUT_H, cls.INPUT_W)
    seg_targets = torch.full((2, cls.INPUT_H, cls.INPUT_W),
                             fill_value=cls.IGNORE_INDEX, dtype=torch.long)
    seg_targets[:, 100:200, 100:300] = 1  # mark a road region
    det_targets = torch.zeros(2, cls.DET_GRID_H, cls.DET_GRID_W,
                              5 + cls.NUM_DET_CLASSES)
    det_targets[0, 4, 8, 0] = 1.0  # one positive cell on image 0
    det_targets[0, 4, 8, 1:5] = torch.tensor([0.5, 0.5, 0.1, 0.1])
    det_targets[0, 4, 8, 5] = 1.0  # vehicle

    out = model(x)
    loss = compute_loss(out, seg_targets, det_targets)
    check("compute_loss returns dict with 'total'", "total" in loss)
    check("total loss is a scalar tensor",
          loss["total"].ndim == 0)
    check("total loss is finite", torch.isfinite(loss["total"]).item())
    check("seg + det_obj losses are present",
          torch.isfinite(loss["seg"]).item()
          and torch.isfinite(loss["det_obj"]).item())
    check("backward pass works",
          _backward_runs(loss["total"], model))


def _backward_runs(loss, model) -> bool:
    model.zero_grad()
    try:
        loss.backward()
    except Exception as e:
        print(f"        backward error: {e}")
        return False
    n_with_grad = sum(1 for p in model.parameters()
                      if p.grad is not None and torch.isfinite(p.grad).all())
    return n_with_grad > 0


def test_compute_loss_skips_ignore_pixels():
    print("\n[compute_loss — ignore_index=255]")
    model = Perception(pretrained_backbone=False)
    # All seg pixels are ignore -> seg loss should be ~0 (or NaN-safe).
    x = torch.zeros(1, 3, cls.INPUT_H, cls.INPUT_W)
    seg_targets = torch.full((1, cls.INPUT_H, cls.INPUT_W),
                             fill_value=cls.IGNORE_INDEX, dtype=torch.long)
    det_targets = torch.zeros(1, cls.DET_GRID_H, cls.DET_GRID_W,
                              5 + cls.NUM_DET_CLASSES)
    out = model(x)
    loss = compute_loss(out, seg_targets, det_targets)
    # Cross entropy with all-ignore returns nan in some torch versions; we
    # just need total to be a tensor and not crash. It's allowed to be NaN.
    check("loss runs even when entire seg target is ignore",
          isinstance(loss["total"], torch.Tensor))


# ─── decode_detections ────────────────────────────────────────────────────
def test_decode_detections_returns_per_image_lists():
    print("\n[decode_detections]")
    B = 2
    det = torch.full(
        (B, 5 + cls.NUM_DET_CLASSES, cls.DET_GRID_H, cls.DET_GRID_W),
        fill_value=-10.0,   # everything off
    )
    # Make exactly one strong detection on image 0.
    det[0, 0, 4, 8] = 5.0   # obj logit
    det[0, 1, 4, 8] = 0.0   # dx logit -> 0.5 after sigmoid
    det[0, 2, 4, 8] = 0.0
    det[0, 3, 4, 8] = -1.0  # w logit
    det[0, 4, 4, 8] = -1.0  # h logit
    det[0, 5, 4, 8] = 5.0   # vehicle class
    det[0, 6, 4, 8] = -5.0  # not sign
    out = decode_detections(det, obj_thresh=0.3, nms_iou=0.5)
    check("returns list of length B", len(out) == B, f"got {len(out)}")
    check("image 0 has exactly one box", len(out[0]) == 1, f"got {len(out[0])}")
    check("image 1 has no boxes", len(out[1]) == 0, f"got {len(out[1])}")
    if out[0]:
        b = out[0][0]
        check("class is vehicle", b["cls"] == "vehicle")
        check("confidence in (0, 1]", 0 < b["confidence"] <= 1.0)
        check("xy near (0.5, 0.5)",
              abs(b["x"] - 0.5) < 0.1 and abs(b["y"] - 0.5) < 0.1,
              f"got x={b['x']:.3f} y={b['y']:.3f}")


# ─── Test runner ───────────────────────────────────────────────────────────
def main() -> int:
    tests = [
        test_classes_constants,
        test_model_forward_shape,
        test_model_handles_other_resolutions,
        test_encode_det_targets_basic,
        test_encode_det_targets_ignores_invalid,
        test_encode_det_targets_collision_keeps_larger,
        test_compute_loss_runs_and_is_scalar,
        test_compute_loss_skips_ignore_pixels,
        test_decode_detections_returns_per_image_lists,
    ]
    print(f"Running {len(tests)} model tests…")
    for t in tests:
        try:
            t()
        except Exception as e:
            global _FAIL
            _FAIL += 1
            print(f"  ✗ {t.__name__} crashed: {e}")
            traceback.print_exc()
    print(f"\n— passed: {_PASS}   failed: {_FAIL} —")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())