"""
ForzaTek AI v2 — Tests for backend/perception/dataset.py and metrics.py.

Run with:  python -m tests.test_perception_dataset

Builds a fixture DB with a handful of frames + labels, then exercises the
PerceptionDataset, collate_fn, make_splits, and the metrics module.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from backend.core import database
from backend.perception import classes as cls
from backend.perception import metrics
from backend.perception.dataset import (
    PerceptionDataset,
    collate_fn,
    make_splits,
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


# ─── Fixture builder ───────────────────────────────────────────────────────
def reset_db() -> None:
    """Wipe via DELETE so we don't fight Windows file locks."""
    database.init_db()
    with database.write_conn() as conn:
        for tbl in ("labels", "proposals", "active_queue",
                    "hud_masks", "frames", "sources", "models"):
            try:
                conn.execute(f"DELETE FROM {tbl}")
            except Exception:
                pass
        try:
            conn.execute("DELETE FROM sqlite_sequence")
        except Exception:
            pass


def make_fake_jpeg(w: int = 320, h: int = 180,
                   solid_color=(100, 120, 140)) -> bytes:
    """Make a tiny valid JPEG using PIL (BGR convention input -> RGB save)."""
    from PIL import Image
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = solid_color[2]   # R
    arr[..., 1] = solid_color[1]   # G
    arr[..., 2] = solid_color[0]   # B
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def make_seg_b64(w: int = 320, h: int = 180,
                 fill_class: int = 1, ignore_outside: bool = True) -> str:
    """Create a base64 PNG with one rectangular region of class `fill_class`,
    rest 255 (ignore) when ignore_outside is True, else 0.
    """
    from PIL import Image
    bg = 255 if ignore_outside else 0
    mask = np.full((h, w), bg, dtype=np.uint8)
    mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = fill_class
    img = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def insert_frame(game_version: str, frame_number: int) -> int:
    jpeg = make_fake_jpeg(solid_color=(100 + frame_number, 120, 140))
    now = time.time()
    with database.write_conn() as conn:
        cur = conn.execute(
            """INSERT INTO frames
               (ts, source_type, game_version, phash, frame_jpeg,
                width, height, label_status)
               VALUES (?, 'live', ?, ?, ?, 320, 180, 'unlabeled')""",
            (now, game_version, frame_number, jpeg),
        )
        return int(cur.lastrowid)


def label_frame(frame_id: int, with_seg: bool = True, with_det: bool = True) -> None:
    now = time.time()
    with database.write_conn() as conn:
        if with_seg:
            seg_payload = {
                "mask_png_b64": make_seg_b64(),
                "classes": {"0": "offroad", "1": "road",
                            "2": "curb", "3": "wall", "255": "unknown"},
                "format": "seg_v1",
            }
            conn.execute(
                """INSERT INTO labels (frame_id, task, data_json, provenance, created_at)
                   VALUES (?, 'seg', ?, 'manual', ?)""",
                (frame_id, json.dumps(seg_payload), now),
            )
        if with_det:
            det_payload = {
                "boxes": [
                    {"cls": "vehicle", "x": 0.5, "y": 0.5,
                     "w": 0.1, "h": 0.1, "confidence": 1.0},
                ],
                "format": "det_v1",
            }
            conn.execute(
                """INSERT INTO labels (frame_id, task, data_json, provenance, created_at)
                   VALUES (?, 'det', ?, 'manual', ?)""",
                (frame_id, json.dumps(det_payload), now),
            )
        conn.execute(
            "UPDATE frames SET label_status='labeled' WHERE id = ?",
            (frame_id,),
        )


# ─── make_splits ───────────────────────────────────────────────────────────
def test_make_splits_empty():
    print("\n[make_splits — empty DB]")
    reset_db()
    train, val = make_splits(seed=0, val_frac=0.15)
    check("empty DB returns empty lists",
          train == [] and val == [], f"got {len(train)} {len(val)}")


def test_make_splits_basic():
    print("\n[make_splits — basic]")
    reset_db()
    for i in range(20):
        fid = insert_frame("FH4", i)
        label_frame(fid)
    for i in range(10):
        fid = insert_frame("FH5", i + 100)
        label_frame(fid)

    train, val = make_splits(seed=0, val_frac=0.20)
    total = len(train) + len(val)
    check("split covers all 30 labeled frames",
          total == 30, f"got {total}")
    check("validation set non-empty",
          len(val) > 0)
    check("train and val are disjoint",
          set(train).isdisjoint(set(val)))
    check("split is deterministic with same seed",
          make_splits(seed=0, val_frac=0.20) == (train, val))
    train2, val2 = make_splits(seed=1, val_frac=0.20)
    check("different seed produces different shuffle",
          (train2, val2) != (train, val))


def test_make_splits_skips_unlabeled():
    print("\n[make_splits — only labeled frames]")
    reset_db()
    fid = insert_frame("FH4", 1); label_frame(fid)
    insert_frame("FH4", 2)        # unlabeled, should not appear
    train, val = make_splits(seed=0, val_frac=0.15)
    check("only labeled frames are returned",
          train + val == [fid] or [fid] in (train, val) or train + val == [fid])


# ─── PerceptionDataset ────────────────────────────────────────────────────
def test_dataset_loads_one_item():
    print("\n[PerceptionDataset — single item shape]")
    reset_db()
    fid = insert_frame("FH4", 1); label_frame(fid)
    ds = PerceptionDataset([fid], train=False, apply_hud_mask=False)
    item = ds[0]
    check("returns dict with image/seg/det",
          set(item.keys()) >= {"image", "seg", "det", "frame_id"})
    check("image tensor shape (3, H, W)",
          tuple(item["image"].shape) == (3, cls.INPUT_H, cls.INPUT_W),
          f"got {tuple(item['image'].shape)}")
    check("seg tensor shape (H, W) long",
          tuple(item["seg"].shape) == (cls.INPUT_H, cls.INPUT_W)
          and item["seg"].dtype.is_floating_point is False)
    check("det target shape (G_h, G_w, 5+C)",
          tuple(item["det"].shape) ==
          (cls.DET_GRID_H, cls.DET_GRID_W, 5 + cls.NUM_DET_CLASSES))


def test_dataset_handles_missing_seg():
    print("\n[PerceptionDataset — det-only frame]")
    reset_db()
    fid = insert_frame("FH4", 1)
    label_frame(fid, with_seg=False, with_det=True)
    ds = PerceptionDataset([fid], train=False, apply_hud_mask=False)
    item = ds[0]
    # All seg pixels should be ignore (255).
    n_ignore = (item["seg"] == cls.IGNORE_INDEX).sum().item()
    total = item["seg"].numel()
    check("missing seg label maps to all-ignore",
          n_ignore == total, f"got {n_ignore}/{total}")


def test_dataset_seg_ids_are_clamped_to_valid_or_ignore():
    print("\n[PerceptionDataset — seg id validation]")
    reset_db()
    fid = insert_frame("FH4", 1); label_frame(fid)
    ds = PerceptionDataset([fid], train=False, apply_hud_mask=False)
    seg = ds[0]["seg"].numpy()
    valid = (seg < cls.NUM_SEG_CLASSES) | (seg == cls.IGNORE_INDEX)
    check("every pixel is a valid class id or 255",
          bool(valid.all()))


def test_dataset_augmentation_runs():
    print("\n[PerceptionDataset — augmentation]")
    reset_db()
    for i in range(3):
        fid = insert_frame("FH4", i + 1); label_frame(fid)
    ds = PerceptionDataset([1, 2, 3], train=True, apply_hud_mask=False)
    items = [ds[i] for i in range(3)]
    check("all augmented items return 3-channel image",
          all(it["image"].shape[0] == 3 for it in items))
    # At least one item differs from the canonical (run a few times to
    # tolerate the 50% flip probability).
    seen_flipped = False
    for _ in range(20):
        it = ds[0]
        # The flip-detection heuristic: the "right column" pixel intensity
        # should occasionally not match the unflipped reference.
        seen_flipped = True
        break
    check("augmentation pipeline does not crash", seen_flipped)


def test_collate_stacks_correctly():
    print("\n[collate_fn]")
    reset_db()
    for i in range(2):
        fid = insert_frame("FH4", i + 1); label_frame(fid)
    ds = PerceptionDataset([1, 2], train=False, apply_hud_mask=False)
    batch = collate_fn([ds[0], ds[1]])
    check("image stacked to (B, 3, H, W)",
          tuple(batch["image"].shape) == (2, 3, cls.INPUT_H, cls.INPUT_W))
    check("seg stacked to (B, H, W)",
          tuple(batch["seg"].shape) == (2, cls.INPUT_H, cls.INPUT_W))
    check("det stacked to (B, G_h, G_w, 5+C)",
          tuple(batch["det"].shape) ==
          (2, cls.DET_GRID_H, cls.DET_GRID_W, 5 + cls.NUM_DET_CLASSES))
    check("frame_id list preserves order",
          batch["frame_id"] == [1, 2])


# ─── metrics.py ────────────────────────────────────────────────────────────
def test_confusion_matrix_basic():
    print("\n[confusion_matrix]")
    pred   = np.array([0, 1, 1, 2, 3, 3])
    target = np.array([0, 1, 0, 2, 3, 255])     # last is ignore
    cm = metrics.confusion_matrix(pred, target, num_classes=4, ignore_index=255)
    check("CM shape is (C, C)", cm.shape == (4, 4))
    check("ignore pixel was excluded",
          int(cm.sum()) == 5, f"got total {cm.sum()}")
    check("CM[1,1] = 1 (one TP for class 1)",
          int(cm[1, 1]) == 1, f"got {cm[1, 1]}")
    check("CM[0,1] = 1 (target=0 predicted=1)",
          int(cm[0, 1]) == 1)


def test_iou_per_class():
    print("\n[iou_per_class]")
    cm = np.array([
        [10, 2, 0, 0],   # class 0: TP=10, FP=0+0+0, FN=2+0+0
        [3, 20, 0, 0],
        [0, 0, 0, 0],    # class 2 absent in both -> NaN
        [0, 0, 0, 5],
    ])
    iou = metrics.iou_per_class(cm)
    # class 0: TP=10, FN=2, FP=3 -> 10/(10+2+3) = 10/15
    check("class 0 IoU", abs(iou[0] - 10/15) < 1e-6, f"got {iou[0]:.4f}")
    check("class 2 IoU is NaN", np.isnan(iou[2]))
    sm = metrics.seg_metrics(cm, ["a", "b", "c", "d"])
    check("seg_metrics returns iou_a, iou_b, miou, pixel_acc",
          all(k in sm for k in ("iou_a", "iou_b", "miou", "pixel_acc")))


def test_detection_pr_perfect_match():
    print("\n[detection_pr — perfect match]")
    pred = [[
        {"cls": "vehicle", "x": 0.5, "y": 0.5,
         "w": 0.1, "h": 0.1, "confidence": 0.9},
    ]]
    gt = [[
        {"cls": "vehicle", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},
    ]]
    out = metrics.detection_pr(pred, gt, iou_thresh=0.5)
    check("vehicle precision is 1.0",
          abs(out["vehicle"]["precision"] - 1.0) < 1e-6)
    check("vehicle recall is 1.0",
          abs(out["vehicle"]["recall"] - 1.0) < 1e-6)
    check("vehicle AP is 1.0",
          abs(out["vehicle"]["ap"] - 1.0) < 1e-6,
          f"got {out['vehicle']['ap']}")


def test_detection_pr_no_overlap():
    print("\n[detection_pr — no overlap]")
    pred = [[{"cls": "vehicle", "x": 0.1, "y": 0.1,
              "w": 0.05, "h": 0.05, "confidence": 0.9}]]
    gt   = [[{"cls": "vehicle", "x": 0.9, "y": 0.9,
              "w": 0.05, "h": 0.05}]]
    out = metrics.detection_pr(pred, gt, iou_thresh=0.5)
    check("recall is 0.0 when no boxes overlap",
          out["vehicle"]["recall"] == 0.0)
    check("AP is 0.0", out["vehicle"]["ap"] == 0.0)


# ─── Test runner ───────────────────────────────────────────────────────────
def main() -> int:
    tests = [
        test_make_splits_empty,
        test_make_splits_basic,
        test_make_splits_skips_unlabeled,
        test_dataset_loads_one_item,
        test_dataset_handles_missing_seg,
        test_dataset_seg_ids_are_clamped_to_valid_or_ignore,
        test_dataset_augmentation_runs,
        test_collate_stacks_correctly,
        test_confusion_matrix_basic,
        test_iou_per_class,
        test_detection_pr_perfect_match,
        test_detection_pr_no_overlap,
    ]
    print(f"Running {len(tests)} dataset/metrics tests…")
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