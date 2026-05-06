"""
Module 5 — Metrics
==================
Pure-functional. No I/O, no side effects. Used by:
  * train.py validation loop
  * compare/service.py (Module 6) for side-by-side scoring
  * tests

All functions accept torch tensors OR numpy arrays. Internally everything
is numpy for predictability.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from backend.perception.classes import (
    DET_CLASSES,
    IGNORE_INDEX,
    NUM_DET_CLASSES,
    NUM_SEG_CLASSES,
    SEG_CLASSES,
)


# ─── Helpers ───────────────────────────────────────────────────────────────
def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


# ─── Segmentation ──────────────────────────────────────────────────────────
def confusion_matrix(
    pred: np.ndarray,                # (N,) or (H, W) class ids
    target: np.ndarray,              # same shape
    num_classes: int = NUM_SEG_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Return (C, C) int matrix where row=target, col=pred."""
    pred = _to_numpy(pred).reshape(-1).astype(np.int64)
    target = _to_numpy(target).reshape(-1).astype(np.int64)
    keep = (target != ignore_index)
    pred = pred[keep]
    target = target[keep]
    # Clip predictions to valid range — defensive, shouldn't happen.
    pred = np.clip(pred, 0, num_classes - 1)
    target = np.clip(target, 0, num_classes - 1)
    idx = target * num_classes + pred
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def count_target_pixels(
    target: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> tuple:
    """Return (total_pixels, real_pixels) — total is everything, real is
    pixels that are not ignore_index.

    Used by _validate() to compute pixel_coverage = real / total alongside
    the confusion matrix. We need this separately because `confusion_matrix`
    drops ignore pixels, so its sum() doesn't include them.
    """
    target = _to_numpy(target).reshape(-1).astype(np.int64)
    total = int(target.size)
    real  = int((target != ignore_index).sum())
    return total, real


def iou_per_class(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU from a confusion matrix.

    IoU_c = TP_c / (TP_c + FP_c + FN_c) = diag / (rows + cols - diag).
    Classes that never appear in target *and* never predicted yield NaN.
    """
    cm = cm.astype(np.float64)
    diag = np.diag(cm)
    rows = cm.sum(axis=1)
    cols = cm.sum(axis=0)
    denom = rows + cols - diag
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, diag / denom, np.nan)
    return iou


def seg_metrics(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    total_pixels: Optional[int] = None,
    real_pixels: Optional[int] = None,
) -> Dict[str, float]:
    """Friendly name -> IoU. Plus aggregates: mIoU, pixel accuracy, coverage.

    Coverage diagnostics:
      pixel_coverage           : real_pixels / total_pixels (overall)
      class_coverage_<name>    : pixels of that class / real_pixels
      coverage_warning         : True if pixel_coverage < 10% — IoU likely meaningless
      class_collapse_warning   : True if any class has 0 pixels in target — IoU NaN/1.0

    `total_pixels` and `real_pixels` come from count_target_pixels() and
    must be aggregated by the caller across the whole validation set —
    otherwise coverage is unknown and the warning fields default to False.
    """
    if class_names is None:
        class_names = SEG_CLASSES
    iou = iou_per_class(cm)
    out: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        v = iou[i]
        out[f"iou_{name}"] = float(v) if np.isfinite(v) else float("nan")
    finite = iou[np.isfinite(iou)]
    out["miou"] = float(finite.mean()) if finite.size else float("nan")

    total = cm.sum()
    out["pixel_acc"] = float(np.diag(cm).sum() / total) if total > 0 else 0.0

    # Pixel coverage — fraction of all val pixels that weren't ignore_index.
    # When labels are too sparse this drops below 10% and IoU becomes
    # essentially meaningless.
    if total_pixels is not None and real_pixels is not None and total_pixels > 0:
        out["pixel_coverage"] = float(real_pixels) / float(total_pixels)
    else:
        out["pixel_coverage"] = float("nan")

    # Per-class coverage — what fraction of the *real* pixels are this class.
    rows = cm.sum(axis=1)
    real_total = float(rows.sum())
    for i, name in enumerate(class_names):
        out[f"class_coverage_{name}"] = (
            float(rows[i] / real_total) if real_total > 0 else 0.0
        )

    # Warnings — set as ints (0/1) so JSON-serializing is unambiguous.
    cov = out.get("pixel_coverage", float("nan"))
    out["coverage_warning"] = int(np.isfinite(cov) and cov < 0.10)
    # Class collapse: any class missing entirely from target. With our
    # 4-class setup this manifests as "iou_curb=1.0" because there are
    # zero target pixels of that class and the model also (correctly)
    # predicts none — division resolves to NaN, but if the model DID
    # predict any, the IoU calculation gives 1.0 trivially.
    out["class_collapse_warning"] = int(any(rows[i] == 0 for i in range(len(class_names))))

    return out


# ─── Detection ─────────────────────────────────────────────────────────────
def _iou_xywh(a: dict, b: dict) -> float:
    ax0 = a["x"] - a["w"] / 2; ay0 = a["y"] - a["h"] / 2
    ax1 = a["x"] + a["w"] / 2; ay1 = a["y"] + a["h"] / 2
    bx0 = b["x"] - b["w"] / 2; by0 = b["y"] - b["h"] / 2
    bx1 = b["x"] + b["w"] / 2; by1 = b["y"] + b["h"] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def detection_pr(
    predictions: List[List[dict]],   # per-image list of pred boxes
    ground_truth: List[List[dict]],  # per-image list of gt boxes
    iou_thresh: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-class precision / recall / AP at a single IoU threshold.

    Each box dict: {cls, x, y, w, h, confidence?}. Pred boxes need a
    confidence; gt does not.
    """
    if class_names is None:
        class_names = DET_CLASSES
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions / ground_truth length mismatch")

    out: Dict[str, Dict[str, float]] = {}
    for cls in class_names:
        # Flatten all preds for this class with image-id, sorted by confidence.
        all_preds: List[tuple] = []
        gt_per_image: List[List[dict]] = []
        n_gt = 0
        for img_idx in range(len(predictions)):
            pred_boxes = [p for p in predictions[img_idx] if p["cls"] == cls]
            gt_boxes   = [g for g in ground_truth[img_idx] if g["cls"] == cls]
            n_gt += len(gt_boxes)
            gt_per_image.append([dict(g) for g in gt_boxes])  # copy for mutation
            for p in pred_boxes:
                all_preds.append((float(p.get("confidence", 1.0)), img_idx, p))
        all_preds.sort(key=lambda t: t[0], reverse=True)

        if n_gt == 0 and not all_preds:
            out[cls] = {"precision": float("nan"), "recall": float("nan"),
                        "ap": float("nan"), "n_gt": 0, "n_pred": 0}
            continue

        tp = np.zeros(len(all_preds), dtype=np.float64)
        fp = np.zeros(len(all_preds), dtype=np.float64)
        gt_matched = [[False] * len(g) for g in gt_per_image]

        for i, (_, img_idx, p) in enumerate(all_preds):
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gt_per_image[img_idx]):
                if gt_matched[img_idx][j]:
                    continue
                v = _iou_xywh(p, g)
                if v > best_iou:
                    best_iou = v
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0:
                tp[i] = 1.0
                gt_matched[img_idx][best_j] = True
            else:
                fp[i] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall    = cum_tp / max(n_gt, 1)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

        # 11-point interpolated AP — simple, well-defined.
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            mask = recall >= r
            p_at_r = float(precision[mask].max()) if mask.any() else 0.0
            ap += p_at_r / 11.0

        out[cls] = {
            "precision": float(precision[-1]) if len(precision) else float("nan"),
            "recall":    float(recall[-1])    if len(recall)    else 0.0,
            "ap":        float(ap),
            "n_gt":      int(n_gt),
            "n_pred":    int(len(all_preds)),
        }

    # Mean AP across classes (exclude NaN classes from mean).
    aps = [v["ap"] for v in out.values() if np.isfinite(v["ap"])]
    out["mAP"] = {"map": float(np.mean(aps)) if aps else float("nan"),
                  "iou_thresh": iou_thresh}
    return out