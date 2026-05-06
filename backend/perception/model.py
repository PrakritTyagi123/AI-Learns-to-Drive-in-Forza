"""
Module 5 — Model architecture
==============================
MobileNetV3-Small backbone, two heads sharing the trunk:
  * Segmentation head: lightweight FPN-ish decoder, upsamples to (H/4, W/4),
    final bilinear to (H, W). 4 logits per pixel.
  * Detection head: SSD-lite single-scale on the deepest feature map.
    Each cell predicts [obj, x, y, w, h, cls...] = 5 + NUM_DET_CLASSES.

Loss = seg_ce_loss(ignore_index=255) * seg_weight
     + det_obj_bce + det_box_l1 + det_cls_ce
The detection loss only contributes from cells with assigned ground-truth
boxes — empty cells contribute only the obj-BCE "no-object" term.

Why MobileNetV3-Small:
  * Pretrained on ImageNet — converges fast on small label sets (500-2000).
  * Tiny enough (~2.5M params) to run at 30 Hz in Module 8.
  * Single forward pass yields both seg and det — important for runtime.

Public surface (what other files import):
    Perception(...)         — the nn.Module
    compute_loss(...)       — combined multi-task loss
    encode_det_targets(...) — convert label boxes -> (G_h, G_w, 5+C) tensor
    decode_detections(...)  — (B, G_h, G_w, 5+C) -> [{cls, x, y, w, h, conf}]
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.perception.classes import (
    DET_GRID_H,
    DET_GRID_W,
    IGNORE_INDEX,
    INPUT_H,
    INPUT_W,
    NUM_DET_CLASSES,
    NUM_SEG_CLASSES,
)


# ─── Backbone ──────────────────────────────────────────────────────────────
def _mobilenet_v3_small_backbone(pretrained: bool = True) -> Tuple[nn.Module, List[int]]:
    """Return (feature_extractor, channel_list).

    The extractor returns a list of feature maps at strides [4, 8, 16, 32].
    Channel list aligns with those strides.
    """
    try:
        from torchvision.models import (
            MobileNet_V3_Small_Weights,
            mobilenet_v3_small,
        )
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v3_small(weights=weights)
    except Exception:
        # Older torchvision: fall back to the boolean kwarg.
        from torchvision.models import mobilenet_v3_small
        net = mobilenet_v3_small(pretrained=pretrained)

    features = net.features  # nn.Sequential of 13 blocks for v3-small

    # In MobileNetV3-Small, the strides change at:
    #   block 0:  stride 2  -> /2
    #   block 1:  stride 2  -> /4   (16 ch)   <- C2
    #   block 3:  stride 2  -> /8   (24 ch)   <- C3
    #   block 8:  stride 2  -> /16  (48 ch)   <- C4
    #   block 11: stride 2  -> /32  (96 ch)
    #   block 12: 1x1 to 576 ch                <- C5 (576 ch)
    tap_indices = [1, 3, 8, 12]
    channels = [16, 24, 48, 576]

    class _Tapped(nn.Module):
        def __init__(self, seq: nn.Sequential, taps: List[int]):
            super().__init__()
            self.seq = seq
            self.taps = sorted(taps)

        def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
            outs: List[torch.Tensor] = []
            tap_set = set(self.taps)
            for i, layer in enumerate(self.seq):
                x = layer(x)
                if i in tap_set:
                    outs.append(x)
            return outs

    return _Tapped(features, tap_indices), channels


# ─── Segmentation head ─────────────────────────────────────────────────────
class _SegHead(nn.Module):
    """Lightweight FPN-style decoder, then bilinear upsample to input size."""

    def __init__(self, in_channels: List[int], num_classes: int, hidden: int = 64):
        super().__init__()
        c2, c3, c4, c5 = in_channels
        self.lat2 = nn.Conv2d(c2, hidden, 1)
        self.lat3 = nn.Conv2d(c3, hidden, 1)
        self.lat4 = nn.Conv2d(c4, hidden, 1)
        self.lat5 = nn.Conv2d(c5, hidden, 1)

        self.smooth4 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.smooth3 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.smooth2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))

        self.classifier = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        f2, f3, f4, f5 = feats
        p5 = self.lat5(f5)
        p4 = self.lat4(f4) + F.interpolate(p5, size=f4.shape[-2:], mode="nearest")
        p4 = self.smooth4(p4)
        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="nearest")
        p2 = self.smooth2(p2)
        logits = self.classifier(p2)
        # Upsample to input resolution.
        return F.interpolate(logits, size=(INPUT_H, INPUT_W),
                             mode="bilinear", align_corners=False)


# ─── Detection head ────────────────────────────────────────────────────────
class _DetHead(nn.Module):
    """Single-scale anchor-free SSD-lite. Operates on the deepest feature map.

    Output channels per cell: 5 + NUM_DET_CLASSES.
        [obj_logit, x_offset, y_offset, w, h, cls_logits...]
        x_offset, y_offset:  sigmoid -> 0..1 within cell
        w, h:                sigmoid -> 0..1 in normalized image coords
    """

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 96):
        super().__init__()
        self.num_classes = num_classes
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
        )
        self.predictor = nn.Conv2d(hidden, 5 + num_classes, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.body(feat)
        return self.predictor(x)  # (B, 5+C, G_h, G_w)


# ─── Full model ────────────────────────────────────────────────────────────
class Perception(nn.Module):
    """The shared multi-task perception model.

    forward(x) -> {"seg_logits": (B, S, H, W), "det_logits": (B, 5+C, G_h, G_w)}
    """

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        self.backbone, channels = _mobilenet_v3_small_backbone(pretrained_backbone)
        self.seg_head = _SegHead(channels, NUM_SEG_CLASSES)
        self.det_head = _DetHead(channels[-1], NUM_DET_CLASSES)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.backbone(x)
        return {
            "seg_logits": self.seg_head(feats),
            "det_logits": self.det_head(feats[-1]),
        }


# ─── Target encoding (label box list -> tensor) ────────────────────────────
def encode_det_targets(boxes: List[dict], grid_h: int = DET_GRID_H,
                       grid_w: int = DET_GRID_W) -> torch.Tensor:
    """Convert a list of normalized boxes to a (G_h, G_w, 5+C) tensor.

    Each entry: [obj, dx, dy, w, h, cls_onehot...]
    Boxes assigned to the cell their center falls into. If two boxes land
    in the same cell, the larger one wins (more important to detect).
    """
    C = NUM_DET_CLASSES
    target = torch.zeros((grid_h, grid_w, 5 + C), dtype=torch.float32)
    # Track existing assignments by cell to resolve collisions.
    assigned_area = [[0.0] * grid_w for _ in range(grid_h)]

    for b in boxes or []:
        try:
            cx = float(b["x"]); cy = float(b["y"])
            w  = float(b["w"]); h  = float(b["h"])
            cls_name = str(b["cls"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            continue
        if w <= 0 or h <= 0 or w > 1.0 or h > 1.0:
            continue
        if cls_name not in ("vehicle", "sign"):
            continue
        cls_id = 0 if cls_name == "vehicle" else 1

        # Locate cell.
        gx = min(grid_w - 1, int(cx * grid_w))
        gy = min(grid_h - 1, int(cy * grid_h))
        area = w * h
        if area < assigned_area[gy][gx]:
            continue   # smaller box loses
        # In-cell offset 0..1.
        dx = (cx * grid_w) - gx
        dy = (cy * grid_h) - gy

        cell = torch.zeros(5 + C)
        cell[0] = 1.0
        cell[1] = dx
        cell[2] = dy
        cell[3] = w
        cell[4] = h
        cell[5 + cls_id] = 1.0
        target[gy, gx] = cell
        assigned_area[gy][gx] = area

    return target


# ─── Loss ──────────────────────────────────────────────────────────────────
def compute_loss(
    outputs: dict,
    seg_targets: torch.Tensor,        # (B, H, W) long tensor, 255=ignore
    det_targets: torch.Tensor,        # (B, G_h, G_w, 5+C)
    seg_weight: float = 1.0,
    det_obj_weight: float = 1.0,
    det_box_weight: float = 2.0,
    det_cls_weight: float = 0.5,
) -> dict:
    """Combined multi-task loss.

    Returns dict with components for logging plus the scalar `total`.
    """
    # ─── Seg loss ────────────────────────────────────────────────
    seg_logits = outputs["seg_logits"]
    seg_loss = F.cross_entropy(
        seg_logits, seg_targets.long(),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )

    # ─── Detection loss ──────────────────────────────────────────
    det_pred = outputs["det_logits"]                          # (B, 5+C, G_h, G_w)
    det_pred = det_pred.permute(0, 2, 3, 1).contiguous()      # (B, G_h, G_w, 5+C)

    obj_logits = det_pred[..., 0]
    pred_xy    = torch.sigmoid(det_pred[..., 1:3])
    pred_wh    = torch.sigmoid(det_pred[..., 3:5])
    pred_cls   = det_pred[..., 5:]

    tgt_obj = det_targets[..., 0]
    tgt_xy  = det_targets[..., 1:3]
    tgt_wh  = det_targets[..., 3:5]
    tgt_cls = det_targets[..., 5:]

    obj_loss = F.binary_cross_entropy_with_logits(
        obj_logits, tgt_obj, reduction="mean",
    )

    pos_mask = (tgt_obj > 0.5)
    n_pos = int(pos_mask.sum().item())
    if n_pos > 0:
        # Localization losses only on positive cells.
        xy_loss = F.smooth_l1_loss(pred_xy[pos_mask], tgt_xy[pos_mask], reduction="mean")
        wh_loss = F.smooth_l1_loss(pred_wh[pos_mask], tgt_wh[pos_mask], reduction="mean")
        box_loss = xy_loss + wh_loss
        # Classification: cross-entropy over class logits.
        cls_target_idx = tgt_cls[pos_mask].argmax(dim=-1)  # (n_pos,)
        cls_loss = F.cross_entropy(pred_cls[pos_mask], cls_target_idx, reduction="mean")
    else:
        box_loss = torch.zeros((), device=det_pred.device)
        cls_loss = torch.zeros((), device=det_pred.device)

    total = (
        seg_weight * seg_loss
        + det_obj_weight * obj_loss
        + det_box_weight * box_loss
        + det_cls_weight * cls_loss
    )
    return {
        "total":    total,
        "seg":      seg_loss.detach(),
        "det_obj":  obj_loss.detach(),
        "det_box":  box_loss.detach() if isinstance(box_loss, torch.Tensor) else torch.tensor(0.0),
        "det_cls":  cls_loss.detach() if isinstance(cls_loss, torch.Tensor) else torch.tensor(0.0),
        "n_pos":    n_pos,
    }


# ─── Decoding (inference -> human-readable boxes) ─────────────────────────
@torch.no_grad()
def decode_detections(
    det_logits: torch.Tensor,           # (B, 5+C, G_h, G_w) raw
    obj_thresh: float = 0.3,
    nms_iou: float = 0.5,
    max_per_image: int = 50,
) -> List[List[dict]]:
    """Run sigmoid + NMS, return per-image list of box dicts.

    Each box: {cls, x, y, w, h, confidence}  (normalized coords)
    """
    if det_logits.dim() != 4:
        raise ValueError(f"expected 4D tensor, got {det_logits.shape}")
    B, _, G_h, G_w = det_logits.shape

    pred = det_logits.permute(0, 2, 3, 1).contiguous()
    obj   = torch.sigmoid(pred[..., 0])
    xy    = torch.sigmoid(pred[..., 1:3])
    wh    = torch.sigmoid(pred[..., 3:5])
    cls_p = torch.softmax(pred[..., 5:], dim=-1)

    # Build cell-grid coordinates once.
    ys = torch.arange(G_h, device=pred.device).view(G_h, 1).expand(G_h, G_w).float()
    xs = torch.arange(G_w, device=pred.device).view(1, G_w).expand(G_h, G_w).float()

    out: List[List[dict]] = []
    for b in range(B):
        candidates: List[dict] = []
        mask = obj[b] > obj_thresh
        if mask.sum() == 0:
            out.append([])
            continue
        idx_y, idx_x = torch.where(mask)
        for i in range(idx_y.numel()):
            gy = idx_y[i].item()
            gx = idx_x[i].item()
            dx, dy = xy[b, gy, gx].tolist()
            w, h   = wh[b, gy, gx].tolist()
            cls_id = int(cls_p[b, gy, gx].argmax().item())
            cls_conf = float(cls_p[b, gy, gx, cls_id].item())
            confidence = float(obj[b, gy, gx].item()) * cls_conf
            cx = (gx + dx) / G_w
            cy = (gy + dy) / G_h
            candidates.append({
                "cls":        "vehicle" if cls_id == 0 else "sign",
                "x":          float(cx),
                "y":          float(cy),
                "w":          float(w),
                "h":          float(h),
                "confidence": confidence,
            })
        # NMS.
        candidates = _nms(candidates, nms_iou)
        candidates.sort(key=lambda d: d["confidence"], reverse=True)
        out.append(candidates[:max_per_image])
    return out


def _nms(boxes: List[dict], iou_thr: float) -> List[dict]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda d: d["confidence"], reverse=True)
    kept: List[dict] = []
    while boxes:
        a = boxes.pop(0)
        kept.append(a)
        boxes = [b for b in boxes if b["cls"] != a["cls"] or _iou_xywh(a, b) < iou_thr]
    return kept


def _iou_xywh(a: dict, b: dict) -> float:
    ax0 = a["x"] - a["w"] / 2; ay0 = a["y"] - a["h"] / 2
    ax1 = a["x"] + a["w"] / 2; ay1 = a["y"] + a["h"] / 2
    bx0 = b["x"] - b["w"] / 2; by0 = b["y"] - b["h"] / 2
    bx1 = b["x"] + b["w"] / 2; by1 = b["y"] + b["h"] / 2
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0