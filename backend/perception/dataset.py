"""
Module 5 — Dataset
==================
PyTorch Dataset that pulls labeled frames straight from SQLite.

Reads frame JPEG + labels rows. Decodes the seg PNG (class ids 0..3 + 255)
and the det boxes JSON. Applies Module 3's HUD mask (so the model never
learns that the speedometer has predictive value). Augmentation is mild and
safe — flips, brightness/hue jitter, no warps that could distort road
geometry.

Public surface:
    PerceptionDataset(split: 'train'|'val', game_versions: list[str] | None)
    collate_fn(batch) -> dict
    make_splits(seed: int = 0, val_frac: float = 0.15) -> (train_ids, val_ids)
"""
from __future__ import annotations

import base64
import io
import json
import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from backend.core import database
from backend.perception.classes import (
    EXPECTED_DET_FORMAT,
    EXPECTED_SEG_FORMAT,
    IGNORE_INDEX,
    INPUT_H,
    INPUT_W,
    NUM_SEG_CLASSES,
    SEG_CLASSES,
)
from backend.perception.model import encode_det_targets

log = logging.getLogger("forzatek.perception.dataset")

# Lazy imports for cv2/Image so tests that mock the dataset still work without
# OpenCV installed.
try:
    import cv2  # type: ignore
except Exception:                                 # pragma: no cover
    cv2 = None
try:
    from PIL import Image
except Exception:                                 # pragma: no cover
    Image = None

# Reuse Module 3's HUD mask only if it's installed (it always is in production).
try:
    from backend.hud_mask import auto_propagate as _hud_mask
except Exception:                                 # pragma: no cover
    _hud_mask = None


# ─── Split building ────────────────────────────────────────────────────────
def make_splits(
    seed: int = 0,
    val_frac: float = 0.15,
    game_versions: Optional[List[str]] = None,
) -> Tuple[List[int], List[int]]:
    """Stratified split by game_version.

    Picks all frames where label_status='labeled' AND has at least one labels
    row. Then within each game_version bucket, shuffles deterministically and
    peels off `val_frac` for validation. Each bucket contributes to both
    splits, so val never has frames from a version unseen at train time.
    """
    if val_frac <= 0 or val_frac >= 1:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")

    sql = """
        SELECT DISTINCT f.id, f.game_version
        FROM frames f
        WHERE f.label_status = 'labeled'
          AND EXISTS (SELECT 1 FROM labels l WHERE l.frame_id = f.id)
    """
    if game_versions:
        placeholders = ",".join("?" * len(game_versions))
        sql += f" AND f.game_version IN ({placeholders})"
        params = list(game_versions)
    else:
        params = []
    sql += " ORDER BY f.id ASC"

    with database.read_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    by_version: dict[str, List[int]] = {}
    for r in rows:
        by_version.setdefault(r["game_version"] or "_unknown", []).append(r["id"])

    rng = random.Random(seed)
    train_ids: List[int] = []
    val_ids:   List[int] = []
    for ver, ids in sorted(by_version.items()):
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * val_frac)) if len(ids) >= 4 else 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    log.info("split: %d train, %d val (versions=%s)",
             len(train_ids), len(val_ids), list(by_version.keys()))
    return train_ids, val_ids


# ─── Helpers ───────────────────────────────────────────────────────────────
def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Return BGR uint8 (H, W, 3) — same convention as OpenCV."""
    if cv2 is not None:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img
    if Image is None:                              # pragma: no cover
        raise RuntimeError("Need OpenCV or Pillow to decode JPEGs.")
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    arr = np.asarray(img)[:, :, ::-1]               # RGB -> BGR
    return np.ascontiguousarray(arr)


def _decode_seg_mask(b64: str) -> np.ndarray:
    """Decode the seg PNG to a (H, W) uint8 array of class ids."""
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
    pil = Image.fromarray(img[:, :, ::-1])          # BGR->RGB
    pil = pil.resize((w, h), Image.BILINEAR)
    return np.asarray(pil)[:, :, ::-1]              # RGB->BGR


def _resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if Image is None:                              # pragma: no cover
        raise RuntimeError("Need OpenCV or Pillow to resize mask.")
    pil = Image.fromarray(mask)
    pil = pil.resize((w, h), Image.NEAREST)
    return np.array(pil, dtype=np.uint8)


def _validate_seg_classes(arr: np.ndarray) -> np.ndarray:
    """Map any unexpected class id to IGNORE_INDEX so loss skips it."""
    valid = (arr < NUM_SEG_CLASSES) | (arr == IGNORE_INDEX)
    if not valid.all():
        # Stamp invalid pixels as ignore.
        bad = ~valid
        out = arr.copy()
        out[bad] = IGNORE_INDEX
        return out
    return arr


# ─── The Dataset ───────────────────────────────────────────────────────────
class PerceptionDataset(Dataset):
    """One sample per frame_id. Returns image+seg+det targets.

    Args:
        ids: list of frame ids to draw from. Use make_splits() to produce these.
        train: if True, augmentation is applied. False for the val set.
        apply_hud_mask: if True (default), Module 3's mask is applied.
            Tests can disable this.
    """

    def __init__(
        self,
        ids: List[int],
        train: bool = True,
        apply_hud_mask: bool = True,
        augment_strength: float = 1.0,
    ):
        self.ids: List[int] = list(ids)
        self.train: bool = bool(train)
        self.apply_hud_mask: bool = bool(apply_hud_mask)
        self.aug: float = float(augment_strength)

    def __len__(self) -> int:
        return len(self.ids)

    # ─── Loading ─────────────────────────────────────────────
    def _load_row(self, frame_id: int) -> Optional[dict]:
        with database.read_conn() as conn:
            frame = conn.execute(
                "SELECT id, frame_jpeg, game_version, width, height "
                "FROM frames WHERE id = ?",
                (frame_id,),
            ).fetchone()
            if frame is None:
                return None
            label_rows = conn.execute(
                "SELECT task, data_json FROM labels "
                "WHERE frame_id = ? ORDER BY id ASC",
                (frame_id,),
            ).fetchall()

        seg_payload = None
        det_payload = None
        for r in label_rows:
            try:
                payload = json.loads(r["data_json"])
            except Exception:
                log.warning("frame %d label task=%s: bad JSON", frame_id, r["task"])
                continue
            if r["task"] == "seg":
                if payload.get("format") != EXPECTED_SEG_FORMAT:
                    log.warning("frame %d seg label has unexpected format: %s",
                                frame_id, payload.get("format"))
                seg_payload = payload
            elif r["task"] == "det":
                if payload.get("format") != EXPECTED_DET_FORMAT:
                    log.warning("frame %d det label has unexpected format: %s",
                                frame_id, payload.get("format"))
                det_payload = payload

        return {
            "frame_id":     frame["id"],
            "jpeg_bytes":   bytes(frame["frame_jpeg"]),
            "game_version": frame["game_version"],
            "width":        frame["width"],
            "height":       frame["height"],
            "seg":          seg_payload,
            "det":          det_payload,
        }

    # ─── Augmentation ────────────────────────────────────────
    def _augment(self, img: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Apply train-time augmentation. Returns (img, seg, transform_meta).
        transform_meta tracks whether we flipped — boxes need the same flip.
        """
        flipped = False
        if random.random() < 0.5 * self.aug:
            img = img[:, ::-1, :].copy()
            seg = seg[:, ::-1].copy()
            flipped = True

        # Brightness in HSV value channel.
        if cv2 is not None and random.random() < 0.7 * self.aug:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
            delta = int((random.random() - 0.5) * 50 * self.aug)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta, 0, 255)
            # Hue jitter.
            hue_delta = int((random.random() - 0.5) * 20 * self.aug)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return img, seg, {"flipped": flipped}

    @staticmethod
    def _flip_boxes(boxes: List[dict]) -> List[dict]:
        out = []
        for b in boxes:
            out.append({
                "cls":        b["cls"],
                "x":          1.0 - float(b["x"]),
                "y":          float(b["y"]),
                "w":          float(b["w"]),
                "h":          float(b["h"]),
                "confidence": float(b.get("confidence", 1.0)),
            })
        return out

    # ─── __getitem__ ─────────────────────────────────────────
    def __getitem__(self, idx: int) -> dict:
        frame_id = self.ids[idx]
        row = self._load_row(frame_id)
        if row is None:
            raise IndexError(f"frame {frame_id} not in DB")

        # Decode image.
        img = _decode_jpeg(row["jpeg_bytes"])
        h0, w0 = img.shape[:2]

        # Apply HUD mask BEFORE resize so the masked-out pixels are clearly
        # zero in the final tensor.
        if self.apply_hud_mask and _hud_mask is not None and row["game_version"]:
            img = _hud_mask.apply_mask(img, row["game_version"])

        # Decode seg mask. May be missing if labels were det-only.
        if row["seg"] is None or "mask_png_b64" not in row["seg"]:
            seg = np.full((h0, w0), IGNORE_INDEX, dtype=np.uint8)
        else:
            seg = _decode_seg_mask(row["seg"]["mask_png_b64"])
            seg = _validate_seg_classes(seg)
            if seg.shape[:2] != (h0, w0):
                seg = _resize_mask(seg, w0, h0)

        # Resize to model input.
        img = _resize_image(img, INPUT_W, INPUT_H)
        seg = _resize_mask(seg, INPUT_W, INPUT_H)

        # Augmentation (train only).
        boxes = row["det"]["boxes"] if row["det"] else []
        if self.train:
            img, seg, meta = self._augment(img, seg)
            if meta["flipped"]:
                boxes = self._flip_boxes(boxes)

        # To tensors.
        # BGR -> RGB, HWC -> CHW, normalize 0..1.
        img_rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        # ImageNet normalization (matches MobileNetV3 pretrained weights).
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_rgb = (img_rgb - mean) / std
        img_t = torch.from_numpy(np.ascontiguousarray(img_rgb.transpose(2, 0, 1)))

        seg_t = torch.from_numpy(np.ascontiguousarray(seg)).long()
        det_t = encode_det_targets(boxes)

        return {
            "image":    img_t,
            "seg":      seg_t,
            "det":      det_t,
            "frame_id": frame_id,
        }


def collate_fn(batch: List[dict]) -> dict:
    """Stack tensors. Keep frame_id as a list."""
    return {
        "image":     torch.stack([b["image"] for b in batch], dim=0),
        "seg":       torch.stack([b["seg"]   for b in batch], dim=0),
        "det":       torch.stack([b["det"]   for b in batch], dim=0),
        "frame_id":  [b["frame_id"] for b in batch],
    }