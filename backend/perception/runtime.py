"""
Module 5 — Runtime inference wrapper
====================================
Production inference path. Loads a checkpoint, runs in fp16 on CUDA when
available, channels-last memory layout, with two warm-up passes during init
so the first real frame doesn't take 200 ms.

Used by:
  * drive/runtime.py        (Module 8 — the live drive loop)
  * compare/service.py      (Module 6 — side-by-side YOLO vs ours)
  * ppo/env.py              (Module 8 — RL observation builder)
  * labeling/auto_labeler.py once a trained model exists (Module 4)

Public surface:
    PerceptionRuntime(ckpt_path: str | None, device: str | None)
        .infer(bgr_frame: np.ndarray) -> dict
        .reload(ckpt_path)
    load_active() -> PerceptionRuntime  (uses models table is_active=1 row)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from backend.core import database
from backend.perception.classes import INPUT_H, INPUT_W
from backend.perception.model import Perception, decode_detections

log = logging.getLogger("forzatek.perception.runtime")

try:
    import cv2  # type: ignore
except Exception:                                  # pragma: no cover
    cv2 = None

try:
    from backend.hud_mask import auto_propagate as _hud_mask
except Exception:                                  # pragma: no cover
    _hud_mask = None


# ─── ImageNet stats — match dataset preprocessing exactly ──────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PerceptionRuntime:
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
        obj_threshold: float = 0.3,
        nms_iou: float = 0.5,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_amp = (self.device.type == "cuda")
        self.obj_threshold = float(obj_threshold)
        self.nms_iou = float(nms_iou)
        self.model = Perception(pretrained_backbone=False)
        self.ckpt_path: Optional[str] = None

        if ckpt_path:
            self.reload(ckpt_path)
        else:
            self.model.to(self.device).eval()
            self._warmup()

    # ─── Loading ─────────────────────────────────────────────
    def reload(self, ckpt_path: str) -> None:
        ck = torch.load(ckpt_path, map_location=self.device)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last).half()
        self.ckpt_path = ckpt_path
        self._warmup()
        log.info("perception runtime ready: %s on %s", ckpt_path, self.device)

    def _warmup(self) -> None:
        dummy = torch.zeros((1, 3, INPUT_H, INPUT_W), device=self.device)
        if self.device.type == "cuda":
            dummy = dummy.to(memory_format=torch.channels_last).half()
        with torch.inference_mode():
            for _ in range(2):
                self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ─── Inference ───────────────────────────────────────────
    @torch.inference_mode()
    def infer(self, bgr_frame: np.ndarray, game_version: Optional[str] = None) -> dict:
        """Run one frame through the model.

        Args:
            bgr_frame: HxWx3 uint8 BGR (OpenCV convention). Any resolution.
            game_version: if provided, Module 3's HUD mask is applied.

        Returns:
            dict with keys:
              seg_map:   (H, W) uint8 class ids, original frame resolution
              road_mask: (H, W) bool — True where seg predicts road
              boxes:     [{cls, x, y, w, h, confidence}, ...]
              features:  (C,) float32 — global avg-pooled deepest feature
                         (consumed by ppo/env.py as part of the observation)
              latency_ms: float
        """
        if bgr_frame is None or bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError(f"infer expects HxWx3 BGR uint8 frame, got {bgr_frame!r}")
        h0, w0 = bgr_frame.shape[:2]
        t0 = time.time()

        x_bgr = bgr_frame
        if game_version and _hud_mask is not None:
            x_bgr = _hud_mask.apply_mask(x_bgr, game_version)

        if cv2 is not None:
            x_resized = cv2.resize(x_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        else:
            from PIL import Image
            pil = Image.fromarray(x_bgr[:, :, ::-1])
            pil = pil.resize((INPUT_W, INPUT_H), Image.BILINEAR)
            x_resized = np.asarray(pil)[:, :, ::-1]

        x_rgb = x_resized[:, :, ::-1].astype(np.float32) / 255.0
        x_rgb = (x_rgb - _MEAN) / _STD
        x_chw = np.ascontiguousarray(x_rgb.transpose(2, 0, 1))[None, ...]  # (1, 3, H, W)
        x = torch.from_numpy(x_chw).to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            x = x.to(memory_format=torch.channels_last).half()

        out = self.model(x)
        seg_logits = out["seg_logits"]              # (1, S, H, W)
        det_logits = out["det_logits"]              # (1, 5+C, G_h, G_w)

        seg_pred = seg_logits.argmax(dim=1)[0]      # (H, W)
        seg_np = seg_pred.detach().cpu().to(torch.uint8).numpy()
        if (h0, w0) != (INPUT_H, INPUT_W) and cv2 is not None:
            seg_np = cv2.resize(seg_np, (w0, h0), interpolation=cv2.INTER_NEAREST)

        boxes_per_image = decode_detections(
            det_logits.float(), obj_thresh=self.obj_threshold, nms_iou=self.nms_iou
        )
        boxes = boxes_per_image[0] if boxes_per_image else []

        # Backbone deep feature → global average pool → (C,)
        with torch.inference_mode():
            feats_list = self.model.backbone(x)
            deep_feat = feats_list[-1]              # (1, C, h, w)
            features = deep_feat.mean(dim=(2, 3))[0]    # (C,)

        latency_ms = (time.time() - t0) * 1000.0
        return {
            "seg_map":     seg_np,
            "road_mask":   (seg_np == 1),
            "boxes":       boxes,
            "features":    features.detach().cpu().float().numpy(),
            "latency_ms":  float(latency_ms),
        }


# ─── DB-backed loader ──────────────────────────────────────────────────────
def load_active() -> Optional[PerceptionRuntime]:
    """Load whichever model row has is_active=1. None if no active model."""
    with database.read_conn() as conn:
        row = conn.execute(
            "SELECT path FROM models WHERE is_active=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    p = Path(row["path"])
    if not p.exists():
        log.warning("active model row points at missing file: %s", p)
        return None
    return PerceptionRuntime(ckpt_path=str(p))