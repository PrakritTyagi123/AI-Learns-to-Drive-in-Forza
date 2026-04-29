"""
Module 4 — Eel API
==================
JS-callable functions for the labeling page.

Exposed names (call from JS as `eel.fn_name(args)()`):

    label_next(strategy)                     -> {frame_id, ...} | None
    label_frame_image(frame_id)              -> {jpeg_b64, ...} | None
    label_proposals(frame_id)                -> {seg, det, ...}
    label_progress()                         -> dict
    label_queue(limit)                       -> [...]
    label_accept(frame_id)                   -> bool
    label_submit(frame_id, seg_b64, boxes)   -> bool
    label_manual(frame_id, seg_b64, boxes)   -> bool
    label_unlabel(frame_id)                  -> bool
    label_skip(frame_id)                     -> bool
    auto_label_start(conf_thr, ent_thr)      -> dict
    auto_label_cancel()                      -> bool
    auto_label_status()                      -> dict

Binary frame JPEGs go through routes.py — base64 over the JS bridge is
slow, and the canvas swaps frames every few seconds.
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

from backend.labeling import auto_labeler, service

log = logging.getLogger("forzatek.labeling.eel_api")


def register_eel(eel) -> None:
    """Register every @eel.expose function with the running Eel instance.
    Called once from backend/main.py during boot.
    """

    @eel.expose
    def label_next(strategy: Optional[dict] = None) -> Optional[dict]:
        try:
            strategy = strategy or {}
            return service.next_frame(
                game_version=strategy.get("game_version"),
                biome=strategy.get("biome"),
                weather=strategy.get("weather"),
                time_of_day=strategy.get("time_of_day"),
            )
        except Exception as e:
            log.exception("label_next failed: %s", e)
            return None

    @eel.expose
    def label_frame_image(frame_id: int) -> Optional[dict]:
        """Return frame JPEG as base64. Convenience for small canvases —
        the high-traffic path uses the FastAPI route instead.
        """
        try:
            data = service.get_frame_image(int(frame_id), apply_hud_mask=False)
            if data is None:
                return None
            return {
                "frame_id":     data["frame_id"],
                "jpeg_b64":     base64.b64encode(data["jpeg_bytes"]).decode("ascii"),
                "game_version": data["game_version"],
                "width":        data["width"],
                "height":       data["height"],
            }
        except Exception as e:
            log.exception("label_frame_image failed: %s", e)
            return None

    @eel.expose
    def label_proposals(frame_id: int) -> dict:
        try:
            return service.get_proposals(int(frame_id))
        except Exception as e:
            log.exception("label_proposals failed: %s", e)
            return {"seg": None, "det": None, "has_proposals": False}

    @eel.expose
    def label_progress() -> dict:
        try:
            return service.progress()
        except Exception as e:
            log.exception("label_progress failed: %s", e)
            return {"error": str(e)}

    @eel.expose
    def label_queue(limit: int = 20) -> list:
        try:
            return service.list_proposals_summary(limit=int(limit))
        except Exception as e:
            log.exception("label_queue failed: %s", e)
            return []

    @eel.expose
    def label_accept(frame_id: int) -> bool:
        try:
            return service.accept_proposal(int(frame_id))
        except Exception as e:
            log.exception("label_accept failed: %s", e)
            return False

    @eel.expose
    def label_submit(
        frame_id: int,
        seg_mask_png_b64: Optional[str],
        boxes: Optional[list],
    ) -> bool:
        try:
            return service.submit_edit(
                int(frame_id), seg_mask_png_b64 or "", list(boxes or []),
            )
        except Exception as e:
            log.exception("label_submit failed: %s", e)
            return False

    @eel.expose
    def label_manual(
        frame_id: int,
        seg_mask_png_b64: Optional[str],
        boxes: Optional[list],
    ) -> bool:
        try:
            return service.submit_manual(
                int(frame_id), seg_mask_png_b64 or "", list(boxes or []),
            )
        except Exception as e:
            log.exception("label_manual failed: %s", e)
            return False

    @eel.expose
    def label_unlabel(frame_id: int) -> bool:
        try:
            return service.unlabel_frame(int(frame_id))
        except Exception as e:
            log.exception("label_unlabel failed: %s", e)
            return False

    @eel.expose
    def label_skip(frame_id: int) -> bool:
        try:
            return service.skip_frame(int(frame_id))
        except Exception as e:
            log.exception("label_skip failed: %s", e)
            return False

    # ─── Auto-labeler controls ──────────────────────────────────────────
    @eel.expose
    def auto_label_start(
        confidence_threshold: float = 0.85,
        entropy_threshold:    float = 0.50,
        batch_size:           int   = 8,
        decisive_threshold:   float = 0.85,
        include_queued:       bool  = False,
    ) -> dict:
        try:
            return auto_labeler.start(
                confidence_threshold=float(confidence_threshold),
                entropy_threshold=float(entropy_threshold),
                batch_size=int(batch_size),
                decisive_threshold=float(decisive_threshold),
                include_queued=bool(include_queued),
            )
        except Exception as e:
            log.exception("auto_label_start failed: %s", e)
            return {"started": False, "error": str(e)}

    @eel.expose
    def auto_label_cancel() -> bool:
        try:
            return auto_labeler.cancel()
        except Exception as e:
            log.exception("auto_label_cancel failed: %s", e)
            return False

    @eel.expose
    def auto_label_status() -> dict:
        try:
            return auto_labeler.status()
        except Exception as e:
            log.exception("auto_label_status failed: %s", e)
            return {"error": str(e)}

    @eel.expose
    def auto_label_preview() -> dict:
        """Most recently processed frame's seg + boxes, for the live UI panel."""
        try:
            return auto_labeler.get_preview()
        except Exception as e:
            log.exception("auto_label_preview failed: %s", e)
            return {"error": str(e)}

    log.info("Labeling Eel API registered")