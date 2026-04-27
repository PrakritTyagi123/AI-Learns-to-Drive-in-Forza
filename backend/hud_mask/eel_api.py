"""
Module 3 — Eel API
==================
JS-callable functions for the HUD mask page.

Exposed names (from JS: `eel.fn_name(args)()`):

    hud_mask_list_versions()              -> [str, ...]
    hud_mask_list()                       -> [{game_version, rect_count, ...}]
    hud_mask_get(game_version)            -> {rects, sample_frame, ...} | None
    hud_mask_get_sample_frame(game_ver)   -> {frame_id, jpeg_b64, w, h} | None
    hud_mask_save(game_ver, rects, fid)   -> {game_version, rects, ...}
    hud_mask_delete(game_version)         -> bool

Binary preview frames go through routes.py (the FastAPI side server) —
JPEGs over base64 are slow and bloat the JS bridge.
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

from backend.hud_mask import service

log = logging.getLogger("forzatek.hud_mask.eel_api")


def register_eel(eel) -> None:
    """Register every @eel.expose function with the running Eel instance.
    Called once from backend/main.py during boot.
    """

    @eel.expose
    def hud_mask_list_versions() -> list[str]:
        try:
            return service.list_known_game_versions()
        except Exception as e:
            log.exception("list_known_game_versions failed: %s", e)
            return []

    @eel.expose
    def hud_mask_list() -> list[dict]:
        try:
            return service.list_masks()
        except Exception as e:
            log.exception("list_masks failed: %s", e)
            return []

    @eel.expose
    def hud_mask_get(game_version: str) -> Optional[dict]:
        try:
            return service.get_mask(game_version)
        except Exception as e:
            log.exception("get_mask failed: %s", e)
            return None

    @eel.expose
    def hud_mask_get_sample_frame(game_version: str) -> Optional[dict]:
        """Return one random recent frame for the canvas. JPEG → base64."""
        try:
            sample = service.get_sample_frame(game_version)
            if sample is None:
                return None
            return {
                "frame_id": sample["frame_id"],
                "jpeg_b64": base64.b64encode(sample["jpeg_bytes"]).decode("ascii"),
                "width":    sample["width"],
                "height":   sample["height"],
            }
        except Exception as e:
            log.exception("get_sample_frame failed: %s", e)
            return None

    @eel.expose
    def hud_mask_save(
        game_version: str,
        rects: list,
        sample_frame_id: Optional[int] = None,
    ) -> Optional[dict]:
        try:
            return service.save_mask(game_version, rects, sample_frame_id)
        except ValueError as e:
            log.warning("save_mask rejected: %s", e)
            return {"error": str(e)}
        except Exception as e:
            log.exception("save_mask failed: %s", e)
            return {"error": "internal error — see logs"}

    @eel.expose
    def hud_mask_delete(game_version: str) -> bool:
        try:
            return service.delete_mask(game_version)
        except Exception as e:
            log.exception("delete_mask failed: %s", e)
            return False

    log.info("HUD mask Eel API registered")