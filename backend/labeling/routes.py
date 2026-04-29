"""
Module 4 — FastAPI side routes
==============================
Binary endpoints that the canvas uses heavily and shouldn't go over the
Eel base64 bridge:

    GET /api/label/frame/{frame_id}?mask=1
        Returns a JPEG of the frame, HUD mask applied by default.
        Use ?mask=0 to get the raw frame.

    GET /api/label/proposal_mask/{frame_id}
        Returns a PNG of the latest segmentation proposal (paletted,
        small). The canvas overlays this with low alpha so the user
        can see what YOLO/SegFormer thinks before painting.
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Response

from backend.labeling import service

log = logging.getLogger("forzatek.labeling.routes")


def register_routes(app: FastAPI) -> None:
    """Attach Module 4 endpoints to the FastAPI side server."""

    @app.get("/api/label/frame/{frame_id}")
    def frame_image(frame_id: int, mask: int = 0) -> Response:
        # NOTE: mask=0 by default in road-only mode — the HUD mask is now
        # used only as a YOLO box-suppression region, not as a frame overlay.
        # Pass ?mask=1 explicitly if you want the legacy black-out preview.
        data = service.get_frame_image(frame_id, apply_hud_mask=bool(mask))
        if data is None:
            raise HTTPException(404, f"frame {frame_id} not found")
        return Response(content=data["jpeg_bytes"], media_type="image/jpeg")

    @app.get("/api/label/proposal_mask/{frame_id}")
    def proposal_mask(frame_id: int) -> Response:
        proposals = service.get_proposals(frame_id)
        seg = proposals.get("seg")
        if seg is None or "mask_png_b64" not in seg:
            raise HTTPException(404, f"no seg proposal for frame {frame_id}")
        try:
            png_bytes = base64.b64decode(seg["mask_png_b64"])
        except Exception as e:
            raise HTTPException(500, f"bad b64 payload: {e}")
        return Response(content=png_bytes, media_type="image/png")