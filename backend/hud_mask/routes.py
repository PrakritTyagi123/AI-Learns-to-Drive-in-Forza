"""
Module 3 — FastAPI side routes
==============================
Binary endpoints that don't go well over Eel's JS bridge:

    GET /api/hud_mask/preview/{game_version}/{frame_id}
        Returns a JPEG with the current mask applied. Used by the preview
        grid below the canvas to verify mask coverage on multiple scenes.

    GET /api/hud_mask/raw/{frame_id}
        Returns the unmasked frame JPEG. Used when the user wants to
        re-pick the canvas sample frame.

    GET /api/hud_mask/preview_list/{game_version}?n=4
        Returns JSON with N random frame_ids for that version. The UI
        calls /preview/{gv}/{frame_id} for each one to fill the grid.
"""
from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Response

from backend.core import database
from backend.hud_mask import auto_propagate, service

log = logging.getLogger("forzatek.hud_mask.routes")


def register_routes(app: FastAPI) -> None:
    """Attach Module 3 endpoints to the FastAPI side server."""

    @app.get("/api/hud_mask/preview_list/{game_version}")
    def preview_list(game_version: str, n: int = 4) -> dict:
        n = max(1, min(12, int(n)))
        frames = service.get_preview_frames(game_version, n)
        return {
            "game_version": game_version,
            "frame_ids":    [f["frame_id"] for f in frames],
        }

    @app.get("/api/hud_mask/raw/{frame_id}")
    def raw_frame(frame_id: int) -> Response:
        with database.read_conn() as conn:
            row = conn.execute(
                "SELECT frame_jpeg FROM frames WHERE id = ?",
                (frame_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(404, f"frame {frame_id} not found")
        return Response(content=bytes(row["frame_jpeg"]), media_type="image/jpeg")

    @app.get("/api/hud_mask/preview/{game_version}/{frame_id}")
    def preview(game_version: str, frame_id: int) -> Response:
        with database.read_conn() as conn:
            row = conn.execute(
                "SELECT frame_jpeg, width, height FROM frames WHERE id = ?",
                (frame_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(404, f"frame {frame_id} not found")

        # Decode → apply mask → re-encode. The masked-out regions become
        # solid black, which is visually obvious in the preview grid.
        jpeg_bytes = bytes(row["frame_jpeg"])
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(500, "frame decode failed")

        masked = auto_propagate.apply_mask(img, game_version)
        ok, buf = cv2.imencode(".jpg", masked, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(500, "frame encode failed")
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    log.info("HUD mask routes registered")