"""
Module 6 — Compare FastAPI routes
==================================
Three binary overlay endpoints. Eel is bad at large binary returns, so the
JPEGs go through the FastAPI port (8001) instead.

These endpoints CAN trigger inference (when a frame is requested that's
not in the cache). That's fine because:
  1. service.compare_frame now routes through the inference worker pool,
     freeing the gevent thread to send the reply.
  2. After compare_next runs once, the result is cached and the three
     overlay fetches just read bytes.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse

from backend.compare import service

log = logging.getLogger("forzatek.compare.routes")

_router = APIRouter(prefix="/api/compare", tags=["compare"])

_LAST_CACHE: dict = {}
_CACHE_ORDER: list = []
_CACHE_MAX = 4


def _get_cached_or_render(frame_id: int) -> Optional[service.CompareResult]:
    cached = _LAST_CACHE.get(frame_id)
    if cached is not None:
        return cached
    log.info("compare.routes: cache miss for frame %d — rendering", frame_id)
    res = service.compare_frame(int(frame_id), render_overlays=True)
    if res is None:
        return None
    _LAST_CACHE[frame_id] = res
    _CACHE_ORDER.append(frame_id)
    while len(_CACHE_ORDER) > _CACHE_MAX:
        old = _CACHE_ORDER.pop(0)
        _LAST_CACHE.pop(old, None)
    return res


def invalidate_cache() -> None:
    _LAST_CACHE.clear()
    _CACHE_ORDER.clear()


def cache_put(res: service.CompareResult) -> None:
    """Public hook so eel_api can warm the cache after compare_next runs."""
    if res is None:
        return
    _LAST_CACHE[res.frame_id] = res
    _CACHE_ORDER.append(res.frame_id)
    while len(_CACHE_ORDER) > _CACHE_MAX:
        old = _CACHE_ORDER.pop(0)
        _LAST_CACHE.pop(old, None)


def _serve_jpeg(b: Optional[bytes], kind: str, frame_id: int) -> Response:
    if b is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"no {kind} overlay for frame {frame_id}"},
        )
    return Response(
        content=b,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@_router.get("/status")
def get_status() -> JSONResponse:
    try:
        return JSONResponse({"ok": True, **service.status()})
    except Exception as e:
        log.exception("status failed")
        raise HTTPException(status_code=500, detail=str(e))


@_router.get("/frame/{frame_id}/gt_overlay.jpg")
def gt_overlay(frame_id: int) -> Response:
    res = _get_cached_or_render(int(frame_id))
    if res is None:
        raise HTTPException(status_code=404, detail="frame not found or not labeled")
    return _serve_jpeg(res.gt_jpeg, "gt", frame_id)


@_router.get("/frame/{frame_id}/yolo_overlay.jpg")
def yolo_overlay(frame_id: int) -> Response:
    res = _get_cached_or_render(int(frame_id))
    if res is None:
        raise HTTPException(status_code=404, detail="frame not found or not labeled")
    return _serve_jpeg(res.yolo_jpeg, "yolo", frame_id)


@_router.get("/frame/{frame_id}/ours_overlay.jpg")
def ours_overlay(frame_id: int) -> Response:
    res = _get_cached_or_render(int(frame_id))
    if res is None:
        raise HTTPException(status_code=404, detail="frame not found or not labeled")
    return _serve_jpeg(res.ours_jpeg, "ours", frame_id)


@_router.post("/cache/invalidate")
def post_invalidate_cache() -> JSONResponse:
    invalidate_cache()
    service.reload_perception()
    return JSONResponse({"ok": True})


def register_routes(app) -> None:
    app.include_router(_router)
    log.info("Module 6 (Compare) FastAPI routes registered.")