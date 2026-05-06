"""
Module 5 — FastAPI routes
=========================
Side-server endpoints for things that don't go cleanly through Eel:
  * Server-Sent Events for live training progress.
  * Plain GET endpoint for the log tail (handy for curl-based debugging).

Eel polling at 1 Hz is fine for the sparkline + epoch counter — the SSE
endpoint is here for users who hammer F5 or want to graph multiple runs
with an external dashboard.

Mounted from backend/main.py:

    from backend.perception import routes as perception_routes
    perception_routes.register_routes(app)
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.perception import runner

log = logging.getLogger("forzatek.perception.routes")
_router = APIRouter()


def register_routes(app) -> None:
    """Wire the router onto the FastAPI app."""
    app.include_router(_router, prefix="/api/perception", tags=["perception"])


# ─── Progress (one-shot JSON) ──────────────────────────────────────────────
@_router.get("/progress")
def get_progress() -> JSONResponse:
    """Single GET — same payload as eel.perception_progress()."""
    return JSONResponse(runner.progress())


# ─── Progress (Server-Sent Events) ─────────────────────────────────────────
@_router.get("/progress/stream")
async def progress_stream(request: Request, hz: float = Query(2.0, ge=0.5, le=10.0)):
    """SSE stream of progress JSON.

    Stops automatically when the client disconnects, or when status reaches
    'done' / 'error' / 'cancelled' (with one final flush before close).
    """
    period = max(0.1, 1.0 / float(hz))

    async def gen():
        last_serialized: str | None = None
        terminal_seen_at: float | None = None
        loop = asyncio.get_event_loop()
        while True:
            if await request.is_disconnected():
                break
            payload: Dict[str, Any] = runner.progress()
            data = json.dumps(payload, separators=(",", ":"))
            if data != last_serialized:
                yield f"data: {data}\n\n"
                last_serialized = data

            status = payload.get("status")
            if status in ("done", "error", "cancelled"):
                # Flush one more time then stop.
                if terminal_seen_at is None:
                    terminal_seen_at = loop.time()
                elif loop.time() - terminal_seen_at > 1.5:
                    break

            await asyncio.sleep(period)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Log tail ──────────────────────────────────────────────────────────────
@_router.get("/log")
def get_log(n: int = Query(200, ge=1, le=10000)) -> JSONResponse:
    return JSONResponse({"lines": runner.tail_log(int(n))})


@_router.get("/checkpoints")
def get_checkpoints() -> JSONResponse:
    return JSONResponse({"checkpoints": runner.list_checkpoints()})