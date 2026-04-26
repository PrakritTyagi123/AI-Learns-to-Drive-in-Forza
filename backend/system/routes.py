"""
ForzaTek AI v2 — System / FastAPI routes
=========================================
The system module's contribution to the FastAPI side server (port 8001).

Most UI traffic goes through Eel, NOT through these routes. This server
exists for things Eel can't do well:
  - Streams (MJPEG, SSE) added by later modules
  - Cheap liveness checks (this module)
  - Routes called from outside the Chromium window (e.g. the global hotkey
    listener, which is not a JS context and therefore can't use Eel)

Keep this file boring. Anything stateful belongs in `stats.py` or other
backend modules.
"""
from __future__ import annotations

from fastapi import FastAPI

from backend.system import stats


def register_routes(app: FastAPI) -> None:
    """Attach system routes to the given FastAPI app."""

    @app.get("/api/system/health")
    def _health():
        """Liveness probe. Cheap; do not add DB calls here."""
        return stats.health()

    @app.get("/api/system/stats")
    def _stats():
        """Same payload as the Eel `system_stats` call. Used by the hotkey
        listener (which is a separate process context) and for ad-hoc curl
        debugging."""
        return stats.dashboard_snapshot()