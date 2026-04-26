"""
ForzaTek AI v2 — System / Eel API
==================================
The @eel.expose wrappers for the System module. Stays thin; real logic lives
in `system.stats`. JS calls these as `eel.system_stats()()`.

Eel quirks worth knowing:
  - Decorated functions must be at module load time, BEFORE eel.start().
  - Async-style call from JS: `eel.fn(args)(resultCallback)`.
  - Sync-style call from JS:  `await eel.fn(args)()`.
  - We never raise from an exposed function — Eel surfaces exceptions to JS
    poorly. Always return {"ok": False, "error": str(e)} on failure so the
    frontend can display something useful.
"""
from __future__ import annotations

import logging
from typing import Any

from backend.system import stats

log = logging.getLogger("forzatek.system.eel")


def register_eel(eel) -> None:
    """Called once from main.py to register all @eel.expose functions.

    Eel registration must happen before eel.start(). We define the wrappers
    inside this function so they are created exactly once at boot.
    """

    @eel.expose
    def system_stats() -> dict[str, Any]:
        try:
            return {"ok": True, "data": stats.dashboard_snapshot()}
        except Exception as e:
            log.exception("system_stats failed")
            return {"ok": False, "error": str(e)}

    @eel.expose
    def system_health() -> dict[str, Any]:
        try:
            return {"ok": True, "data": stats.health()}
        except Exception as e:
            log.exception("system_health failed")
            return {"ok": False, "error": str(e)}

    log.info("system: registered eel functions [system_stats, system_health]")