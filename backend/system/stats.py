"""
ForzaTek AI v2 — System / Stats
================================
The single rollup query that powers the dashboard.

Pure function. Reads from `backend.core.database` only. No side effects, no
HTTP, no Eel. Calling `dashboard_snapshot()` 100 times in a loop is safe and
cheap — the SQLite reads are all indexed counts.

The dashboard polls this every 2 seconds. Later modules contribute their own
state (capture state, gamepad state) via the `runtime_state()` registry —
they register a callback at boot, and we ask them for their current state
when a snapshot is taken. Module 1 has no contributors yet, so those fields
return idle defaults.
"""
from __future__ import annotations

import time
from typing import Callable

from backend.core import database

# ─── Runtime state registry ────────────────────────────────────────────────
# Later modules register callbacks here so the dashboard can ask them
# "what's your current state?" without us needing to import them directly.
# Module 1 ships with the registry empty.
#
# A callback returns a dict; we merge them all under namespaced keys.
#
# Example (from Module 2 when it's built):
#     stats.register_state("capture", recorder.get_state)
#
_STATE_PROVIDERS: dict[str, Callable[[], dict]] = {}


def register_state(namespace: str, provider: Callable[[], dict]) -> None:
    """Modules call this at boot to expose their runtime state."""
    if namespace in _STATE_PROVIDERS:
        raise ValueError(f"state namespace {namespace!r} already registered")
    _STATE_PROVIDERS[namespace] = provider


def _safe_call(provider: Callable[[], dict]) -> dict:
    """Don't let one broken provider take down the dashboard."""
    try:
        out = provider()
        return out if isinstance(out, dict) else {"error": "provider returned non-dict"}
    except Exception as e:
        return {"error": str(e)}


# ─── The dashboard rollup ──────────────────────────────────────────────────
def dashboard_snapshot() -> dict:
    """One-shot snapshot for the dashboard. Cheap. Call as often as you like.

    Returns the same `overall_stats()` payload from the database, plus:
      - server_ts:      epoch seconds at the moment of the snapshot
      - runtime:        {namespace: provider_dict, ...} from registered modules
                        ("capture", "gamepad", "telemetry", etc. — empty for
                        Module 1 since no module has registered yet)
    """
    base = database.overall_stats()

    runtime: dict[str, dict] = {}
    for ns, provider in _STATE_PROVIDERS.items():
        runtime[ns] = _safe_call(provider)

    # Module 1 default placeholders so the dashboard cards always have
    # *something* to render. Later modules will overwrite these by
    # registering a real provider with the same namespace.
    runtime.setdefault("capture",  {"state": "idle"})
    runtime.setdefault("gamepad",  {"state": "disconnected"})
    runtime.setdefault("telemetry", {"state": "idle", "stale_sec": None})

    return {
        "server_ts": time.time(),
        **base,
        "runtime": runtime,
    }


def health() -> dict:
    """Cheap liveness payload — fixed-cost, no DB queries."""
    return {
        "ok": True,
        "service": "forzatek-v2",
        "ts": time.time(),
    }


if __name__ == "__main__":
    # Manual: `python -m backend.system.stats`
    import json
    print(json.dumps(dashboard_snapshot(), indent=2, default=str))