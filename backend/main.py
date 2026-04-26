"""
ForzaTek AI v2 — Application Entry Point
=========================================
This is the only file that imports every module. It wires them together and
launches the desktop window.

Boot sequence (intentional order, do not reorder):

    1. Initialize database (idempotent, safe on every boot)
    2. Start FastAPI on port 8001 in a background thread
       — for streams (MJPEG, SSE) that Eel can't handle well
    3. Initialize Eel pointing at frontend/
    4. Call every module's register_eel(eel) and register_routes(app)
       — later modules append themselves here as they're built
    5. Start the telemetry UDP listener (no-op until Module 7)
    6. Start the global hotkey listener (no-op until handlers are registered)
    7. Open the Chromium window via eel.start("dashboard.html")
    8. On exit, run shutdown hooks (disengage drive, panic gamepad, stop capture)

If anything in the boot sequence fails, we print a clear error and exit
non-zero — this is a desktop app, not a server, so failing loudly is correct.
"""
from __future__ import annotations

import atexit
import logging
import sys
import threading
import time
from pathlib import Path

# ─── Path bootstrap ────────────────────────────────────────────────────────
# Allow `python -m backend.main` from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── Third-party ───────────────────────────────────────────────────────────
import eel
from fastapi import FastAPI
import uvicorn

# ─── Local ─────────────────────────────────────────────────────────────────
from backend.core import database
from backend.core.paths import FRONTEND_DIR

# Module 1 — System
from backend.system import eel_api as system_eel_api
from backend.system import routes  as system_routes
from backend.system import hotkeys as system_hotkeys

# Module 2 — Ingest
from backend.ingest import eel_api as ingest_eel_api
from backend.ingest import routes  as ingest_routes

# ─── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("forzatek.main")

# ─── Configuration ─────────────────────────────────────────────────────────
EEL_PORT      = 8000        # The Chromium window port (Eel's HTTP server)
FASTAPI_PORT  = 8001        # The side port for streams and JSON routes
EEL_HOST      = "localhost"
FASTAPI_HOST  = "127.0.0.1"
START_PAGE    = "dashboard.html"


# ─── Shutdown hooks ────────────────────────────────────────────────────────
# Modules can register cleanup callbacks here. They run in reverse order on
# exit. Drive/gamepad modules will hook into this in later modules.
_SHUTDOWN_HOOKS: list = []


def register_shutdown_hook(fn) -> None:
    """Add a function to run on app exit. Runs in reverse registration order."""
    _SHUTDOWN_HOOKS.append(fn)


def _run_shutdown_hooks() -> None:
    log.info("running shutdown hooks (%d)", len(_SHUTDOWN_HOOKS))
    for fn in reversed(_SHUTDOWN_HOOKS):
        try:
            fn()
        except Exception as e:
            log.error("shutdown hook %s failed: %s", fn.__name__, e)


atexit.register(_run_shutdown_hooks)


# ─── Module 2 cleanup hook ─────────────────────────────────────────────────
def _stop_recorder_on_exit() -> None:
    """If the user closes the window mid-recording, stop the capture loop
    cleanly so the worker thread joins and dxcam releases the desktop
    duplication handle. Otherwise the next launch can fail to acquire it."""
    try:
        from backend.ingest.recorder import RECORDER
        st = RECORDER.get_state()
        if st.get("running"):
            log.info("stopping recorder on exit")
            RECORDER.stop()
    except Exception as e:
        log.warning("recorder stop on exit failed: %s", e)


register_shutdown_hook(_stop_recorder_on_exit)


# ─── FastAPI side server ───────────────────────────────────────────────────
def _build_fastapi_app() -> FastAPI:
    """Create the FastAPI instance and register every module's HTTP routes."""
    app = FastAPI(
        title="ForzaTek AI v2 — side server",
        description="Streams and binary endpoints. UI uses Eel, not this.",
        version="2.0.0",
    )

    # Module 1 — System routes (health check)
    system_routes.register_routes(app)

    # Module 2 — Ingest routes (live MJPEG preview, snapshot, health)
    ingest_routes.register_routes(app)

    # As later modules are built, they register here:
    # hud_mask_routes.register_routes(app)
    # labeling_routes.register_routes(app)
    # ...etc

    return app


def _start_fastapi_thread(app: FastAPI) -> threading.Thread:
    """Launch uvicorn in a daemon thread so Eel owns the main thread."""
    config = uvicorn.Config(
        app,
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        log_level="warning",   # uvicorn is noisy; we have our own logging
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _run():
        try:
            server.run()
        except Exception as e:
            log.error("FastAPI side server crashed: %s", e)

    t = threading.Thread(target=_run, daemon=True, name="fastapi-side")
    t.start()
    # Give uvicorn a moment to bind the port before Eel starts.
    # If we don't, hotkeys that POST to FastAPI on first key-press will fail.
    for _ in range(20):
        if server.started:
            break
        time.sleep(0.1)
    if not server.started:
        log.warning("FastAPI side server slow to start; continuing anyway")
    else:
        log.info("FastAPI side server up on http://%s:%d", FASTAPI_HOST, FASTAPI_PORT)
    return t


# ─── Eel registration ──────────────────────────────────────────────────────
def _register_all_eel_apis() -> None:
    """Wire every module's @eel.expose functions.

    Each module owns a `register_eel(eel)` function. Calling it imports the
    decorated functions into Eel's exposed-functions registry.
    """
    # Module 1 — System
    system_eel_api.register_eel(eel)

    # Module 2 — Ingest (record_*, ingest_*, pick_local_video)
    ingest_eel_api.register_eel(eel)

    # Later modules register here:
    # hud_mask_eel_api.register_eel(eel)
    # labeling_eel_api.register_eel(eel)
    # ...etc


# ─── Boot ──────────────────────────────────────────────────────────────────
def main() -> int:
    log.info("ForzaTek AI v2 — booting")

    # 1. Database
    try:
        database.init_db()
        log.info("database OK")
    except Exception as e:
        log.error("database init failed: %s", e)
        return 1

    # 2. FastAPI side server
    fastapi_app = _build_fastapi_app()
    _start_fastapi_thread(fastapi_app)

    # 3. Eel — point at frontend folder
    if not FRONTEND_DIR.exists():
        log.error("frontend directory not found: %s", FRONTEND_DIR)
        return 1
    eel.init(str(FRONTEND_DIR))
    log.info("Eel initialized; serving from %s", FRONTEND_DIR)

    # 4. Register every module's Eel functions
    _register_all_eel_apis()

    # 5. Telemetry UDP listener — placeholder until Module 7
    # telemetry_listener.start()  # noqa

    # 6. Global hotkeys — Module 1 starts the listener; later modules
    #    register their handlers via system_hotkeys.register(key, fn).
    try:
        system_hotkeys.start()
    except Exception as e:
        # Hotkeys are nice-to-have, not required. Don't crash boot for them.
        log.warning("hotkey listener could not start: %s (non-fatal)", e)

    # 7. Open the Chromium window. This blocks until the window closes.
    log.info("opening window: http://%s:%d/%s", EEL_HOST, EEL_PORT, START_PAGE)
    try:
        eel.start(
            START_PAGE,
            host=EEL_HOST,
            port=EEL_PORT,
            size=(1400, 900),
            block=True,
        )
    except (SystemExit, KeyboardInterrupt):
        log.info("window closed by user")
    except Exception as e:
        log.error("Eel exited with error: %s", e)
        return 1
    finally:
        # 8. atexit will fire shutdown hooks. Nothing else to do here.
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())