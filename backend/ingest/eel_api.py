"""
ForzaTek AI v2 — Module 2 / Eel API
====================================
@eel.expose wrappers for the Ingest module. Stays thin — real work is in
`recorder.py` and `video_ingester.py`. Frontend calls these as
`eel.fn_name(args)()`.

Public functions (all JSON-safe):

  Live recording:
    record_start(monitor_index=0)
    record_stop()
    record_stats()

  Video ingest (YouTube + local):
    ingest_register_youtube(url, game_version=None, biome_override=None)
    ingest_register_local(file_path, game_version=None, biome_override=None)
    ingest_start(source_id)
    ingest_cancel(source_id)
    ingest_list_sources()
    ingest_progress(source_id)

  Native dialogs:
    pick_local_video()         -> open OS file picker, return path or None
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from backend.ingest import recorder as _rec
from backend.ingest import video_ingester as _vid

log = logging.getLogger("forzatek.ingest.eel")

_REGISTERED = False


def register_eel(eel) -> None:
    """Called once from `backend/main.py`."""
    global _REGISTERED
    if _REGISTERED:
        return

    # Hook the recorder into the dashboard runtime registry so the
    # Module 1 dashboard's "capture" card lights up while we're recording.
    _rec.register_with_system()

    @eel.expose
    def record_start(monitor_index: int = 0) -> dict:
        try:
            return _rec.RECORDER.start(monitor_index=int(monitor_index))
        except Exception as e:
            log.exception("record_start failed")
            return {"error": str(e), **_rec.RECORDER.get_state()}

    @eel.expose
    def record_stop() -> dict:
        try:
            return _rec.RECORDER.stop()
        except Exception as e:
            log.exception("record_stop failed")
            return {"error": str(e), **_rec.RECORDER.get_state()}

    @eel.expose
    def record_stats() -> dict:
        return _rec.RECORDER.get_state()

    @eel.expose
    def ingest_register_youtube(
        url: str,
        game_version: Optional[str] = None,
        biome_override: Optional[str] = None,
    ) -> dict:
        try:
            return _vid.register_youtube(url, game_version, biome_override)
        except Exception as e:
            log.exception("ingest_register_youtube failed")
            return {"error": str(e)}

    @eel.expose
    def ingest_register_local(
        file_path: str,
        game_version: Optional[str] = None,
        biome_override: Optional[str] = None,
    ) -> dict:
        try:
            return _vid.register_local(file_path, game_version, biome_override)
        except Exception as e:
            log.exception("ingest_register_local failed")
            return {"error": str(e)}

    @eel.expose
    def ingest_start(source_id: int) -> dict:
        try:
            return _vid.start(int(source_id))
        except Exception as e:
            log.exception("ingest_start failed")
            return {"error": str(e), "source_id": source_id}

    @eel.expose
    def ingest_cancel(source_id: int) -> dict:
        try:
            return _vid.cancel(int(source_id))
        except Exception as e:
            log.exception("ingest_cancel failed")
            return {"error": str(e), "source_id": source_id}

    @eel.expose
    def ingest_list_sources() -> list:
        try:
            return _vid.list_sources()
        except Exception as e:
            log.exception("ingest_list_sources failed")
            return [{"error": str(e)}]

    @eel.expose
    def ingest_progress(source_id: int) -> Optional[dict]:
        try:
            return _vid.progress(int(source_id))
        except Exception as e:
            log.exception("ingest_progress failed")
            return {"error": str(e), "source_id": source_id}

    @eel.expose
    def pick_local_video() -> Optional[str]:
        """Open a native OS file dialog for a video file. Returns the
        absolute path, or None if cancelled."""
        return _native_file_picker()

    _REGISTERED = True
    log.info(
        "ingest: registered eel functions "
        "[record_start, record_stop, record_stats, "
        "ingest_register_youtube, ingest_register_local, "
        "ingest_start, ingest_cancel, ingest_list_sources, "
        "ingest_progress, pick_local_video]"
    )


# ─── Native file picker (tkinter) ──────────────────────────────────────────
# tkinter must be touched from the main thread on macOS, but Eel runs
# our handler on a worker thread. We solve this by spinning a short-lived
# Tk root just long enough to show the dialog and return the path.
# On Windows / Linux the threading nuance doesn't matter, but the same
# code path works.
_picker_lock = threading.Lock()


def _native_file_picker() -> Optional[str]:
    with _picker_lock:
        try:
            import tkinter
            from tkinter import filedialog
        except Exception as e:
            log.warning("tkinter not available: %s", e)
            return None

        try:
            root = tkinter.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                path = filedialog.askopenfilename(
                    title="Choose a video file",
                    filetypes=[
                        ("Video files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"),
                        ("All files", "*.*"),
                    ],
                )
            finally:
                root.destroy()
        except Exception as e:
            log.exception("file picker failed")
            return None

        return path or None


__all__ = ["register_eel"]