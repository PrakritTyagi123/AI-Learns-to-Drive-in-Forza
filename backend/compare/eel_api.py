"""
Module 6 — Compare Eel API
==========================
JS-callable wrappers. All real work is in service.py.

NaN-safety
----------
Python lets `json.dumps` emit `NaN` and `Infinity` literals by default,
but `JSON.parse` in the browser REJECTS them. Eel's websocket layer uses
JSON.parse on the JS side, so any unscrubbed NaN crashes the message
handler and the UI hangs forever waiting for a reply that already came.

We run every return value through `_scrub_nan` which recursively walks
dicts/lists and replaces any non-finite float with `None` (which becomes
`null` in JSON — valid, and the JS code already treats `null` as "—").
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any, Dict, List, Optional

from backend.compare import service
from backend.compare import routes as compare_routes

log = logging.getLogger("forzatek.compare.eel_api")


_summary_cancel_flag = threading.Event()
_summary_lock = threading.Lock()
_summary_running = False


def _cancel_check() -> bool:
    return _summary_cancel_flag.is_set()


# ─── NaN scrubber ─────────────────────────────────────────────────────────
def _scrub_nan(obj: Any) -> Any:
    """Recursively replace non-finite floats with None so JSON.parse on the
    JS side doesn't choke. Walks dicts, lists, tuples. Anything else
    (str, int, bool, None) passes through.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _scrub_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_nan(v) for v in obj]
    return obj


def _shape_compare_result(res: Optional[service.CompareResult]) -> Dict[str, Any]:
    if res is None:
        return {"ok": False, "error": "frame not found or not labeled"}
    try:
        compare_routes.cache_put(res)
    except Exception as e:
        log.warning("cache_put failed: %s", e)
    return _scrub_nan({
        "ok": True,
        "frame_id":     int(res.frame_id),
        "width":        int(res.width),
        "height":       int(res.height),
        "game_version": res.game_version,
        "yolo_metrics": res.yolo_metrics,
        "ours_metrics": res.ours_metrics,
        "disagreement": float(res.disagreement),
        "have_yolo":    bool(res.have_yolo),
        "have_ours":    bool(res.have_ours),
        "have_overlays": {
            "gt":   res.gt_jpeg   is not None,
            "yolo": res.yolo_jpeg is not None,
            "ours": res.ours_jpeg is not None,
        },
    })


def register_eel(eel) -> None:

    @eel.expose
    def compare_status() -> Dict[str, Any]:
        try:
            return _scrub_nan({"ok": True, **service.status()})
        except Exception as e:
            log.exception("compare_status failed")
            return {"ok": False, "error": str(e)}

    @eel.expose
    def compare_warmup_status() -> Dict[str, Any]:
        return _scrub_nan({"ok": True, **service.warmup_status()})

    @eel.expose
    def compare_warmup() -> Dict[str, Any]:
        ws = service.warmup(force=True)
        return _scrub_nan({"ok": True, **ws})

    @eel.expose
    def compare_next(strategy: str = "random",
                     exclude: Optional[List[int]] = None) -> Dict[str, Any]:
        log.info("compare_next called (strategy=%s, exclude=%d items)",
                 strategy, len(exclude or []))
        try:
            fid = service.pick_next_frame(
                strategy=str(strategy or "random"),
                exclude=list(exclude or []),
            )
            if fid is None:
                log.warning("compare_next: no labeled frames available")
                return {"ok": False, "error": "no labeled frames available"}
            log.info("compare_next: picked frame %d", fid)
            res = service.compare_frame(int(fid), render_overlays=True)
            return _shape_compare_result(res)
        except Exception as e:
            log.exception("compare_next failed")
            return {"ok": False, "error": str(e)}

    @eel.expose
    def compare_frame(frame_id: int) -> Dict[str, Any]:
        log.info("compare_frame called (frame_id=%d)", frame_id)
        try:
            res = service.compare_frame(int(frame_id), render_overlays=True)
            return _shape_compare_result(res)
        except Exception as e:
            log.exception("compare_frame failed")
            return {"ok": False, "error": str(e)}

    @eel.expose
    def compare_summary(n: int = 50) -> Dict[str, Any]:
        global _summary_running
        with _summary_lock:
            if _summary_running:
                return {"ok": False, "error": "summary already running"}
            _summary_cancel_flag.clear()
            _summary_running = True
        try:
            result = service.summary(n=int(n), cancel_check=_cancel_check)
            return _scrub_nan({"ok": True,
                                "cancelled": _summary_cancel_flag.is_set(),
                                **result})
        except Exception as e:
            log.exception("compare_summary failed")
            return {"ok": False, "error": str(e)}
        finally:
            with _summary_lock:
                _summary_running = False
            _summary_cancel_flag.clear()

    @eel.expose
    def compare_summary_cancel() -> Dict[str, Any]:
        was_running = _summary_running
        _summary_cancel_flag.set()
        return {"ok": True, "was_running": bool(was_running)}

    @eel.expose
    def compare_reload_perception() -> Dict[str, Any]:
        try:
            ok = service.reload_perception()
            compare_routes.invalidate_cache()
            return {"ok": True, "have_ours": bool(ok)}
        except Exception as e:
            log.exception("compare_reload_perception failed")
            return {"ok": False, "error": str(e)}

    try:
        service.warmup(force=False)
        log.info("Module 6 (Compare): warmup kicked off in background")
    except Exception as e:
        log.exception("compare warmup kickoff failed: %s", e)

    log.info("Module 6 (Compare) Eel API registered.")