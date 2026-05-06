"""
Module 5 — Eel API
==================
JS-callable surface for the Train page.

Exposed names:
    perception_stats()                 -> dict
    perception_arch()                  -> dict (classes.describe())
    perception_train_start(params)     -> dict
    perception_cancel()                -> bool
    perception_progress()              -> dict
    perception_log(n_lines=200)        -> list[str]
    perception_checkpoints()           -> list[dict]
    perception_activate(ckpt_path)     -> bool
    perception_predict(n=200, top_k=50) -> dict      (active learning trigger)

Long-poll progress streaming uses routes.py SSE — Eel polling at 1 Hz is
fine for the simple "hi I'm alive" pulse from the UI's sparkline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from backend.perception import predict, runner
from backend.perception.classes import describe

log = logging.getLogger("forzatek.perception.eel_api")


def register_eel(eel) -> None:
    """Register every @eel.expose with the running Eel instance.
    Called once from backend/main.py during boot.
    """

    @eel.expose
    def perception_stats() -> Dict[str, Any]:
        try:
            stats = runner.perception_stats_summary()
            stats["is_running"] = runner.is_running()
            return stats
        except Exception as e:
            log.exception("perception_stats failed: %s", e)
            return {"error": str(e)}

    @eel.expose
    def perception_arch() -> Dict[str, Any]:
        try:
            return describe()
        except Exception as e:
            log.exception("perception_arch failed: %s", e)
            return {"error": str(e)}

    @eel.expose
    def perception_device_info() -> Dict[str, Any]:
        """Diagnostic: what does torch see on this machine?
        Used by the Train page to show GPU status before training starts.
        """
        try:
            import torch
            available = bool(torch.cuda.is_available())
            return {
                "torch_version":  str(torch.__version__),
                "cuda_built":     str(torch.version.cuda or ""),
                "cuda_available": available,
                "device_count":   int(torch.cuda.device_count()) if available else 0,
                "gpu_name":       str(torch.cuda.get_device_name(0)) if available else "",
            }
        except Exception as e:
            log.exception("perception_device_info failed: %s", e)
            return {"error": str(e)}

    @eel.expose
    def perception_train_start(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            return runner.start_training(params or {})
        except Exception as e:
            log.exception("perception_train_start failed: %s", e)
            return {"started": False, "error": str(e)}

    @eel.expose
    def perception_cancel() -> bool:
        try:
            return bool(runner.cancel_training())
        except Exception as e:
            log.exception("perception_cancel failed: %s", e)
            return False

    @eel.expose
    def perception_progress() -> Dict[str, Any]:
        try:
            return runner.progress()
        except Exception as e:
            log.exception("perception_progress failed: %s", e)
            return {"status": "error", "error": str(e)}

    @eel.expose
    def perception_log(n_lines: int = 200) -> List[str]:
        try:
            return runner.tail_log(int(n_lines))
        except Exception as e:
            log.exception("perception_log failed: %s", e)
            return [f"[log read error: {e}]"]

    @eel.expose
    def perception_checkpoints() -> List[Dict[str, Any]]:
        try:
            return runner.list_checkpoints()
        except Exception as e:
            log.exception("perception_checkpoints failed: %s", e)
            return []

    @eel.expose
    def perception_activate(ckpt_path: str) -> bool:
        try:
            return bool(runner.activate_checkpoint(ckpt_path))
        except Exception as e:
            log.exception("perception_activate failed: %s", e)
            return False

    @eel.expose
    def perception_predict(n_frames: int = 200, top_k: int = 50) -> Dict[str, Any]:
        """Run active-learning batch inference against the active model."""
        try:
            return predict.run_active_learning(
                n_frames=int(n_frames),
                top_k_queue=int(top_k),
            )
        except Exception as e:
            log.exception("perception_predict failed: %s", e)
            return {"ok": False, "error": str(e)}

    # ─── Cache management ──────────────────────────────────────────
    @eel.expose
    def perception_cache_status() -> Dict[str, Any]:
        try:
            return runner.cache_status()
        except Exception as e:
            log.exception("perception_cache_status failed: %s", e)
            return {"error": str(e)}

    @eel.expose
    def perception_cache_build(workers: int = 0) -> Dict[str, Any]:
        try:
            return runner.start_cache_build(workers=int(workers or 0))
        except Exception as e:
            log.exception("perception_cache_build failed: %s", e)
            return {"started": False, "error": str(e)}

    @eel.expose
    def perception_cache_cancel() -> bool:
        try:
            return bool(runner.cancel_cache_build())
        except Exception as e:
            log.exception("perception_cache_cancel failed: %s", e)
            return False

    @eel.expose
    def perception_cache_progress() -> Dict[str, Any]:
        try:
            return runner.cache_build_progress()
        except Exception as e:
            log.exception("perception_cache_progress failed: %s", e)
            return {"status": "error", "error": str(e)}

    @eel.expose
    def perception_cache_log(n_lines: int = 200) -> List[str]:
        try:
            return runner.cache_build_log(int(n_lines))
        except Exception as e:
            log.exception("perception_cache_log failed: %s", e)
            return [f"[log read error: {e}]"]

    @eel.expose
    def perception_cache_clear() -> bool:
        try:
            return bool(runner.cache_clear())
        except Exception as e:
            log.exception("perception_cache_clear failed: %s", e)
            return False