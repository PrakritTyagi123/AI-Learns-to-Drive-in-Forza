"""
Module 5 — Eel API
==================
JS-callable surface for the Train page.

Exposed names:
    perception_stats()                 -> dict
    perception_arch()                  -> dict (classes.describe())
    perception_device_info()           -> dict
    perception_train_start(params)     -> dict
    perception_cancel()                -> bool
    perception_progress()              -> dict
    perception_log(n_lines=200)        -> list[str]
    perception_checkpoints()           -> list[dict]
    perception_activate(ckpt_path)     -> bool
    perception_predict(n=200, top_k=50) -> dict      (active learning trigger)
    perception_history(run_name=None)  -> dict       (chart rehydration on reload)

    perception_cache_status()          -> dict
    perception_cache_build(workers=0)  -> dict
    perception_cache_cancel()          -> bool
    perception_cache_progress()        -> dict
    perception_cache_log(n=200)        -> list[str]
    perception_cache_clear()           -> bool

Long-poll progress streaming uses routes.py SSE — Eel polling at 1 Hz is
fine for the simple "hi I'm alive" pulse from the UI's sparkline.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from backend.perception import predict, runner
from backend.perception.classes import describe

log = logging.getLogger("forzatek.perception.eel_api")


def _safe_float(v: Any) -> Optional[float]:
    """Convert to float if finite, else None. JSON-safe."""
    try:
        f = float(v)
        if f != f:           # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


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

    @eel.expose
    def perception_history(run_name: Optional[str] = None) -> Dict[str, Any]:
        """Return the per-epoch history for a run.

        Used by the Train page to rebuild its chart after a page reload —
        the chart series otherwise live only in JS memory and get wiped on
        refresh while training is mid-run.

        Args:
            run_name: optional. If empty/None, picks the most recently
                modified history file.

        Returns:
            {
                "run_name":    str,
                "epochs":      [int, ...],
                "train_loss":  [float|None, ...],
                "val_loss":    [float|None, ...],
                "miou":        [float|None, ...],
                "iou_road":    [...], "iou_offroad":[...],
                "iou_curb":    [...], "iou_wall":   [...],
                "coverage":    [...],
            }
        Or {"error": "..."} on failure / missing run.
        """
        try:
            from backend.perception.train import history_path

            resolved = (run_name or "").strip()
            if not resolved:
                # Auto-pick: most recently modified _history.json in models dir.
                from backend.core import paths
                mdir = paths.MODELS_DIR
                if not mdir.exists():
                    return {"error": "no models dir"}
                cands = sorted(
                    mdir.glob("*_history.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not cands:
                    return {"error": "no history files"}
                resolved = cands[0].name[:-len("_history.json")]

            p = history_path(resolved)
            if not p.exists():
                return {"error": f"no history for {resolved}", "run_name": resolved}

            raw = json.loads(p.read_text())
            if not isinstance(raw, list):
                return {"error": "unexpected history format", "run_name": resolved}

            out: Dict[str, Any] = {
                "run_name":    resolved,
                "epochs":      [],
                "train_loss":  [],
                "val_loss":    [],
                "miou":        [],
                "iou_road":    [],
                "iou_offroad": [],
                "iou_curb":    [],
                "iou_wall":    [],
                "coverage":    [],
            }
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                ep = entry.get("epoch")
                if ep is None:
                    continue
                v = entry.get("val") or {}
                out["epochs"].append(int(ep))
                out["train_loss"].append(_safe_float(entry.get("train_loss")))
                out["val_loss"].append(_safe_float(v.get("val_loss")))
                out["miou"].append(_safe_float(v.get("miou")))
                out["iou_road"].append(_safe_float(v.get("iou_road")))
                out["iou_offroad"].append(_safe_float(v.get("iou_offroad")))
                out["iou_curb"].append(_safe_float(v.get("iou_curb")))
                out["iou_wall"].append(_safe_float(v.get("iou_wall")))
                out["coverage"].append(_safe_float(v.get("pixel_coverage")))
            return out
        except Exception as e:
            log.exception("perception_history failed: %s", e)
            return {"error": str(e)}

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