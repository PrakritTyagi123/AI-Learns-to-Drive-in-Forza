"""
Module 5 — Training run manager
===============================
Spawns `python -m backend.perception.train` as a subprocess so a crashed
training run cannot kill the Eel app. Watches the progress JSON file written
by the trainer and exposes start/cancel/progress/log to the UI.

Why subprocess and not threading.Thread:
  * CUDA crashes propagate as native faults that take down the process.
  * Re-importing torch in a subprocess avoids "Cuda already initialized"
    issues on Windows when the user re-tries.
  * The UI can stream stdout via the log file without inheriting Python
    state.

Public surface:
    start_training(params: dict) -> dict
    cancel_training() -> bool
    is_running() -> bool
    progress() -> dict
    tail_log(n_lines: int) -> list[str]
    list_checkpoints() -> list[dict]
    activate_checkpoint(path: str) -> bool
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core import database, paths
from backend.perception.train import (
    log_path,
    progress_path,
)

log = logging.getLogger("forzatek.perception.runner")

_PROC: Optional[subprocess.Popen] = None
_LOCK = threading.Lock()

# Separate slot for the cache-build subprocess so we can run a build while
# training is paused and vice versa. They are conceptually distinct.
_CACHE_PROC: Optional[subprocess.Popen] = None
_CACHE_LOCK = threading.Lock()


def _models_dir() -> Path:
    p = paths.MODELS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cancel_flag() -> Path:
    return _models_dir() / "_perception_cancel.flag"


# ─── Lifecycle ─────────────────────────────────────────────────────────────
def is_running() -> bool:
    """True if a training subprocess is currently active."""
    global _PROC
    with _LOCK:
        if _PROC is None:
            return False
        if _PROC.poll() is None:
            return True
        # Reap finished process.
        _PROC = None
        return False


def start_training(params: Dict[str, Any]) -> Dict[str, Any]:
    """Spawn the trainer as a subprocess.

    `params` keys (all optional, defaults match train.py):
        epochs, batch_size, lr, weight_decay, val_frac, run_name,
        resume, num_workers, seed, no_pretrained, device,
        game_versions (list[str]), no_activate (bool)

    Returns dict with `started`/`pid` on success or `error` on failure.
    """
    global _PROC
    if is_running():
        return {"started": False, "error": "training already running",
                "pid": _PROC.pid if _PROC else None}

    args = _build_args(params)

    # Clear any stale cancel flag.
    flag = _cancel_flag()
    if flag.exists():
        try:
            flag.unlink()
        except Exception:
            pass

    # Truncate log so the UI doesn't show old runs.
    try:
        log_path().write_text("")
    except Exception:
        pass

    cmd = [_resolve_python_executable(), "-m", "backend.perception.train"] + args
    log.info("starting trainer: %s", " ".join(cmd))

    # Capture stderr to a file so subprocess crashes are visible. The
    # trainer's normal logging goes through _emit() to the log file, but if
    # the subprocess can't even import its modules (e.g. missing dependency,
    # bad CWD), nothing reaches the log file — that error appears on stderr.
    stderr_path = _models_dir() / "_perception_subproc_stderr.txt"
    try:
        stderr_handle = open(stderr_path, "wb")
    except Exception:
        stderr_handle = subprocess.DEVNULL  # type: ignore[assignment]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,    # trainer writes to log_path itself
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            cwd=str(_project_root()),
            env=os.environ.copy(),
        )
    except Exception as e:
        log.exception("failed to spawn trainer: %s", e)
        return {"started": False, "error": str(e)}

    with _LOCK:
        _PROC = proc

    # Reset progress.json to a clean "starting" state.
    try:
        progress_path().write_text(json.dumps({
            "status": "starting", "ts": time.time(),
            "epochs": int(params.get("epochs", 30)),
            "run_name": params.get("run_name", "perception_v1"),
        }, indent=2))
    except Exception:
        pass

    return {"started": True, "pid": proc.pid}


def cancel_training() -> bool:
    """Signal cancellation and wait briefly for the subprocess to exit."""
    if not is_running():
        return False
    _cancel_flag().write_text(str(time.time()))
    log.info("cancel flag written, waiting for trainer to exit")
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if not is_running():
            return True
        time.sleep(0.5)
    # Hard kill.
    try:
        with _LOCK:
            if _PROC is not None:
                _PROC.terminate()
        log.warning("trainer didn't exit cleanly, terminated")
    except Exception:
        pass
    return True


# ─── Status / progress ─────────────────────────────────────────────────────
def progress() -> Dict[str, Any]:
    """Read progress JSON. Returns {} if no run has been started."""
    p = progress_path()
    if not p.exists():
        return {"status": "idle"}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"status": "unknown", "error": f"progress read failed: {e}"}


def tail_log(n_lines: int = 200) -> List[str]:
    """Return the last N lines of the trainer log."""
    p = log_path()
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        return [l.rstrip("\n") for l in lines[-int(n_lines):]]
    except Exception:
        return []


def tail_subproc_stderr(n_lines: int = 100) -> List[str]:
    """Return the tail of the trainer subprocess stderr file.

    This is what catches errors that prevent the trainer from starting up
    cleanly — missing imports, bad CWD, CUDA init failures. If the trainer's
    own logging never produced output, this is where to look.
    """
    p = _models_dir() / "_perception_subproc_stderr.txt"
    if not p.exists():
        return []
    try:
        # Read as bytes to handle any encoding weirdness, decode lossily.
        raw = p.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        return [l for l in text.splitlines()[-int(n_lines):]]
    except Exception:
        return []


# ─── Checkpoints / activation ──────────────────────────────────────────────
def list_checkpoints() -> List[Dict[str, Any]]:
    """All `.pt` files in MODELS_DIR with sidecar metadata if available."""
    out: List[Dict[str, Any]] = []
    md = _models_dir()
    for p in sorted(md.glob("*.pt")):
        if p.name.startswith("_"):
            continue
        try:
            stat = p.stat()
            entry: Dict[str, Any] = {
                "path":        str(p),
                "name":        p.stem,
                "size_bytes":  int(stat.st_size),
                "created_at":  stat.st_mtime,
            }
        except Exception:
            continue
        # Pull val summary from the matching history file if present.
        hist = md / f"{p.stem.replace('_last','')}_history.json"
        if hist.exists():
            try:
                h = json.loads(hist.read_text())
                if h.get("val"):
                    entry["last_val"] = h["val"][-1]
                if h.get("epoch"):
                    entry["epochs_seen"] = int(h["epoch"][-1])
            except Exception:
                pass
        # Pull is_active flag from DB if a row matches.
        try:
            with database.read_conn() as conn:
                row = conn.execute(
                    "SELECT id, is_active FROM models WHERE path = ?",
                    (str(p),),
                ).fetchone()
            if row is not None:
                entry["model_row_id"] = int(row["id"])
                entry["is_active"] = bool(row["is_active"])
        except Exception:
            pass
        out.append(entry)
    return out


def activate_checkpoint(path: str) -> bool:
    """Set models.is_active=1 for the row whose path matches.

    Inserts a row first if no row exists for this checkpoint yet.
    """
    p = Path(path)
    if not p.exists():
        return False
    now = time.time()
    with database.write_conn() as conn:
        row = conn.execute("SELECT id FROM models WHERE path=?", (str(p),)).fetchone()
        if row is None:
            max_round = conn.execute(
                "SELECT COALESCE(MAX(round_num), 0) AS r FROM models"
            ).fetchone()
            round_num = int(max_round["r"]) + 1 if max_round else 1
            labeled = conn.execute(
                "SELECT COUNT(*) AS n FROM frames WHERE label_status='labeled'"
            ).fetchone()
            conn.execute(
                """INSERT INTO models
                   (name, round_num, path, trained_on, metrics_json,
                    game_versions, is_active, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
                (p.stem, round_num, str(p),
                 int(labeled["n"] if labeled else 0),
                 None, None, now),
            )
        conn.execute("UPDATE models SET is_active=0")
        conn.execute("UPDATE models SET is_active=1 WHERE path=?", (str(p),))
    log.info("activated checkpoint: %s", p)
    return True


# ─── Stats for the readiness panel ─────────────────────────────────────────
def perception_stats_summary() -> Dict[str, Any]:
    """Counts the UI uses to decide if training is ready."""
    with database.read_conn() as conn:
        labeled = conn.execute(
            "SELECT COUNT(*) AS n FROM frames WHERE label_status='labeled'"
        ).fetchone()
        with_seg = conn.execute(
            "SELECT COUNT(DISTINCT frame_id) AS n FROM labels WHERE task='seg'"
        ).fetchone()
        with_det = conn.execute(
            "SELECT COUNT(DISTINCT frame_id) AS n FROM labels WHERE task='det'"
        ).fetchone()
        gv = conn.execute(
            """SELECT game_version, COUNT(*) AS n FROM frames
               WHERE label_status='labeled' GROUP BY game_version"""
        ).fetchall()
        active = conn.execute(
            "SELECT name, path, round_num, metrics_json FROM models "
            "WHERE is_active=1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return {
        "labeled_frames":  int(labeled["n"]) if labeled else 0,
        "frames_with_seg": int(with_seg["n"]) if with_seg else 0,
        "frames_with_det": int(with_det["n"]) if with_det else 0,
        "by_version":      {r["game_version"]: int(r["n"]) for r in gv},
        "active_model":    {
            "name":       active["name"],
            "path":       active["path"],
            "round_num":  int(active["round_num"]),
            "metrics":    json.loads(active["metrics_json"]) if active and active["metrics_json"] else {},
        } if active else None,
    }


# ─── Cache management ─────────────────────────────────────────────────────
def cache_status() -> Dict[str, Any]:
    """UI-friendly status of the memmap cache."""
    from backend.perception import cache as _cache
    s = _cache.status()
    s["build_running"] = cache_is_building()
    return s


def cache_is_building() -> bool:
    global _CACHE_PROC
    with _CACHE_LOCK:
        if _CACHE_PROC is None:
            return False
        if _CACHE_PROC.poll() is None:
            return True
        _CACHE_PROC = None
        return False


def cache_build_progress() -> Dict[str, Any]:
    from backend.perception import cache as _cache
    return _cache.build_progress()


def cache_build_log(n_lines: int = 200) -> List[str]:
    from backend.perception import cache as _cache
    return _cache.build_log_tail(int(n_lines))


def start_cache_build(workers: int = 0) -> Dict[str, Any]:
    """Spawn a subprocess that runs `python -m backend.perception.cache_cli`.

    Args:
        workers: 0 = auto (default). Otherwise the exact number of decode
            worker processes. Passed via FORZATEK_CACHE_WORKERS env var so
            cache.build() doesn't need a different CLI flag.
    """
    global _CACHE_PROC
    if cache_is_building():
        return {"started": False, "error": "cache build already running"}
    if is_running():
        return {"started": False,
                "error": "training is currently running — cancel it first"}

    cmd = [_resolve_python_executable(), "-m", "backend.perception.cache_cli", "--build"]
    env = os.environ.copy()
    if workers and workers > 0:
        env["FORZATEK_CACHE_WORKERS"] = str(int(workers))
    log.info("starting cache build: %s (workers=%s)", " ".join(cmd),
             env.get("FORZATEK_CACHE_WORKERS", "auto"))
    cache_stderr_path = _models_dir() / "_cache_subproc_stderr.txt"
    try:
        cache_stderr_handle = open(cache_stderr_path, "wb")
    except Exception:
        cache_stderr_handle = subprocess.DEVNULL  # type: ignore[assignment]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=cache_stderr_handle,
            stdin=subprocess.DEVNULL,
            cwd=str(_project_root()),
            env=env,
        )
    except Exception as e:
        log.exception("failed to spawn cache builder: %s", e)
        return {"started": False, "error": str(e)}

    with _CACHE_LOCK:
        _CACHE_PROC = proc
    return {"started": True, "pid": proc.pid, "workers": workers or "auto"}


def cancel_cache_build() -> bool:
    if not cache_is_building():
        return False
    from backend.perception import cache as _cache
    _cache.request_cancel()
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if not cache_is_building():
            return True
        time.sleep(0.5)
    try:
        with _CACHE_LOCK:
            if _CACHE_PROC is not None:
                _CACHE_PROC.terminate()
    except Exception:
        pass
    return True


def cache_clear() -> bool:
    from backend.perception import cache as _cache
    if cache_is_building():
        return False
    return _cache.clear()


# ─── Helpers ───────────────────────────────────────────────────────────────
def _project_root() -> Path:
    """The directory containing the `backend` package.

    Walk up from this file until we hit a parent whose name isn't 'backend'.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name != "backend" and (parent / "backend").exists():
            return parent
    return here.parents[2]


def _resolve_python_executable() -> str:
    """Pick the right Python for spawning subprocesses.

    Prefer a `.venv` next to the project root if one exists. Falls back to
    the currently-running interpreter (sys.executable) if no venv is found.

    Why: the Eel app may be launched from system Python (e.g. via a desktop
    shortcut), but the project's actual dependencies (torch, torchvision,
    etc.) live in `.venv`. If we just used sys.executable, the spawned
    subprocess would be missing those deps.
    """
    root = _project_root()
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",   # Windows venv
        root / ".venv" / "bin" / "python",           # POSIX venv
        root / "venv"  / "Scripts" / "python.exe",
        root / "venv"  / "bin" / "python",
    ]
    for cand in candidates:
        if cand.exists():
            log.info("using venv interpreter: %s", cand)
            return str(cand)
    log.info("no venv found at %s; using current interpreter %s",
             root, sys.executable)
    return sys.executable


def _build_args(params: Dict[str, Any]) -> List[str]:
    """Translate the dict from the UI into CLI flags for train.py."""
    args: List[str] = []
    if "epochs" in params:        args += ["--epochs", str(int(params["epochs"]))]
    if "batch_size" in params:    args += ["--batch-size", str(int(params["batch_size"]))]
    if "lr" in params:            args += ["--lr", str(float(params["lr"]))]
    if "weight_decay" in params:  args += ["--weight-decay", str(float(params["weight_decay"]))]
    if "val_frac" in params:      args += ["--val-frac", str(float(params["val_frac"]))]
    if "run_name" in params and params["run_name"]:
        args += ["--run-name", str(params["run_name"])]
    if "resume" in params and params["resume"]:
        args += ["--resume", str(params["resume"])]
    if "num_workers" in params:   args += ["--num-workers", str(int(params["num_workers"]))]
    if "seed" in params:          args += ["--seed", str(int(params["seed"]))]
    if params.get("no_pretrained"):
        args += ["--no-pretrained"]
    if "device" in params and params["device"]:
        args += ["--device", str(params["device"])]
    if params.get("game_versions"):
        args += ["--game-versions", ",".join(str(v) for v in params["game_versions"])]
    if params.get("no_activate"):
        args += ["--no-activate"]
    if params.get("no_cache"):
        args += ["--no-cache"]
    return args