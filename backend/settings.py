"""
ForzaTek AI v2 — Settings
=========================
Reads / writes `data/settings.json`. Single source of truth for every
user-tunable knob: capture FPS, default game version, compute device,
training thresholds, telemetry port, map bin size.

Sits at `backend/settings.py` (NOT inside `backend/core/`) on purpose:
core is the foundation that holds zero policy. Settings are policy
(defaults that the user might want to change), so they live one level up.

Usage:
    from backend.settings import get_settings, save_settings, get
    fps = get("capture_fps")                         # one key
    s   = get_settings()                             # whole dict
    save_settings({**s, "capture_fps": 30})          # partial update
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from backend.core.paths import SETTINGS_PATH, ensure_dirs

# ─────────── Defaults ───────────
# Edit here when you add a new setting. Anything missing from settings.json
# falls back to these on read, so old config files keep working after upgrades.
DEFAULTS: dict[str, Any] = {
    # ── capture ──
    "capture_fps":         15,           # frames per second written to DB
    "capture_backend":     "dxcam",      # dxcam | mss
    "jpeg_quality":        85,
    "frame_resize_height": 720,          # frames are resized to this height before storage

    # ── game ──
    "default_game_version": "fh5",       # fh4 | fh5 | fh6
    "available_game_versions": ["fh4", "fh5", "fh6"],

    # ── compute ──
    "device":              "auto",       # auto | cpu | cuda
    "perception_batch":    8,
    "ppo_batch":           64,

    # ── telemetry ──
    "telemetry_port":      5300,         # UDP port Forza Data Out broadcasts to
    "telemetry_max_age_s": 60,           # rolling buffer window for the telemetry page

    # ── world map ──
    "map_bin_size_m":      5.0,          # world-space bin size for world_map_cells
    "map_flush_interval_s": 2.0,

    # ── labeling ──
    "yolo_confidence":     0.35,
    "active_queue_target": 50,

    # ── drive ──
    "drive_loop_hz":       20,
    "safety_max_steer":    0.85,
    "safety_throttle_cap": 0.80,

    # ── ui ──
    "open_browser_on_start": True,
}

# In-process cache so we don't hit disk every call. Invalidated on save.
_lock = threading.Lock()
_cache: dict[str, Any] | None = None


def _load_from_disk() -> dict[str, Any]:
    """Read settings.json. Missing file → empty dict. Missing keys → defaults."""
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt file: don't crash the app, just fall back to defaults.
        # The user can fix it from the Settings page.
        return {}


def _merged(stored: dict[str, Any]) -> dict[str, Any]:
    """Defaults overlaid with whatever is on disk. Stored values win."""
    return {**DEFAULTS, **stored}


def get_settings() -> dict[str, Any]:
    """Return the full settings dict (defaults + whatever's on disk)."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = _merged(_load_from_disk())
        # Return a copy so callers can't mutate the cache.
        return dict(_cache)


def get(key: str, default: Any = None) -> Any:
    """Convenience: fetch one setting by key."""
    s = get_settings()
    if key in s:
        return s[key]
    return DEFAULTS.get(key, default)


def save_settings(new_values: dict[str, Any]) -> dict[str, Any]:
    """Persist `new_values` (merged on top of whatever's currently saved).

    Only keys that exist in DEFAULTS are accepted — silently drops unknown
    keys so a typo'd field can't pollute the config.

    Returns the resulting full settings dict.
    """
    ensure_dirs()
    accepted = {k: v for k, v in new_values.items() if k in DEFAULTS}

    with _lock:
        stored = _load_from_disk()
        stored.update(accepted)
        SETTINGS_PATH.write_text(
            json.dumps(stored, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        # Invalidate cache so next read picks up the change.
        global _cache
        _cache = _merged(stored)
        return dict(_cache)


def reset_to_defaults() -> dict[str, Any]:
    """Wipe settings.json. Mostly useful for tests and the Settings page's
    'reset' button."""
    with _lock:
        if SETTINGS_PATH.exists():
            SETTINGS_PATH.unlink()
        global _cache
        _cache = dict(DEFAULTS)
        return dict(_cache)


__all__ = [
    "DEFAULTS",
    "get_settings",
    "get",
    "save_settings",
    "reset_to_defaults",
]