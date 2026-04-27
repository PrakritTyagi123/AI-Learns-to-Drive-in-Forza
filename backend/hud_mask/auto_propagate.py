"""
Module 3 — Auto-propagate
=========================
The helper that every downstream consumer (Module 4 labeling, Module 5
training dataset, Module 6 compare, Module 8 drive runtime) calls to get
a frame with the HUD masked out.

Stored frames on disk are NEVER modified. We apply the mask at read-time
by multiplying a binary mask array against the frame.

Two public functions:

    apply_mask(image_bgr, game_version) -> np.ndarray
        Returns a copy of the image with HUD regions zeroed out.

    get_mask_array(game_version, h, w) -> np.ndarray | None
        Returns the rasterized mask at the given resolution. None means
        no mask is configured for that game_version (callers should treat
        the entire frame as valid in that case).

The rasterized mask is cached per (game_version, h, w) so we don't
re-rasterize on every frame. Cache is invalidated by service.save_mask()
and service.delete_mask().
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

log = logging.getLogger("forzatek.hud_mask.auto_propagate")

# ─── Cache ──────────────────────────────────────────────────────────────────
# (game_version, h, w) -> np.ndarray of shape (h, w), dtype=uint8, values 0 or 1.
# 0 means HUD (mask out), 1 means valid game pixel (keep).
_CACHE: dict[tuple, np.ndarray] = {}
_CACHE_LOCK = threading.Lock()


def invalidate_cache(game_version: Optional[str] = None) -> None:
    """Drop cached masks. If `game_version` is given, only entries for that
    version. Called by service.save_mask / service.delete_mask.
    """
    with _CACHE_LOCK:
        if game_version is None:
            _CACHE.clear()
        else:
            for key in [k for k in _CACHE if k[0] == game_version]:
                _CACHE.pop(key, None)


def _rasterize(rects: list[dict], h: int, w: int) -> np.ndarray:
    """Build an (h, w) uint8 array. 1 = keep, 0 = HUD."""
    mask = np.ones((h, w), dtype=np.uint8)
    for r in rects:
        x0 = int(round(r["x"] * w))
        y0 = int(round(r["y"] * h))
        x1 = int(round((r["x"] + r["w"]) * w))
        y1 = int(round((r["y"] + r["h"]) * h))
        # Clamp — float rounding can push us 1px out of bounds.
        x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
    return mask


def get_mask_array(game_version: str, h: int, w: int) -> Optional[np.ndarray]:
    """Return the cached or freshly-rasterized mask for this resolution.
    None if no mask exists for `game_version`.
    """
    if not game_version or h <= 0 or w <= 0:
        return None
    key = (game_version, int(h), int(w))
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached

    # Lazy import — avoid circular dependency with service.
    from backend.hud_mask import service
    row = service.get_mask(game_version)
    if row is None or not row["rects"]:
        return None

    mask = _rasterize(row["rects"], int(h), int(w))
    with _CACHE_LOCK:
        _CACHE[key] = mask
    return mask


def apply_mask(image_bgr: np.ndarray, game_version: str) -> np.ndarray:
    """Return a copy of `image_bgr` with HUD regions zeroed out.

    If no mask is configured for `game_version`, returns a copy of the
    input unchanged (callers shouldn't have to special-case that).
    """
    if image_bgr is None or image_bgr.ndim < 2:
        raise ValueError("apply_mask requires a 2D or 3D image array")

    h, w = image_bgr.shape[:2]
    mask = get_mask_array(game_version, h, w)
    if mask is None:
        return image_bgr.copy()

    out = image_bgr.copy()
    if out.ndim == 2:
        out *= mask
    else:
        # Broadcast (h, w) over the channel axis.
        out *= mask[:, :, None]
    return out


def has_mask(game_version: str) -> bool:
    """Cheap check used by training/labeling code that wants to log whether
    a mask is in effect.
    """
    from backend.hud_mask import service
    return service.get_mask(game_version) is not None