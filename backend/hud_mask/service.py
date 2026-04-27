"""
Module 3 — Service layer
========================
Stores HUD masks as JSON arrays of normalized 0..1 rectangles in the
`hud_masks` table. One row per `game_version` (PRIMARY KEY).

mask_json shape:
    [
        {"x": 0.02, "y": 0.78, "w": 0.18, "h": 0.20},   # minimap
        {"x": 0.42, "y": 0.86, "w": 0.16, "h": 0.10},   # speedometer
        ...
    ]

All coordinates are 0..1 floats. The recorder/training code multiplies
by the actual frame dimensions at read-time, so a mask painted at
1920×1080 still works on a 2560×1440 frame.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from backend.core import database

log = logging.getLogger("forzatek.hud_mask.service")


# ─── Validation ─────────────────────────────────────────────────────────────
def _validate_rects(rects: list[dict]) -> list[dict]:
    """Coerce + clamp every rect to {x,y,w,h} floats in [0,1]. Raises ValueError."""
    if not isinstance(rects, list):
        raise ValueError("rects must be a list")
    cleaned: list[dict] = []
    for i, r in enumerate(rects):
        if not isinstance(r, dict):
            raise ValueError(f"rect[{i}] is not an object")
        try:
            x = float(r["x"]); y = float(r["y"])
            w = float(r["w"]); h = float(r["h"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"rect[{i}] missing/invalid x,y,w,h: {e}")
        # Clamp to [0,1]. A rect that goes off the edge is fine — we crop it.
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0 - x, w))
        h = max(0.0, min(1.0 - y, h))
        if w <= 0 or h <= 0:
            # Skip empty rects rather than reject — common when a brush stroke
            # ends right where it began.
            continue
        cleaned.append({"x": x, "y": y, "w": w, "h": h})
    return cleaned


# ─── Public API ─────────────────────────────────────────────────────────────
def save_mask(
    game_version: str,
    rects: list[dict],
    sample_frame_id: Optional[int] = None,
) -> dict:
    """Insert or replace the mask for `game_version`.

    Returns the saved row as a dict.
    """
    if not game_version or not isinstance(game_version, str):
        raise ValueError("game_version must be a non-empty string")
    cleaned = _validate_rects(rects)
    mask_json = json.dumps(cleaned, separators=(",", ":"))
    now = time.time()

    with database.write_conn() as conn:
        conn.execute(
            """
            INSERT INTO hud_masks (game_version, mask_json, sample_frame, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(game_version) DO UPDATE SET
                mask_json    = excluded.mask_json,
                sample_frame = excluded.sample_frame,
                updated_at   = excluded.updated_at
            """,
            (game_version, mask_json, sample_frame_id, now),
        )
    log.info(
        "saved HUD mask for %s: %d rects (sample_frame=%s)",
        game_version, len(cleaned), sample_frame_id,
    )
    # Bust the auto_propagate cache so callers see the new mask immediately.
    from backend.hud_mask import auto_propagate
    auto_propagate.invalidate_cache(game_version)

    return {
        "game_version": game_version,
        "rects": cleaned,
        "sample_frame": sample_frame_id,
        "updated_at": now,
    }


def get_mask(game_version: str) -> Optional[dict]:
    """Return the mask row for `game_version`, or None if not set."""
    with database.read_conn() as conn:
        row = conn.execute(
            "SELECT game_version, mask_json, sample_frame, updated_at "
            "FROM hud_masks WHERE game_version = ?",
            (game_version,),
        ).fetchone()
    if row is None:
        return None
    return {
        "game_version": row["game_version"],
        "rects": json.loads(row["mask_json"]),
        "sample_frame": row["sample_frame"],
        "updated_at": row["updated_at"],
    }


def list_masks() -> list[dict]:
    """All masks. Sorted by game_version for stable UI rendering."""
    with database.read_conn() as conn:
        rows = conn.execute(
            "SELECT game_version, mask_json, sample_frame, updated_at "
            "FROM hud_masks ORDER BY game_version"
        ).fetchall()
    return [
        {
            "game_version": r["game_version"],
            "rect_count":   len(json.loads(r["mask_json"])),
            "sample_frame": r["sample_frame"],
            "updated_at":   r["updated_at"],
        }
        for r in rows
    ]


def delete_mask(game_version: str) -> bool:
    """Remove the mask. Returns True if a row was deleted."""
    with database.write_conn() as conn:
        cur = conn.execute(
            "DELETE FROM hud_masks WHERE game_version = ?",
            (game_version,),
        )
        deleted = cur.rowcount > 0
    if deleted:
        from backend.hud_mask import auto_propagate
        auto_propagate.invalidate_cache(game_version)
        log.info("deleted HUD mask for %s", game_version)
    return deleted


def get_sample_frame(game_version: str) -> Optional[dict]:
    """Pick a random recent frame for `game_version` so the user has
    something to paint on. Returns {frame_id, jpeg_bytes, width, height}
    or None if no frames exist for that version yet.
    """
    with database.read_conn() as conn:
        # Prefer the most recently ingested frame — it'll have the current
        # HUD layout. Random pick from the top 50 keeps it from being
        # the same frame every time.
        row = conn.execute(
            """
            SELECT id, frame_jpeg, width, height
            FROM frames
            WHERE game_version = ?
            ORDER BY id DESC
            LIMIT 50
            """,
            (game_version,),
        ).fetchall()
    if not row:
        return None
    import random
    pick = random.choice(row)
    return {
        "frame_id":   pick["id"],
        "jpeg_bytes": bytes(pick["frame_jpeg"]),
        "width":      pick["width"],
        "height":     pick["height"],
    }


def get_preview_frames(game_version: str, n: int = 4) -> list[dict]:
    """Random N frames for the preview grid below the canvas. Each item
    is {frame_id, jpeg_bytes, width, height}.
    """
    with database.read_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, frame_jpeg, width, height
            FROM frames
            WHERE game_version = ?
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (game_version, n),
        ).fetchall()
    return [
        {
            "frame_id":   r["id"],
            "jpeg_bytes": bytes(r["frame_jpeg"]),
            "width":      r["width"],
            "height":     r["height"],
        }
        for r in rows
    ]


def list_known_game_versions() -> list[str]:
    """Distinct game_version values present in the frames table.
    Used to populate the dropdown in the UI.
    """
    with database.read_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT game_version FROM frames "
            "WHERE game_version IS NOT NULL "
            "ORDER BY game_version"
        ).fetchall()
    return [r["game_version"] for r in rows]