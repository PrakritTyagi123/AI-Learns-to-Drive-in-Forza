"""
Module 4 — Service
==================
Pure DB logic for the labeling page. No Eel, no FastAPI, no HTTP.

Functions split into four groups:

  * Picker      — `next_frame(strategy)` decides which frame the user
                  sees next. Active queue first (lowest confidence first),
                  then unlabeled frames. Stratification supported.

  * Mutators    — `accept_proposal`, `submit_edit`, `submit_manual`,
                  `unlabel_frame`, `skip_frame`. These write to `labels`
                  and update `frames.label_status`.

  * Reads       — `get_frame_image(id)`, `get_proposals(id)`, `progress()`.

  * Queue mgmt  — `enqueue_uncertain(frame_id, uncertainty)` is called
                  by `auto_labeler.py` for low-confidence proposals;
                  `dequeue(frame_id)` removes once the user is done.

This module DOES read from `hud_mask.auto_propagate` to apply the mask
when serving frame images, but it never writes to `hud_masks`. That's
Module 3's job.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import cv2
import numpy as np

from backend.core import database
from backend.hud_mask import auto_propagate

log = logging.getLogger("forzatek.labeling.service")

# Module 4 owns these label_status string values on the `frames` table:
#   'unlabeled'  | 'queued'  | 'labeled'  | 'skipped'
# (Module 2 inserts every frame as 'unlabeled' by default.)


# ─── Picker ─────────────────────────────────────────────────────────────────
def next_frame(
    game_version: Optional[str] = None,
    biome: Optional[str] = None,
    weather: Optional[str] = None,
    time_of_day: Optional[str] = None,
) -> Optional[dict]:
    """Return the next frame the user should label.

    Priority:
      1. active_queue rows whose frame still has label_status='queued',
         ordered by uncertainty DESC (worst first → most useful to fix).
      2. frames where label_status='unlabeled', ordered by id ASC.

    All optional filters are AND'd together. They apply to BOTH stages
    (queue and unlabeled), so "give me FH5 desert sunny next" works
    regardless of where the frame currently lives.

    Returns:
        {
            "frame_id":    int,
            "from_queue":  bool,
            "uncertainty": float | None,
            "game_version": str | None,
            "width":       int,
            "height":      int,
            "biome":       str | None,
            ...
        }
        or None if nothing matches.
    """
    where_extra: list[str] = []
    params: list[Any] = []

    if game_version:
        where_extra.append("f.game_version = ?")
        params.append(game_version)
    if biome:
        where_extra.append("f.biome = ?")
        params.append(biome)
    if weather:
        where_extra.append("f.weather = ?")
        params.append(weather)
    if time_of_day:
        where_extra.append("f.time_of_day = ?")
        params.append(time_of_day)

    extra_sql = (" AND " + " AND ".join(where_extra)) if where_extra else ""

    with database.read_conn() as conn:
        # 1) Queued first.
        row = conn.execute(
            f"""
            SELECT f.id, f.game_version, f.width, f.height, f.biome,
                   f.weather, f.time_of_day, q.uncertainty
            FROM active_queue q
            JOIN frames f ON f.id = q.frame_id
            WHERE f.label_status = 'queued'{extra_sql}
            ORDER BY q.uncertainty DESC, q.queued_at ASC
            LIMIT 1
            """,
            params,
        ).fetchone()
        if row is not None:
            return {
                "frame_id":     row["id"],
                "from_queue":   True,
                "uncertainty":  row["uncertainty"],
                "game_version": row["game_version"],
                "width":        row["width"],
                "height":       row["height"],
                "biome":        row["biome"],
                "weather":      row["weather"],
                "time_of_day":  row["time_of_day"],
            }

        # 2) Unlabeled.
        row = conn.execute(
            f"""
            SELECT f.id, f.game_version, f.width, f.height, f.biome,
                   f.weather, f.time_of_day
            FROM frames f
            WHERE f.label_status = 'unlabeled'{extra_sql}
            ORDER BY f.id ASC
            LIMIT 1
            """,
            params,
        ).fetchone()
        if row is None:
            return None
        return {
            "frame_id":     row["id"],
            "from_queue":   False,
            "uncertainty":  None,
            "game_version": row["game_version"],
            "width":        row["width"],
            "height":       row["height"],
            "biome":        row["biome"],
            "weather":      row["weather"],
            "time_of_day":  row["time_of_day"],
        }


# ─── Reads ──────────────────────────────────────────────────────────────────
def get_frame_image(frame_id: int, apply_hud_mask: bool = False) -> Optional[dict]:
    """Return the raw JPEG bytes for a frame, optionally HUD-masked.

    Default is `apply_hud_mask=False` — in road-only mode the HUD mask is
    repurposed as a YOLO box-suppression region (see prelabeler._filter_self_boxes),
    NOT as a black overlay. So the labeling canvas should show the original
    frame, not a masked version.

    Pass `apply_hud_mask=True` only if you specifically want the legacy
    blacked-out preview (e.g. for the HUD-mask page's preview grid).
    """
    with database.read_conn() as conn:
        row = conn.execute(
            "SELECT id, frame_jpeg, game_version, width, height "
            "FROM frames WHERE id = ?",
            (frame_id,),
        ).fetchone()
    if row is None:
        return None

    raw_bytes = bytes(row["frame_jpeg"])
    out_bytes = raw_bytes
    gv = row["game_version"]
    # Only re-encode if caller explicitly asked AND a mask exists.
    if apply_hud_mask and gv and auto_propagate.has_mask(gv):
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            masked = auto_propagate.apply_mask(img, gv)
            ok, buf = cv2.imencode(".jpg", masked, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                out_bytes = bytes(buf.tobytes())

    return {
        "frame_id":     row["id"],
        "jpeg_bytes":   out_bytes,
        "game_version": gv,
        "width":        row["width"],
        "height":       row["height"],
    }


def get_proposals(frame_id: int) -> dict:
    """Return the latest seg + det proposals for a frame, if any.

    Returns:
        {
            "seg":       {"mask_png_b64": str, "mean_entropy": float} | None,
            "det":       {"boxes": [...], "min_confidence": float} | None,
            "has_proposals": bool,
        }
    """
    out: dict[str, Any] = {"seg": None, "det": None, "has_proposals": False}
    with database.read_conn() as conn:
        for task in ("seg", "det"):
            row = conn.execute(
                """
                SELECT data_json, confidence, uncertainty
                FROM proposals
                WHERE frame_id = ? AND task = ?
                ORDER BY id DESC LIMIT 1
                """,
                (frame_id, task),
            ).fetchone()
            if row is None:
                continue
            try:
                payload = json.loads(row["data_json"])
            except json.JSONDecodeError:
                log.warning("malformed proposal data for frame %d task %s", frame_id, task)
                continue
            payload["_confidence"]  = row["confidence"]
            payload["_uncertainty"] = row["uncertainty"]
            out[task] = payload
            out["has_proposals"] = True
    return out


def progress() -> dict:
    """Counts for the UI progress bar."""
    with database.read_conn() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        unlabeled = conn.execute(
            "SELECT COUNT(*) FROM frames WHERE label_status='unlabeled'"
        ).fetchone()[0]
        queued    = conn.execute(
            "SELECT COUNT(*) FROM frames WHERE label_status='queued'"
        ).fetchone()[0]
        labeled   = conn.execute(
            "SELECT COUNT(*) FROM frames WHERE label_status='labeled'"
        ).fetchone()[0]
        skipped   = conn.execute(
            "SELECT COUNT(*) FROM frames WHERE label_status='skipped'"
        ).fetchone()[0]

        # Provenance breakdown of completed labels (frame-level — counts
        # each frame at most once, taking the seg row if present, else det).
        rows = conn.execute(
            """
            SELECT provenance, COUNT(DISTINCT frame_id) AS n
            FROM labels
            GROUP BY provenance
            """
        ).fetchall()
    by_prov = {r["provenance"]: r["n"] for r in rows}

    return {
        "total":        total,
        "unlabeled":    unlabeled,
        "queued":       queued,
        "labeled":      labeled,
        "skipped":      skipped,
        "by_provenance": by_prov,
    }


def list_proposals_summary(limit: int = 20) -> list[dict]:
    """For the UI's queue panel: top N most-uncertain queued frames."""
    with database.read_conn() as conn:
        rows = conn.execute(
            """
            SELECT q.frame_id, q.uncertainty, f.game_version, f.biome
            FROM active_queue q
            JOIN frames f ON f.id = q.frame_id
            WHERE f.label_status = 'queued'
            ORDER BY q.uncertainty DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "frame_id":     r["frame_id"],
            "uncertainty":  r["uncertainty"],
            "game_version": r["game_version"],
            "biome":        r["biome"],
        }
        for r in rows
    ]


# ─── Mutators ───────────────────────────────────────────────────────────────
def accept_proposal(frame_id: int) -> bool:
    """User pressed A. Copy the latest proposals into `labels` verbatim
    with provenance='human_accepted', mark frame labeled, drop from queue.
    """
    return _commit_label(
        frame_id,
        seg_b64=None,
        boxes=None,
        provenance="human_accepted",
        from_proposals=True,
    )


def submit_edit(
    frame_id: int,
    seg_mask_png_b64: str,
    boxes: list[dict],
) -> bool:
    """User pressed Space after editing. Save edited seg + boxes."""
    return _commit_label(
        frame_id,
        seg_b64=seg_mask_png_b64,
        boxes=boxes,
        provenance="human_edited",
        from_proposals=False,
    )


def submit_manual(
    frame_id: int,
    seg_mask_png_b64: str,
    boxes: list[dict],
) -> bool:
    """User pressed R then painted from scratch. Save with manual provenance."""
    return _commit_label(
        frame_id,
        seg_b64=seg_mask_png_b64,
        boxes=boxes,
        provenance="manual_from_scratch",
        from_proposals=False,
    )


def unlabel_frame(frame_id: int) -> bool:
    """User pressed U. Remove all labels for this frame, set status back
    to 'unlabeled'. Proposals are kept (model output, cheap to redo).
    """
    now = time.time()
    with database.write_conn() as conn:
        conn.execute("DELETE FROM labels WHERE frame_id = ?", (frame_id,))
        conn.execute(
            "UPDATE frames SET label_status='unlabeled' WHERE id = ?",
            (frame_id,),
        )
        conn.execute("DELETE FROM active_queue WHERE frame_id = ?", (frame_id,))
    log.info("unlabeled frame %d", frame_id)
    return True


def skip_frame(frame_id: int) -> bool:
    """User pressed N. Mark skipped, drop from queue."""
    with database.write_conn() as conn:
        conn.execute(
            "UPDATE frames SET label_status='skipped' WHERE id = ?",
            (frame_id,),
        )
        conn.execute("DELETE FROM active_queue WHERE frame_id = ?", (frame_id,))
    return True


def auto_trust_proposal(frame_id: int) -> bool:
    """Used by `auto_labeler.py` when confidence is high enough to skip
    review. Copies proposals into labels with provenance='auto_trusted'
    and sets status to 'labeled'. NEVER sets a frame to 'queued'.
    """
    return _commit_label(
        frame_id,
        seg_b64=None,
        boxes=None,
        provenance="auto_trusted",
        from_proposals=True,
    )


# ─── Internal commit ────────────────────────────────────────────────────────
def _det_label_payload(boxes: list[dict]) -> dict:
    """Canonical schema for a det label, used by BOTH writers (auto-labeler
    and human edit/manual paths). Module 5 reads this — keep it stable.

    Each box: {cls: 'vehicle'|'sign', x, y, w, h, confidence}
        x,y,w,h are normalized 0..1 (center xy, width, height).
        confidence is YOLO's score for auto-labeled boxes; 1.0 for human-drawn.

    `format` is a version stamp — bump if the box dict shape changes.
    """
    cleaned = []
    for b in (boxes or []):
        try:
            cleaned.append({
                "cls":        str(b["cls"]),
                "x":          float(b["x"]),
                "y":          float(b["y"]),
                "w":          float(b["w"]),
                "h":          float(b["h"]),
                "confidence": float(b.get("confidence", 1.0)),
            })
        except (KeyError, ValueError, TypeError) as e:
            log.warning("dropping malformed box: %s (%s)", b, e)
    return {
        "boxes":  cleaned,
        "format": "det_v1",
    }


def _seg_label_payload(mask_png_b64: str, extras: Optional[dict] = None) -> dict:
    """Canonical schema for a seg label, used by BOTH writers (auto-labeler
    and human edit/manual paths). Module 5 reads this — keep it stable.

    PNG mask values:
        0   = offroad  (only painted by humans)
        1   = road     (auto-labeled or human-painted)
        2   = curb     (only painted by humans)
        3   = wall     (only painted by humans)
        255 = unknown — IGNORE during training (don't compute loss here)

    `format` is a version stamp. Bump it if the meaning of class ids changes
    so Module 5 can detect old vs new labels.
    """
    payload = {
        "mask_png_b64": mask_png_b64,
        "classes": {
            "0":   "offroad",
            "1":   "road",
            "2":   "curb",
            "3":   "wall",
            "255": "unknown",
        },
        "format": "seg_v1",
    }
    if extras:
        payload.update(extras)
    return payload


def _commit_label(
    frame_id: int,
    *,
    seg_b64: Optional[str],
    boxes: Optional[list[dict]],
    provenance: str,
    from_proposals: bool,
) -> bool:
    """Single write path for every label-creating action.

    If `from_proposals`, pulls the latest seg+det proposals out of the
    `proposals` table. Otherwise expects seg_b64 and boxes to be provided.
    """
    now = time.time()

    if from_proposals:
        with database.read_conn() as conn:
            seg_row = conn.execute(
                """SELECT data_json FROM proposals
                   WHERE frame_id = ? AND task = 'seg'
                   ORDER BY id DESC LIMIT 1""",
                (frame_id,),
            ).fetchone()
            det_row = conn.execute(
                """SELECT data_json FROM proposals
                   WHERE frame_id = ? AND task = 'det'
                   ORDER BY id DESC LIMIT 1""",
                (frame_id,),
            ).fetchone()
        if seg_row is None and det_row is None:
            log.warning("no proposals to accept for frame %d", frame_id)
            return False

        # Re-canonicalize the seg payload from the proposal — proposals were
        # written with the auto-labeler's classes field; we rebuild it
        # through _seg_label_payload so the labels table always has the
        # standard schema regardless of which writer produced it.
        seg_payload = None
        if seg_row is not None:
            raw = json.loads(seg_row["data_json"])
            if "mask_png_b64" in raw:
                # Carry forward the auto-labeler's metrics.
                extras = {
                    k: raw[k]
                    for k in ("mean_entropy", "pct_road", "pct_decisive")
                    if k in raw
                }
                seg_payload = _seg_label_payload(raw["mask_png_b64"], extras)

        det_payload = None
        if det_row is not None:
            raw = json.loads(det_row["data_json"])
            det_payload = _det_label_payload(raw.get("boxes", []))
    else:
        if seg_b64 is None and not boxes:
            log.warning("submit with no data for frame %d", frame_id)
            return False
        seg_payload = _seg_label_payload(seg_b64) if seg_b64 else None
        det_payload = _det_label_payload(boxes) if boxes is not None else None

    with database.write_conn() as conn:
        # Replace any prior labels for this frame.
        conn.execute("DELETE FROM labels WHERE frame_id = ?", (frame_id,))
        if seg_payload is not None:
            conn.execute(
                """INSERT INTO labels
                   (frame_id, task, data_json, provenance, created_at)
                   VALUES (?, 'seg', ?, ?, ?)""",
                (frame_id, json.dumps(seg_payload), provenance, now),
            )
        if det_payload is not None:
            conn.execute(
                """INSERT INTO labels
                   (frame_id, task, data_json, provenance, created_at)
                   VALUES (?, 'det', ?, ?, ?)""",
                (frame_id, json.dumps(det_payload), provenance, now),
            )
        conn.execute(
            "UPDATE frames SET label_status='labeled' WHERE id = ?",
            (frame_id,),
        )
        conn.execute("DELETE FROM active_queue WHERE frame_id = ?", (frame_id,))

    log.info("labeled frame %d as %s", frame_id, provenance)
    return True


# ─── Queue management (used by auto_labeler) ────────────────────────────────
def enqueue_uncertain(frame_id: int, uncertainty: float, round_num: int = 0) -> None:
    """Mark a frame as needing human review."""
    now = time.time()
    with database.write_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO active_queue
               (frame_id, uncertainty, queued_at, round_num)
               VALUES (?, ?, ?, ?)""",
            (frame_id, float(uncertainty), now, int(round_num)),
        )
        conn.execute(
            "UPDATE frames SET label_status='queued' WHERE id = ?",
            (frame_id,),
        )


def write_proposal(
    frame_id: int,
    task: str,
    payload: dict,
    confidence: float,
    uncertainty: float,
    model_id: Optional[int] = None,
) -> None:
    """Insert a fresh proposal row. Old proposals for the same (frame, task)
    are kept — we always read the most recent by id DESC.
    """
    if task not in ("seg", "det"):
        raise ValueError(f"task must be 'seg' or 'det', got {task!r}")
    now = time.time()
    with database.write_conn() as conn:
        conn.execute(
            """INSERT INTO proposals
               (frame_id, task, data_json, confidence, uncertainty,
                model_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                frame_id, task, json.dumps(payload),
                float(confidence), float(uncertainty),
                model_id, now,
            ),
        )


def list_unlabeled_ids(limit: int = 1000, include_queued: bool = False) -> list[int]:
    """Used by `auto_labeler.py` to walk frames in batches.

    `include_queued`: if True, also include frames currently in the queue.
    Useful for the "re-run queue" button — re-process queued frames against
    current thresholds; some may now auto-trust if thresholds were loosened.
    """
    statuses = ("unlabeled", "queued") if include_queued else ("unlabeled",)
    placeholders = ",".join("?" for _ in statuses)
    with database.read_conn() as conn:
        rows = conn.execute(
            f"""SELECT id FROM frames
                WHERE label_status IN ({placeholders})
                ORDER BY id ASC LIMIT ?""",
            (*statuses, int(limit)),
        ).fetchall()
    return [r["id"] for r in rows]


def reset_queued_to_unlabeled() -> int:
    """Move all currently-queued frames back to 'unlabeled' and clear the
    active_queue table. Returns the number of frames affected.

    Used by the "re-run queue" button: after the user has tweaked thresholds
    or improved the HUD mask, this lets the auto-labeler re-process the
    queue. Old proposals are kept (they get overwritten on the next run).
    """
    with database.write_conn() as conn:
        n = conn.execute(
            "UPDATE frames SET label_status='unlabeled' WHERE label_status='queued'"
        ).rowcount
        conn.execute("DELETE FROM active_queue")
    log.info("re-queue: reset %d queued frames to unlabeled", n)
    return int(n)


def get_frame_for_inference(frame_id: int) -> Optional[dict]:
    """Helper for `auto_labeler.py` — returns the raw JPEG bytes + game_version
    so the worker can decode and prelabel.
    """
    with database.read_conn() as conn:
        row = conn.execute(
            "SELECT id, frame_jpeg, game_version FROM frames WHERE id = ?",
            (frame_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "frame_id":     row["id"],
        "jpeg_bytes":   bytes(row["frame_jpeg"]),
        "game_version": row["game_version"],
    }