"""
Module 4 — Auto-labeler
=======================
Background thread that walks every unlabeled frame, runs the prelabeler
(YOLO + SegFormer), and decides for each frame:

    * Trust it    → write labels with provenance='auto_trusted',
                    set frame.label_status='labeled', skip human review.
    * Queue it    → write proposals + push to active_queue, set
                    frame.label_status='queued', wait for human review.

Decision rule (configurable via settings.json):

    accept if  det.min_confidence >= confidence_threshold
           and seg.mean_entropy   <= entropy_threshold

Defaults: confidence_threshold = 0.85, entropy_threshold = 0.50.
A frame with ZERO detected boxes is fine — that just means "no vehicles
or signs," which is the common case on empty roads. The seg entropy
check still gates it.

Public API (all thread-safe):
    start(confidence_threshold, entropy_threshold) -> bool
    cancel() -> bool
    status() -> dict
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from backend.labeling import prelabeler, service

log = logging.getLogger("forzatek.labeling.auto_labeler")


# ─── Worker state ───────────────────────────────────────────────────────────
@dataclass
class _Progress:
    running:       bool   = False
    total:         int    = 0
    processed:     int    = 0
    auto_trusted:  int    = 0
    queued:        int    = 0
    failed:        int    = 0
    started_at:    float  = 0.0
    finished_at:   float  = 0.0
    last_frame_id: int    = 0
    cancel_flag:   bool   = False
    error:         str    = ""
    confidence_threshold: float = 0.85   # YOLO box min-confidence
    entropy_threshold:    float = 0.50   # legacy, not used in road-only mode
    decisive_threshold:   float = 0.85   # NEW: % pixels SegFormer must be decisive on
    batch_size:           int   = 8

    # Live preview — populated after every frame so the UI can poll it.
    preview_frame_id: int    = 0
    preview_decision: str    = ""    # 'trusted' | 'queued'
    preview_seg_b64:  str    = ""    # PNG mask (binary: 1=road, 255=unknown)
    preview_boxes:    list   = None  # list[dict]
    preview_seg_entropy:  float = 0.0
    preview_min_conf:     float = 1.0
    preview_pct_road:     float = 0.0
    preview_pct_decisive: float = 0.0


_STATE = _Progress()
_STATE.preview_boxes = []   # default mutable separately
_STATE_LOCK = threading.RLock()   # reentrant — _snapshot() may be called from inside other locked sections
_THREAD: Optional[threading.Thread] = None


def _set(**kwargs) -> None:
    with _STATE_LOCK:
        for k, v in kwargs.items():
            setattr(_STATE, k, v)


def _bump(**kwargs) -> None:
    with _STATE_LOCK:
        for k, v in kwargs.items():
            setattr(_STATE, k, getattr(_STATE, k) + v)


# ─── Public control ─────────────────────────────────────────────────────────
def start(
    confidence_threshold: float = 0.85,
    entropy_threshold:    float = 0.50,
    batch_size:           int   = 8,
    decisive_threshold:   float = 0.85,
    include_queued:       bool  = False,
) -> dict:
    """Start the worker thread. Idempotent — returns current status if
    already running.

    decisive_threshold: in road-only mode, the fraction of pixels that
        SegFormer must be decisive about (either confidently road OR
        confidently not-road) for the frame to be auto-trusted. 0.85
        means 85% of pixels must be decisive — the rest is the road
        boundary, where uncertainty is normal.

    include_queued: if True, also process frames currently in the active
        queue. The auto_labeler will reset their status to unlabeled before
        starting and the queue will be cleared. Use this after tweaking
        thresholds — frames that were queued under the old rules may now
        auto-trust under the new ones.
    """
    global _THREAD
    with _STATE_LOCK:
        if _STATE.running:
            return _snapshot()

    ok, msg = prelabeler.is_available()
    if not ok:
        return {"started": False, "error": msg}

    # If asked to re-run the queue, reset queued frames back to unlabeled
    # before counting. The worker loop only ever processes 'unlabeled' rows.
    if include_queued:
        try:
            n = service.reset_queued_to_unlabeled()
            log.info("re-queue: %d queued frames moved back to unlabeled", n)
        except Exception as e:
            log.warning("re-queue reset failed: %s", e)

    # Reset progress for a fresh run.
    with _STATE_LOCK:
        _STATE.running              = True
        _STATE.cancel_flag          = False
        _STATE.total                = 0
        _STATE.processed            = 0
        _STATE.auto_trusted         = 0
        _STATE.queued               = 0
        _STATE.failed               = 0
        _STATE.started_at           = time.time()
        _STATE.finished_at          = 0.0
        _STATE.last_frame_id        = 0
        _STATE.error                = ""
        _STATE.confidence_threshold = float(confidence_threshold)
        _STATE.entropy_threshold    = float(entropy_threshold)
        _STATE.decisive_threshold   = float(decisive_threshold)
        _STATE.batch_size           = max(1, int(batch_size))
        # Reset preview.
        _STATE.preview_frame_id    = 0
        _STATE.preview_decision    = ""
        _STATE.preview_seg_b64     = ""
        _STATE.preview_boxes       = []
        _STATE.preview_seg_entropy = 0.0
        _STATE.preview_min_conf    = 1.0
        _STATE.preview_pct_road    = 0.0
        _STATE.preview_pct_decisive = 0.0

    _THREAD = threading.Thread(
        target=_run,
        name="forzatek.auto_labeler",
        daemon=True,
    )
    _THREAD.start()
    log.info(
        "auto_labeler started (yolo>=%.2f, decisive>=%.2f, batch=%d)",
        confidence_threshold, decisive_threshold, batch_size,
    )
    return {"started": True}


def cancel() -> bool:
    """Signal the worker to stop after the current frame. Returns True
    if a worker was running.
    """
    with _STATE_LOCK:
        if not _STATE.running:
            return False
        _STATE.cancel_flag = True
    log.info("auto_labeler cancel requested")
    return True


def status() -> dict:
    """Snapshot of current progress. Safe to call any time."""
    return _snapshot()


def _snapshot() -> dict:
    with _STATE_LOCK:
        elapsed = (
            (_STATE.finished_at or time.time()) - _STATE.started_at
            if _STATE.started_at else 0.0
        )
        rate = _STATE.processed / elapsed if elapsed > 0 else 0.0
        return {
            "running":              _STATE.running,
            "total":                _STATE.total,
            "processed":            _STATE.processed,
            "auto_trusted":         _STATE.auto_trusted,
            "queued":               _STATE.queued,
            "failed":               _STATE.failed,
            "elapsed_sec":          round(elapsed, 1),
            "frames_per_sec":       round(rate, 2),
            "started_at":           _STATE.started_at,
            "finished_at":          _STATE.finished_at,
            "last_frame_id":        _STATE.last_frame_id,
            "error":                _STATE.error,
            "confidence_threshold": _STATE.confidence_threshold,
            "entropy_threshold":    _STATE.entropy_threshold,
            "decisive_threshold":   _STATE.decisive_threshold,
            "batch_size":           _STATE.batch_size,
        }


# ─── Worker loop ────────────────────────────────────────────────────────────
def _run() -> None:
    try:
        ids = service.list_unlabeled_ids(limit=10_000_000)
        _set(total=len(ids))
        with _STATE_LOCK:
            bs = max(1, _STATE.batch_size)
        log.info("auto_labeler: %d unlabeled frames, batch_size=%d", len(ids), bs)

        i = 0
        while i < len(ids):
            with _STATE_LOCK:
                if _STATE.cancel_flag:
                    log.info("auto_labeler: cancelled at offset %d", i)
                    break
            batch_ids = ids[i:i + bs]
            i += bs
            try:
                _process_batch(batch_ids)
            except prelabeler.PrelabelerUnavailable as e:
                _set(error=str(e))
                log.error("auto_labeler unavailable: %s", e)
                break
            except Exception as e:
                _bump(failed=len(batch_ids))
                log.warning(
                    "auto_labeler: batch failed: %s\n%s",
                    e, traceback.format_exc(),
                )

    finally:
        _set(running=False, finished_at=time.time())
        snap = _snapshot()
        log.info(
            "auto_labeler done: %d processed (%d trusted, %d queued, %d failed)",
            snap["processed"], snap["auto_trusted"],
            snap["queued"], snap["failed"],
        )


def _process_batch(frame_ids: list) -> None:
    """Decode N frames, batch-prelabel them, decide trust/queue per frame."""
    if not frame_ids:
        return

    # Step 1: load + decode (CPU). Drop any frames that fail to decode.
    decoded: list = []   # tuples of (frame_id, img_bgr, game_version)
    for fid in frame_ids:
        fr = service.get_frame_for_inference(fid)
        if fr is None:
            _bump(processed=1)
            continue
        arr = np.frombuffer(fr["jpeg_bytes"], dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("could not decode JPEG for frame %d", fid)
            _bump(processed=1, failed=1)
            continue
        decoded.append((fid, img, fr["game_version"]))

    if not decoded:
        return

    # Step 2: batched GPU inference.
    frames     = [d[1] for d in decoded]
    gvs        = [d[2] for d in decoded]
    out_list   = prelabeler.prelabel_batch(frames, gvs)

    # Step 3: per-frame decision + DB writes.
    with _STATE_LOCK:
        ct = _STATE.confidence_threshold
        et = _STATE.entropy_threshold
        # New: minimum fraction of pixels that must be confidently classified
        # (either road or clearly-not-road) for SegFormer's road answer to
        # be trusted on this frame. 0.85 = 85% of pixels must be decisive.
        decisive_thr = _STATE.decisive_threshold

    for (fid, _img, _gv), out in zip(decoded, out_list):
        seg = out["seg"]
        det = out["det"]

        det_conf      = det["min_confidence"] if det["boxes"] else 1.0
        pct_road      = seg.get("pct_road", 0.0)
        pct_decisive  = seg.get("pct_decisive", 0.0)

        service.write_proposal(
            fid, "seg",
            payload={
                "mask_png_b64": seg["mask_png_b64"],
                "classes":      ["unknown", "road"],   # binary now, not 4-class
                "mean_entropy": seg["mean_entropy"],
                "pct_road":     pct_road,
                "pct_decisive": pct_decisive,
            },
            confidence=pct_decisive,           # decisive frames = high confidence
            uncertainty=1.0 - pct_decisive,
        )
        service.write_proposal(
            fid, "det",
            payload={
                "boxes":          det["boxes"],
                "min_confidence": det["min_confidence"],
            },
            confidence=det_conf,
            uncertainty=1.0 - det_conf,
        )

        # Auto-trust rule (road-only mode):
        #   YOLO boxes look fine        AND
        #   SegFormer is decisive about road boundaries
        # Old entropy threshold no longer drives the decision but is still
        # tracked for monitoring.
        boxes_ok = (not det["boxes"]) or det["min_confidence"] >= ct
        seg_ok   = pct_decisive >= decisive_thr

        if boxes_ok and seg_ok:
            decision = "trusted"
            if service.auto_trust_proposal(fid):
                _bump(auto_trusted=1)
        else:
            decision = "queued"
            u = max(1.0 - pct_decisive, 1.0 - det_conf)
            service.enqueue_uncertain(fid, uncertainty=u)
            _bump(queued=1)

        _bump(processed=1)
        _set(last_frame_id=fid)

        # Update live preview state. Only the LAST frame in the batch ends
        # up shown — the UI polls every 500ms so that's fine.
        with _STATE_LOCK:
            _STATE.preview_frame_id    = fid
            _STATE.preview_decision    = decision
            _STATE.preview_seg_b64     = seg["mask_png_b64"]
            _STATE.preview_boxes       = det["boxes"]
            _STATE.preview_seg_entropy = seg["mean_entropy"]
            _STATE.preview_min_conf    = det_conf
            _STATE.preview_pct_road    = pct_road
            _STATE.preview_pct_decisive = pct_decisive


def get_preview() -> dict:
    """Return live-preview state for the UI's right-side mini-canvas."""
    with _STATE_LOCK:
        return {
            "frame_id":     _STATE.preview_frame_id,
            "decision":     _STATE.preview_decision,
            "seg_b64":      _STATE.preview_seg_b64,
            "boxes":        list(_STATE.preview_boxes or []),
            "seg_entropy":  _STATE.preview_seg_entropy,
            "min_conf":     _STATE.preview_min_conf,
            "pct_road":     _STATE.preview_pct_road,
            "pct_decisive": _STATE.preview_pct_decisive,
        }