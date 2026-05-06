"""
Module 4 — Auto-labeler (4-class mode)
=======================================
Background thread that walks every unlabeled frame, runs the prelabeler
(YOLO + SegFormer), and decides for each frame:

    * Trust it    → write labels with provenance='auto_trusted',
                    set frame.label_status='labeled'.
    * Queue it    → write proposals + push to active_queue, set
                    frame.label_status='queued'.

NEW DECISION RULE (replaces frame-level entropy gating):

    For each pixel, SegFormer's argmax class is kept ONLY if the max
    softmax probability is >= seg_pixel_confidence_threshold (default 0.6),
    AND that class maps to one of our 4 driving classes. Otherwise the
    pixel becomes 255 (ignore at training time).

    A frame is auto-trusted iff:
        (a) all YOLO boxes have confidence >= confidence_threshold, AND
        (b) the fraction of pixels with a real class (not 255) is at least
            min_class_coverage (default 0.30 = 30%).

    The intuition: a frame where SegFormer is confidently wrong gets
    queued (low coverage). A frame where it's confidently right on enough
    pixels gets auto-trusted. The 0.6 pixel threshold is balanced — strict
    enough to avoid garbage labels, loose enough that most frames produce
    actually-trainable masks.

Public API (all thread-safe):
    start(confidence_threshold, min_class_coverage, batch_size,
          include_queued, wipe_existing) -> dict
    cancel() -> bool
    status() -> dict
    get_preview() -> dict
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

    # Configuration (snapshot of what this run was started with)
    confidence_threshold:  float = 0.70   # YOLO box min-confidence
    min_class_coverage:    float = 0.30   # frame-level coverage gate
    pixel_confidence_thr:  float = 0.60   # readback of settings (display only)
    batch_size:            int   = 16

    # Live preview — populated after every frame for the UI poll.
    preview_frame_id: int    = 0
    preview_decision: str    = ""    # 'trusted' | 'queued'
    preview_seg_b64:  str    = ""    # PNG mask, ids 0..3 + 255
    preview_boxes:    list   = None
    preview_seg_entropy:  float = 0.0
    preview_min_conf:     float = 1.0
    preview_pct_real:     float = 0.0
    preview_pct_per_class: dict = None


_STATE = _Progress()
_STATE.preview_boxes = []
_STATE.preview_pct_per_class = {}
_STATE_LOCK = threading.RLock()
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
    confidence_threshold: float = 0.70,
    min_class_coverage:   float = 0.30,
    batch_size:           int   = 16,
    include_queued:       bool  = False,
    wipe_existing:        bool  = False,
) -> dict:
    """Start the worker thread. Idempotent — returns current status if
    already running.

    confidence_threshold : YOLO min box confidence to auto-trust a frame.
    min_class_coverage   : minimum fraction of seg pixels with a real
                           class (not 255) to auto-trust a frame.
    batch_size           : how many frames per GPU batch.
    include_queued       : if True, also re-process active_queue frames.
    wipe_existing        : if True, delete all auto_trusted labels and
                           reset their frames to 'unlabeled' before
                           starting. Manual labels are preserved.
                           USE WITH CARE — this discards prior work.
    """
    global _THREAD
    with _STATE_LOCK:
        if _STATE.running:
            return _snapshot()

    ok, msg = prelabeler.is_available()
    if not ok:
        return {"started": False, "error": msg}

    # Optionally wipe auto_trusted labels first (lets you re-run with new
    # gating logic and start fresh).
    if wipe_existing:
        try:
            n_wiped = service.wipe_auto_trusted()
            log.info("wipe: %d auto_trusted labels deleted, frames reset to unlabeled", n_wiped)
        except Exception as e:
            log.warning("wipe_existing failed: %s", e)

    if include_queued:
        try:
            n = service.reset_queued_to_unlabeled()
            log.info("re-queue: %d queued frames moved back to unlabeled", n)
        except Exception as e:
            log.warning("re-queue reset failed: %s", e)

    # Pull the live pixel-confidence threshold purely for display in the
    # UI; the actual gating uses whatever prelabeler reads from settings.
    try:
        from backend import settings
        pixel_thr = float(settings.get_settings().get("seg_pixel_confidence_threshold", 0.60))
    except Exception:
        pixel_thr = 0.60

    with _STATE_LOCK:
        _STATE.running                = True
        _STATE.cancel_flag            = False
        _STATE.total                  = 0
        _STATE.processed              = 0
        _STATE.auto_trusted           = 0
        _STATE.queued                 = 0
        _STATE.failed                 = 0
        _STATE.started_at             = time.time()
        _STATE.finished_at            = 0.0
        _STATE.last_frame_id          = 0
        _STATE.error                  = ""
        _STATE.confidence_threshold   = float(confidence_threshold)
        _STATE.min_class_coverage     = float(min_class_coverage)
        _STATE.pixel_confidence_thr   = pixel_thr
        _STATE.batch_size             = max(1, int(batch_size))
        _STATE.preview_frame_id       = 0
        _STATE.preview_decision       = ""
        _STATE.preview_seg_b64        = ""
        _STATE.preview_boxes          = []
        _STATE.preview_seg_entropy    = 0.0
        _STATE.preview_min_conf       = 1.0
        _STATE.preview_pct_real       = 0.0
        _STATE.preview_pct_per_class  = {}

    _THREAD = threading.Thread(
        target=_run,
        name="forzatek.auto_labeler",
        daemon=True,
    )
    _THREAD.start()
    log.info(
        "auto_labeler started (yolo>=%.2f, coverage>=%.2f, pixel_thr=%.2f, batch=%d, wipe=%s)",
        confidence_threshold, min_class_coverage, pixel_thr, batch_size, wipe_existing,
    )
    return {"started": True}


def cancel() -> bool:
    with _STATE_LOCK:
        if not _STATE.running:
            return False
        _STATE.cancel_flag = True
    log.info("auto_labeler cancel requested")
    return True


def status() -> dict:
    return _snapshot()


def _snapshot() -> dict:
    with _STATE_LOCK:
        elapsed = (
            (_STATE.finished_at or time.time()) - _STATE.started_at
            if _STATE.started_at else 0.0
        )
        rate = _STATE.processed / elapsed if elapsed > 0 else 0.0
        return {
            "running":               _STATE.running,
            "total":                 _STATE.total,
            "processed":             _STATE.processed,
            "auto_trusted":          _STATE.auto_trusted,
            "queued":                _STATE.queued,
            "failed":                _STATE.failed,
            "elapsed_sec":           round(elapsed, 1),
            "frames_per_sec":        round(rate, 2),
            "started_at":            _STATE.started_at,
            "finished_at":           _STATE.finished_at,
            "last_frame_id":         _STATE.last_frame_id,
            "error":                 _STATE.error,
            "confidence_threshold":  _STATE.confidence_threshold,
            "min_class_coverage":    _STATE.min_class_coverage,
            "pixel_confidence_thr":  _STATE.pixel_confidence_thr,
            "batch_size":            _STATE.batch_size,
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
    if not frame_ids:
        return

    decoded: list = []
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

    frames   = [d[1] for d in decoded]
    gvs      = [d[2] for d in decoded]
    out_list = prelabeler.prelabel_batch(frames, gvs)

    with _STATE_LOCK:
        ct = _STATE.confidence_threshold
        mc = _STATE.min_class_coverage

    for (fid, _img, _gv), out in zip(decoded, out_list):
        seg = out["seg"]
        det = out["det"]

        det_conf = det["min_confidence"] if det["boxes"] else 1.0
        pct_real = float(seg.get("pct_real", 0.0))
        pct_per_class = seg.get("pct_per_class", {}) or {}

        # Write proposals (always, even if we end up auto-trusting — they
        # let the user review what the model thought after the fact).
        service.write_proposal(
            fid, "seg",
            payload={
                "mask_png_b64":  seg["mask_png_b64"],
                "classes":       prelabeler.SEG_CLASS_NAMES,   # 4-class
                "mean_entropy":  seg["mean_entropy"],
                "pct_real":      pct_real,
                "pct_per_class": pct_per_class,
                "format":        "seg_v1",
            },
            confidence=pct_real,           # higher coverage = more confidence in this frame
            uncertainty=1.0 - pct_real,
        )
        service.write_proposal(
            fid, "det",
            payload={
                "boxes":          det["boxes"],
                "min_confidence": det["min_confidence"],
                "format":         "det_v1",
            },
            confidence=det_conf,
            uncertainty=1.0 - det_conf,
        )

        # Auto-trust gate.
        boxes_ok = (not det["boxes"]) or det["min_confidence"] >= ct
        seg_ok   = pct_real >= mc
        decision = "trusted" if (boxes_ok and seg_ok) else "queued"

        if decision == "trusted":
            if service.auto_trust_proposal(fid):
                _bump(auto_trusted=1)
        else:
            u = max(1.0 - pct_real, 1.0 - det_conf)
            service.enqueue_uncertain(fid, uncertainty=u)
            _bump(queued=1)

        _bump(processed=1)
        _set(last_frame_id=fid)

        with _STATE_LOCK:
            _STATE.preview_frame_id      = fid
            _STATE.preview_decision      = decision
            _STATE.preview_seg_b64       = seg["mask_png_b64"]
            _STATE.preview_boxes         = det["boxes"]
            _STATE.preview_seg_entropy   = seg["mean_entropy"]
            _STATE.preview_min_conf      = det_conf
            _STATE.preview_pct_real      = pct_real
            _STATE.preview_pct_per_class = pct_per_class


def get_preview() -> dict:
    with _STATE_LOCK:
        return {
            "frame_id":      _STATE.preview_frame_id,
            "decision":      _STATE.preview_decision,
            "seg_b64":       _STATE.preview_seg_b64,
            "boxes":         list(_STATE.preview_boxes or []),
            "seg_entropy":   _STATE.preview_seg_entropy,
            "min_conf":      _STATE.preview_min_conf,
            "pct_real":      _STATE.preview_pct_real,
            "pct_per_class": dict(_STATE.preview_pct_per_class or {}),
        }