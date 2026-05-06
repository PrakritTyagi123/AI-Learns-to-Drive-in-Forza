"""
Module 5 — Training loop
========================
Importable as `train(epochs=N, ...)`. Used by:
  * runner.py (UI-launched runs via subprocess)
  * scripts/train_perception.py (CLI runs)

Key behaviors:
  * AdamW + cosine LR schedule.
  * Mixed precision on CUDA.
  * Per-class IoU validation each epoch.
  * Best checkpoint kept by validation road-IoU (the metric the agent
    cares about most). Last checkpoint also kept.
  * Live progress JSON written to MODELS_DIR/_perception_progress.json.
    The runner streams this to the UI without parsing logs.
  * Log lines also go to stdout AND to MODELS_DIR/_perception_log.txt for
    UI tail.

The function never raises into the UI: any error is caught at the top
level, logged, and recorded in the progress file with status='error'.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force UTF-8 stdout/stderr. On Windows, when this script is spawned with
# stdout piped to a file or DEVNULL, sys.stdout defaults to cp1252 which
# can't encode characters like ≈, ✓, … that appear in our log lines.
# This must happen BEFORE any print() calls and before importing modules
# that might log on import.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    # Older Python or non-text stream — fall back to wrapping.
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast    # torch >= 2.4
    _GRAD_SCALER_NEEDS_DEVICE = True
except ImportError:                                # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast
    _GRAD_SCALER_NEEDS_DEVICE = False
from torch.utils.data import DataLoader

from backend.core import database, paths
from backend.perception import cache as _cache
from backend.perception import metrics
from backend.perception.classes import IGNORE_INDEX, NUM_SEG_CLASSES, SEG_CLASSES, describe
from backend.perception.dataset import PerceptionDataset, collate_fn, make_splits
from backend.perception.model import Perception, compute_loss

log = logging.getLogger("forzatek.perception.train")


# ─── GPU-side augmentation (perf path) ────────────────────────────────────
# Augmentation happens here, after the H2D copy, instead of in the dataloader
# workers. This means workers can shrink to 4 (vs 12) without choking the
# GPU — they only do two memmap reads + a uint8 transpose per sample.
# ImageNet normalization stats; matches the MobileNetV3 pretrained backbone.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


class _GPUAugment(nn.Module):
    """All training-time augmentation in one GPU kernel pass.

    Inputs (already on device, NCHW):
        image_u8: (B, 3, H, W) uint8 BGR
        seg_u8:   (B, H, W)    uint8 class ids (0..3 + 255)
        det:      (B, G_h, G_w, 5+C) float32 — pre-encoded WITHOUT flip applied

    Returns:
        image_f: (B, 3, H, W) float32 RGB normalized to ImageNet stats
        seg_l:   (B, H, W)    int64 class ids
        det_f:   (B, G_h, G_w, 5+C) float32 — flipped if random flip happened

    The flip is per-batch (one coin flip applied to all samples in the batch),
    not per-sample. This is fine for a 32-sample batch and saves a per-sample
    branch on the GPU. Effective flip ratio is still ~50%.

    Color jitter is also per-batch for the same reason.
    """

    def __init__(self, train: bool = True, strength: float = 1.0):
        super().__init__()
        self.train_mode = bool(train)
        self.strength = float(strength)
        self.register_buffer("mean", _IMAGENET_MEAN.clone())
        self.register_buffer("std",  _IMAGENET_STD.clone())

    @torch.no_grad()
    def forward(
        self,
        image_in: torch.Tensor,
        seg_in:   torch.Tensor,
        det:      torch.Tensor,
    ):
        # Dual-mode: uint8 BGR from the cached pipeline (the perf path), or
        # already-normalized float RGB from the legacy SQLite pipeline (the
        # fallback path used when the cache is stale).
        if image_in.dtype == torch.uint8:
            # 1) BGR uint8 -> RGB float [0,1].
            image_f = image_in.flip(dims=[1]).float().mul_(1.0 / 255.0)
            need_normalize = True
        else:
            image_f = image_in
            need_normalize = False

        # 2) Augmentation (training only).
        flipped = False
        if self.train_mode and self.strength > 0:
            # Random horizontal flip (per batch).
            if torch.rand(1, device=image_f.device).item() < 0.5 * self.strength:
                image_f = image_f.flip(dims=[3])
                seg_in = seg_in.flip(dims=[2 if seg_in.dim() == 3 else 1])
                det = det.flip(dims=[2])
                det = det.clone()
                det[..., 1] = 1.0 - det[..., 1]
                flipped = True

            # Color jitter (only safe on the uint8 path where image_f ∈ [0,1]).
            if need_normalize and torch.rand(1, device=image_f.device).item() < 0.7 * self.strength:
                bs = 1.0 + (torch.rand(1, device=image_f.device).item() - 0.5) * 0.4 * self.strength
                image_f = (image_f * bs).clamp_(0.0, 1.0)
                shift = ((torch.rand(3, device=image_f.device) - 0.5) * 0.10 * self.strength)
                image_f = (image_f + shift.view(1, 3, 1, 1)).clamp_(0.0, 1.0)

        # 3) ImageNet normalize (only if we converted from uint8).
        if need_normalize:
            image_f = (image_f - self.mean) / self.std

        # 4) seg uint8 -> int64 (CE loss requirement). int64 already? leave it.
        seg_l = seg_in.long() if seg_in.dtype != torch.int64 else seg_in

        return image_f, seg_l, det, flipped


# ─── Paths for live status files ───────────────────────────────────────────
def _models_dir() -> Path:
    p = paths.MODELS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def progress_path() -> Path:
    return _models_dir() / "_perception_progress.json"


def log_path() -> Path:
    return _models_dir() / "_perception_log.txt"


def history_path(run_name: str) -> Path:
    return _models_dir() / f"{run_name}_history.json"


def best_ckpt_path(run_name: str) -> Path:
    return _models_dir() / f"{run_name}.pt"


def last_ckpt_path(run_name: str) -> Path:
    return _models_dir() / f"{run_name}_last.pt"


# ─── Progress + log helpers ────────────────────────────────────────────────
def _write_progress(payload: Dict[str, Any]) -> None:
    """Atomic-ish write so the UI never reads half-written JSON."""
    p = progress_path()
    tmp = p.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, p)
    except Exception as e:
        log.warning("could not write progress: %s", e)


def _append_log(line: str) -> None:
    try:
        with log_path().open("a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        pass


def _emit(line: str) -> None:
    """Print to stdout AND append to log file. Robust to encoding errors —
    on Windows with non-UTF-8 stdout this used to crash with cp1252 errors."""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        # Fall back to ASCII-safe encoding if UTF-8 reconfigure failed.
        try:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass
    _append_log(line)


# ─── Validation pass ───────────────────────────────────────────────────────
@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    aug: nn.Module,
) -> Dict[str, float]:
    model.eval()
    cm = np.zeros((NUM_SEG_CLASSES, NUM_SEG_CLASSES), dtype=np.int64)
    total_loss = 0.0
    n_batches = 0
    # Coverage tracking — needed to detect "labels are too sparse" failures.
    # confusion_matrix drops ignore-index pixels, so its sum() doesn't tell
    # us how much of the dataset was actually labeled. We track it here.
    total_px = 0
    real_px  = 0

    for batch in loader:
        img_u8 = batch["image"].to(device, non_blocking=True)
        seg_u8 = batch["seg"].to(device,   non_blocking=True)
        det_f  = batch["det"].to(device,   non_blocking=True)
        # eval aug normalizes but does NOT flip / jitter.
        img, seg, det, _ = aug(img_u8, seg_u8, det_f)
        if device.type == "cuda":
            img = img.to(memory_format=torch.channels_last)
        out = model(img)
        ld = compute_loss(out, seg, det)
        total_loss += float(ld["total"].item())
        n_batches += 1
        seg_pred = out["seg_logits"].argmax(dim=1)
        cm += metrics.confusion_matrix(seg_pred, seg, NUM_SEG_CLASSES, IGNORE_INDEX)
        bt, br = metrics.count_target_pixels(seg, IGNORE_INDEX)
        total_px += bt
        real_px  += br

    sm = metrics.seg_metrics(cm, SEG_CLASSES,
                             total_pixels=total_px, real_pixels=real_px)
    sm["val_loss"] = total_loss / max(n_batches, 1)
    model.train()
    return sm


# ─── Main entry ────────────────────────────────────────────────────────────
def train(
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_frac: float = 0.15,
    run_name: str = "perception_v1",
    resume: Optional[str] = None,
    num_workers: int = 0,
    seed: int = 0,
    pretrained_backbone: bool = True,
    device: Optional[str] = None,
    game_versions: Optional[List[str]] = None,
    activate_best: bool = True,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Train the perception model. Returns a summary dict.

    `resume` is a checkpoint path to warm-start from. If None, starts fresh
    (with optional ImageNet-pretrained backbone, controlled by pretrained_backbone).

    `activate_best` registers the best checkpoint in `models` table at the
    end. The UI's "Activate" button handles re-pointing later.
    """
    started = time.time()
    log_path().write_text("")          # truncate at start of run

    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _random
    _random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    use_amp = (dev.type == "cuda")

    # Diagnostic readout — prints to log so you can see why CPU was chosen.
    cuda_avail = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_avail else 0
    cuda_name  = torch.cuda.get_device_name(0) if cuda_avail else "n/a"
    cuda_ver   = torch.version.cuda or "n/a"
    _emit(f"[train] torch  v{torch.__version__}  cuda_built={cuda_ver}  "
          f"cuda_available={cuda_avail}  device_count={cuda_count}  gpu0={cuda_name}")
    if dev.type == "cuda" and not cuda_avail:
        _emit("[train] WARN  device='cuda' requested but torch.cuda.is_available()=False; "
              "falling back to CPU. Reinstall torch with CUDA wheel, e.g.: "
              "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        dev = torch.device("cpu")
        use_amp = False
    _emit(f"[train] start  epochs={epochs} bs={batch_size} lr={lr} device={dev}")
    _emit(f"[train] arch   {describe()}")

    # Splits & loaders.
    cache_used = False
    if use_cache:
        cstat = _cache.status()
        if cstat.get("fresh"):
            _emit(f"[train] cache  fresh ({cstat.get('n_frames')} frames, "
                  f"{cstat.get('size_bytes', 0)/1e9:.2f} GB) — using memmap dataset")
            train_ids, val_ids = _cache.make_splits_from_cache(
                seed=seed, val_frac=val_frac, game_versions=game_versions)
            train_set = _cache.CachedPerceptionDataset(train_ids, train=True)
            val_set   = _cache.CachedPerceptionDataset(val_ids,   train=False)
            cache_used = True
        else:
            _emit("[train] cache  not fresh — falling back to live SQLite dataset. "
                  "Build the cache from the Train page for ~10x throughput.")

    if not cache_used:
        train_ids, val_ids = make_splits(seed=seed, val_frac=val_frac,
                                         game_versions=game_versions)
        if not train_ids:
            msg = "no labeled frames available — label some in Module 4 first"
            _emit(f"[train] ERROR {msg}")
            _write_progress({"status": "error", "message": msg, "ts": time.time()})
            return {"ok": False, "error": msg}
        train_set = PerceptionDataset(train_ids, train=True)
        val_set   = PerceptionDataset(val_ids,   train=False)

    if not len(train_set):
        msg = "no labeled frames available — label some in Module 4 first"
        _emit(f"[train] ERROR {msg}")
        _write_progress({"status": "error", "message": msg, "ts": time.time()})
        return {"ok": False, "error": msg}
    _emit(f"[train] data   train={len(train_set)} val={len(val_set)} "
          f"cache={'on' if cache_used else 'off'}")

    # DataLoader tuning — tuned for the cache + GPU-augment pipeline.
    # Worker work per sample: 2 memmap reads + 1 transpose ≈ 0.5 ms.
    # That's 4x cheaper than the old SQLite path, so 4 workers feed a
    # 90%+ utilized GPU without the 12-worker RAM cost.
    if cache_used and num_workers > 4:
        _emit(f"[train] WARN  capping num_workers from {num_workers} to 4 — "
              "the cached pipeline doesn't need more and 12 workers used ~9 GB RAM")
        eff_workers = 4
    else:
        eff_workers = max(0, int(num_workers))

    # Eval batch can be larger — no backward, no optimizer state.
    eval_bs = max(1, batch_size * 2)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=eff_workers,
        pin_memory=use_amp,
        drop_last=False,
        # Persistent workers + large prefetch were tuned for the slow SQLite
        # pipeline. With memmap they just hold extra buffers in RAM. The
        # default prefetch_factor=2 is plenty; persistent_workers=False frees
        # the workers between epochs.
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=max(0, eff_workers // 2),
        pin_memory=use_amp,
        drop_last=False,
        persistent_workers=False,
    ) if len(val_set) > 0 else None

    # ─── RAM safety guard ─────────────────────────────────────
    # Refuse to start if expected peak memory would push the system into
    # swapping. Conservative: assume each worker uses ~700 MB of committed
    # memory (Python + torch + numpy + buffers). Add fixed overhead for the
    # main process model + activations.
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        worker_overhead_gb = eff_workers * 0.7
        main_overhead_gb = 5.0       # model + activations + Eel app etc.
        budget_gb = worker_overhead_gb + main_overhead_gb
        _emit(f"[train] memory  available={avail_gb:.1f} GB  "
              f"estimated peak demand={budget_gb:.1f} GB  "
              f"(workers={eff_workers}, main~{main_overhead_gb:.0f} GB)")
        if budget_gb > avail_gb - 2.0:
            msg = (f"insufficient free RAM: need ~{budget_gb:.0f} GB, "
                   f"only {avail_gb:.0f} GB available. "
                   f"Lower workers or close other apps.")
            _emit(f"[train] ERROR {msg}")
            _write_progress({"status": "error", "message": msg, "ts": time.time()})
            return {"ok": False, "error": msg}
    except ImportError:
        # psutil not installed — silently skip the guard. Not critical.
        pass

    # Model, optimizer, scheduler.
    model = Perception(pretrained_backbone=pretrained_backbone).to(dev)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # GPU augmenter — runs on the same device as the model. Cheap; only the
    # ImageNet mean/std buffers live on the GPU permanently.
    train_aug = _GPUAugment(train=True,  strength=1.0).to(dev)
    eval_aug  = _GPUAugment(train=False, strength=0.0).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * max(1, len(train_loader)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    if _GRAD_SCALER_NEEDS_DEVICE:
        scaler = GradScaler("cuda", enabled=use_amp)
    else:
        scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    history: Dict[str, List[Any]] = {"epoch": [], "train_loss": [], "val": []}
    best_road_iou: float = -1.0

    if resume:
        try:
            ck = torch.load(resume, map_location=dev)
            model.load_state_dict(ck["model"])
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
            if "sched" in ck:
                sched.load_state_dict(ck["sched"])
            start_epoch  = int(ck.get("epoch", 0))
            best_road_iou = float(ck.get("best_road_iou", best_road_iou))
            _emit(f"[train] resume from {resume}, epoch={start_epoch}, "
                  f"best_road_iou={best_road_iou:.4f}")
        except Exception as e:
            _emit(f"[train] WARN resume failed: {e}; starting fresh")

    # Initial progress write.
    _write_progress({
        "status":      "running",
        "epoch":       start_epoch,
        "epochs":      epochs,
        "step":        0,
        "steps_per_ep": len(train_loader),
        "loss":        None,
        "val":         None,
        "started_at":  started,
        "ts":          time.time(),
        "best_road_iou": best_road_iou,
        "run_name":    run_name,
        "device":      str(dev),
    })

    cancel_flag = _models_dir() / "_perception_cancel.flag"
    if cancel_flag.exists():
        cancel_flag.unlink()

    last_summary: Optional[Dict[str, Any]] = None
    step_times: List[float] = []   # rolling window of recent step durations
    try:
        for epoch in range(start_epoch, epochs):
            ep_started = time.time()
            running = 0.0
            n_batches = 0
            for step, batch in enumerate(train_loader):
                step_t0 = time.time()
                if cancel_flag.exists():
                    _emit("[train] cancelled by user — saving last and exiting")
                    raise KeyboardInterrupt("cancelled")

                # uint8 BGR (3,H,W); seg uint8 (H,W); det float32 (G_h,G_w,5+C)
                img_u8 = batch["image"].to(dev, non_blocking=True)
                seg_u8 = batch["seg"].to(dev,   non_blocking=True)
                det_f  = batch["det"].to(dev,   non_blocking=True)

                # GPU-side augmentation + normalization. After this:
                #   img -> float32 RGB (3,H,W) ImageNet-normalized
                #   seg -> int64 (H,W)
                #   det -> float32 (G_h,G_w,5+C) with flip applied if any
                img, seg, det, _flipped = train_aug(img_u8, seg_u8, det_f)
                if dev.type == "cuda":
                    img = img.to(memory_format=torch.channels_last)

                opt.zero_grad(set_to_none=True)
                if use_amp:
                    if _GRAD_SCALER_NEEDS_DEVICE:
                        cm_ctx = autocast("cuda", enabled=True)
                    else:
                        cm_ctx = autocast(enabled=True)
                else:
                    cm_ctx = nullcontext()
                with cm_ctx:
                    out = model(img)
                    ld = compute_loss(out, seg, det)
                if use_amp:
                    scaler.scale(ld["total"]).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    ld["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                running += float(ld["total"].item())
                n_batches += 1

                # Rolling-window throughput. Keep last 50 step durations.
                step_times.append(time.time() - step_t0)
                if len(step_times) > 50:
                    step_times.pop(0)

                if step % 10 == 0:
                    avg_step = sum(step_times) / max(1, len(step_times))
                    sps = (1.0 / avg_step) if avg_step > 0 else 0.0
                    samples_per_sec = sps * batch_size
                    _write_progress({
                        "status":         "running",
                        "epoch":          epoch,
                        "epochs":         epochs,
                        "step":           step + 1,
                        "steps_per_ep":   len(train_loader),
                        "loss":           running / max(1, n_batches),
                        "val":            last_summary,
                        "started_at":     started,
                        "ts":             time.time(),
                        "best_road_iou":  best_road_iou,
                        "run_name":       run_name,
                        "device":         str(dev),
                        "lr":             float(sched.get_last_lr()[0]),
                        "steps_per_sec":  float(sps),
                        "samples_per_sec": float(samples_per_sec),
                        "batch_size":     int(batch_size),
                    })

            train_loss = running / max(1, n_batches)
            ep_dur = time.time() - ep_started

            # Validation.
            if val_loader is not None and len(val_set) > 0:
                val = _validate(model, val_loader, dev, eval_aug)
            else:
                val = {"miou": float("nan"), "val_loss": float("nan"),
                       "iou_road": float("nan"), "pixel_acc": float("nan")}
            last_summary = val

            # Track history.
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["val"].append(val)
            try:
                history_path(run_name).write_text(json.dumps(history, indent=2))
            except Exception:
                pass

            cov = val.get("pixel_coverage", float("nan"))
            cov_str = f"{cov:.1%}" if math.isfinite(cov) else "n/a"

            _emit(
                f"[train] ep {epoch+1:>3d}/{epochs}  "
                f"loss={train_loss:.4f}  val_loss={val.get('val_loss', float('nan')):.4f}  "
                f"miou={val.get('miou', float('nan')):.4f}  "
                f"road={val.get('iou_road', float('nan')):.4f}  "
                f"coverage={cov_str}  ({ep_dur:.1f}s)"
            )
            # Per-class breakdown — easier to debug when something's wrong.
            per_class_bits = []
            for cls in SEG_CLASSES:
                v = val.get(f"iou_{cls}", float("nan"))
                cv = val.get(f"class_coverage_{cls}", 0.0)
                v_str = f"{v:.3f}" if math.isfinite(v) else "n/a"
                per_class_bits.append(f"{cls}={v_str}@{cv:.0%}")
            _emit("[train]            " + "  ".join(per_class_bits))

            # Defensive warnings.
            if val.get("coverage_warning"):
                _emit(f"[train] WARN  pixel_coverage={cov_str} is very low — "
                      "labels are likely too sparse. mIoU/road IoU may be misleading.")
            if val.get("class_collapse_warning"):
                missing = [c for c in SEG_CLASSES
                           if val.get(f"class_coverage_{c}", 0.0) == 0.0]
                _emit(f"[train] WARN  classes with zero coverage in val: {missing} — "
                      "their IoU is undefined and inflates mIoU averages.")
            # Catch the exact pathology that fooled us earlier.
            for cls in SEG_CLASSES:
                v = val.get(f"iou_{cls}", float("nan"))
                if math.isfinite(v) and v >= 0.999:
                    _emit(f"[train] WARN  iou_{cls}={v:.4f} is suspiciously perfect — "
                          "check your labels; this almost never happens with real data.")

            # Save last.
            torch.save({
                "model":          model.state_dict(),
                "opt":            opt.state_dict(),
                "sched":          sched.state_dict(),
                "epoch":          epoch + 1,
                "best_road_iou":  best_road_iou,
                "run_name":       run_name,
                "arch":           describe(),
                "history_tail":   {"epoch": epoch + 1, "train_loss": train_loss, "val": val},
            }, last_ckpt_path(run_name))

            # Save best by road IoU (fall back to mIoU if road is NaN).
            #
            # Sanity gate: refuse to mark a checkpoint as "best" if validation
            # metrics are clearly broken:
            #   (a) pixel coverage <10% (labels too sparse to trust) OR
            #   (b) any class IoU is exactly 1.0 (only happens when target
            #       has zero pixels of a class — gives a meaningless trivial win).
            # We do NOT block on class_collapse_warning alone — small datasets
            # may legitimately have a missing class but still produce useful
            # checkpoints. The warning is logged regardless.
            metric_for_best = val.get("iou_road", float("nan"))
            if not math.isfinite(metric_for_best):
                metric_for_best = val.get("miou", float("-inf"))

            has_perfect_iou = any(
                math.isfinite(val.get(f"iou_{c}", float("nan")))
                and val.get(f"iou_{c}", 0.0) >= 0.999
                for c in SEG_CLASSES
            )
            coverage_ok = (not math.isfinite(cov)) or cov >= 0.10
            metric_trustworthy = coverage_ok and not has_perfect_iou

            if (math.isfinite(metric_for_best)
                    and metric_for_best > best_road_iou
                    and metric_trustworthy):
                best_road_iou = float(metric_for_best)
                torch.save({
                    "model":         model.state_dict(),
                    "epoch":         epoch + 1,
                    "best_road_iou": best_road_iou,
                    "run_name":      run_name,
                    "arch":          describe(),
                    "val_summary":   val,
                }, best_ckpt_path(run_name))
                _emit(f"[train] new best  road_iou={best_road_iou:.4f}")
            elif math.isfinite(metric_for_best) and metric_for_best > best_road_iou and not metric_trustworthy:
                _emit(f"[train] skipping 'best' save  road_iou={metric_for_best:.4f}  "
                      "but val metrics are not trustworthy (low coverage or class collapse)")

            _write_progress({
                "status":         "running",
                "epoch":          epoch + 1,
                "epochs":         epochs,
                "step":           len(train_loader),
                "steps_per_ep":   len(train_loader),
                "loss":           train_loss,
                "val":            val,
                "started_at":     started,
                "ts":             time.time(),
                "best_road_iou":  best_road_iou,
                "run_name":       run_name,
                "device":         str(dev),
                "lr":             float(sched.get_last_lr()[0]),
            })

        # Register best in models table on success.
        if activate_best:
            try:
                _register_model(run_name, best_ckpt_path(run_name),
                                last_summary or {}, game_versions)
            except Exception as e:
                _emit(f"[train] WARN model registration failed: {e}")

        _write_progress({
            "status":          "done",
            "epoch":           epochs,
            "epochs":          epochs,
            "loss":            train_loss,
            "val":             last_summary,
            "best_road_iou":   best_road_iou,
            "run_name":        run_name,
            "ckpt_best":       str(best_ckpt_path(run_name)),
            "ckpt_last":       str(last_ckpt_path(run_name)),
            "started_at":      started,
            "finished_at":     time.time(),
            "ts":              time.time(),
        })
        _emit(f"[train] done   best_road_iou={best_road_iou:.4f}")
        return {"ok": True, "best_road_iou": best_road_iou,
                "ckpt_best": str(best_ckpt_path(run_name)),
                "ckpt_last": str(last_ckpt_path(run_name)),
                "val": last_summary}

    except KeyboardInterrupt:
        _write_progress({"status": "cancelled", "ts": time.time(),
                         "best_road_iou": best_road_iou,
                         "run_name": run_name})
        return {"ok": False, "cancelled": True}
    except Exception as e:
        log.exception("train loop crashed: %s", e)
        _emit(f"[train] ERROR {e}")
        _write_progress({"status": "error", "message": str(e), "ts": time.time(),
                         "run_name": run_name})
        return {"ok": False, "error": str(e)}


def _register_model(
    run_name: str,
    ckpt_path: Path,
    val: Dict[str, Any],
    game_versions: Optional[List[str]],
) -> int:
    """Insert into `models` table with a unique round_num. Does not activate."""
    now = time.time()
    with database.write_conn() as conn:
        max_round = conn.execute(
            "SELECT COALESCE(MAX(round_num), 0) AS r FROM models"
        ).fetchone()
        round_num = int(max_round["r"]) + 1 if max_round else 1
        labeled = conn.execute(
            "SELECT COUNT(*) AS n FROM frames WHERE label_status='labeled'"
        ).fetchone()
        cur = conn.execute(
            """INSERT INTO models
               (name, round_num, path, trained_on, metrics_json,
                game_versions, is_active, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
            (run_name, round_num, str(ckpt_path),
             int(labeled["n"] if labeled else 0),
             json.dumps(val),
             json.dumps(game_versions) if game_versions else None,
             now),
        )
    return int(cur.lastrowid) if hasattr(cur, "lastrowid") else 0


# ─── CLI entry (invoked by runner.py via subprocess) ───────────────────────
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the perception model.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--run-name", type=str, default="perception_v1")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--game-versions", type=str, default=None,
                   help="Comma-separated list, or omit for all.")
    p.add_argument("--no-activate", action="store_true")
    p.add_argument("--no-cache", action="store_true",
                   help="Skip the memmap cache; read from live SQLite (slower).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_cli().parse_args(argv)
    gv = [s.strip() for s in args.game_versions.split(",")] if args.game_versions else None
    res = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        run_name=args.run_name,
        resume=args.resume,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained_backbone=not args.no_pretrained,
        device=args.device,
        game_versions=gv,
        activate_best=not args.no_activate,
        use_cache=not args.no_cache,
    )
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())