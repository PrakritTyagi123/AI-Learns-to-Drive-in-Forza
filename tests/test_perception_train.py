"""
ForzaTek AI v2 — Tests for backend/perception/train.py and runner.py.

Run with:  python -m tests.test_perception_train

We fixture in a small DB with ~6 labeled frames, run a 1-epoch training
loop on CPU (no AMP), and verify:
  * Progress JSON is written and reaches status='done'.
  * Best + last checkpoint files are produced.
  * History JSON has one epoch entry.
  * The checkpoint round-trips: state dict reloads into a fresh Perception.
  * runner.list_checkpoints() finds the produced files.
  * runner.activate_checkpoint() flips the is_active flag in the models row.

We monkey-patch torchvision's pretrained weight download to off so the test
runs without network access.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from backend.core import database, paths
from backend.perception import classes as cls
from backend.perception import runner
from backend.perception.train import (
    best_ckpt_path,
    history_path,
    last_ckpt_path,
    progress_path,
    train,
)
from backend.perception.model import Perception


_PASS = 0
_FAIL = 0


def check(name: str, ok: bool, info: str = "") -> None:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  ✓ {name}")
    else:
        _FAIL += 1
        print(f"  ✗ {name}  {info}")


# ─── Fixture builder ───────────────────────────────────────────────────────
def reset_db() -> None:
    database.init_db()
    with database.write_conn() as conn:
        for tbl in ("labels", "proposals", "active_queue",
                    "hud_masks", "frames", "sources", "models"):
            try:
                conn.execute(f"DELETE FROM {tbl}")
            except Exception:
                pass
        try:
            conn.execute("DELETE FROM sqlite_sequence")
        except Exception:
            pass


def reset_models_dir() -> None:
    md = paths.MODELS_DIR
    md.mkdir(parents=True, exist_ok=True)
    for p in md.glob("*"):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass


def make_jpeg(seed: int, w: int = 320, h: int = 180) -> bytes:
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def make_seg_b64(w: int = 320, h: int = 180, fill: int = 1) -> str:
    from PIL import Image
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = fill
    img = Image.fromarray(mask, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def insert_labeled_frame(idx: int, game_version: str = "FH4") -> int:
    now = time.time()
    with database.write_conn() as conn:
        cur = conn.execute(
            """INSERT INTO frames
               (ts, source_type, game_version, phash, frame_jpeg,
                width, height, label_status)
               VALUES (?, 'live', ?, ?, ?, 320, 180, 'labeled')""",
            (now, game_version, idx, make_jpeg(idx)),
        )
        fid = int(cur.lastrowid)
        seg = {
            "mask_png_b64": make_seg_b64(),
            "classes": {"0": "offroad", "1": "road",
                        "2": "curb", "3": "wall", "255": "unknown"},
            "format": "seg_v1",
        }
        det = {
            "boxes": [{"cls": "vehicle", "x": 0.5, "y": 0.5,
                       "w": 0.1, "h": 0.1, "confidence": 1.0}],
            "format": "det_v1",
        }
        conn.execute(
            "INSERT INTO labels (frame_id, task, data_json, provenance, created_at) "
            "VALUES (?, 'seg', ?, 'manual', ?)",
            (fid, json.dumps(seg), now),
        )
        conn.execute(
            "INSERT INTO labels (frame_id, task, data_json, provenance, created_at) "
            "VALUES (?, 'det', ?, 'manual', ?)",
            (fid, json.dumps(det), now),
        )
    return fid


# ─── Tests ─────────────────────────────────────────────────────────────────
def test_train_no_data_yields_error():
    print("\n[train — no data]")
    reset_db()
    reset_models_dir()
    res = train(
        epochs=1, batch_size=2, lr=1e-3,
        run_name="t_empty",
        num_workers=0, device="cpu",
        pretrained_backbone=False,
        activate_best=False,
    )
    check("returns ok=False with error message",
          res.get("ok") is False and "error" in res, f"got {res}")
    p = progress_path()
    if p.exists():
        prog = json.loads(p.read_text())
        check("progress reports status=error", prog.get("status") == "error",
              f"got {prog.get('status')}")


def test_train_one_epoch_runs():
    print("\n[train — one-epoch fixture]")
    reset_db()
    reset_models_dir()
    for i in range(6):
        insert_labeled_frame(i)

    res = train(
        epochs=1,
        batch_size=2,
        lr=1e-3,
        val_frac=0.2,
        run_name="t_run",
        num_workers=0,
        device="cpu",
        pretrained_backbone=False,
        activate_best=True,
    )

    check("train returns ok=True", bool(res.get("ok")), f"got {res}")
    check("best checkpoint exists", best_ckpt_path("t_run").exists())
    check("last checkpoint exists", last_ckpt_path("t_run").exists())
    check("history file exists", history_path("t_run").exists())

    if history_path("t_run").exists():
        h = json.loads(history_path("t_run").read_text())
        check("history has one epoch", len(h.get("epoch", [])) == 1,
              f"got {len(h.get('epoch', []))}")
        check("history train_loss is finite",
              len(h.get("train_loss", [])) == 1
              and np.isfinite(h["train_loss"][0]))

    if progress_path().exists():
        prog = json.loads(progress_path().read_text())
        check("progress reports status=done",
              prog.get("status") == "done", f"got {prog.get('status')}")
        check("progress reports best_road_iou",
              "best_road_iou" in prog)


def test_checkpoint_round_trip():
    print("\n[checkpoint round-trip]")
    bp = best_ckpt_path("t_run")
    if not bp.exists():
        check("checkpoint exists for round-trip", False, "missing best ckpt")
        return
    ck = torch.load(bp, map_location="cpu")
    check("checkpoint is a dict", isinstance(ck, dict))
    check("checkpoint contains model state_dict",
          "model" in ck and isinstance(ck["model"], dict))
    fresh = Perception(pretrained_backbone=False)
    try:
        fresh.load_state_dict(ck["model"])
        check("state_dict loads into a fresh model", True)
    except Exception as e:
        check("state_dict loads into a fresh model", False, str(e))


def test_runner_lists_and_activates_checkpoint():
    print("\n[runner — list + activate]")
    ckpts = runner.list_checkpoints()
    check("runner finds at least one checkpoint",
          len(ckpts) >= 1, f"got {len(ckpts)}")

    bp = best_ckpt_path("t_run")
    if not bp.exists():
        check("best ckpt present for activation", False)
        return
    ok = runner.activate_checkpoint(str(bp))
    check("activate_checkpoint returns True", bool(ok))

    # Now is_active=1 should be on the row whose path matches.
    with database.read_conn() as conn:
        row = conn.execute(
            "SELECT path, is_active FROM models WHERE is_active=1"
        ).fetchone()
    check("models table has the new active row",
          row is not None and row["path"] == str(bp),
          f"got {dict(row) if row else None}")
    n_active = 0
    with database.read_conn() as conn:
        n_active = int(conn.execute(
            "SELECT COUNT(*) AS n FROM models WHERE is_active=1"
        ).fetchone()["n"])
    check("exactly one model is active", n_active == 1, f"got {n_active}")


def test_runner_progress_and_log_helpers():
    print("\n[runner — progress + log helpers]")
    p = runner.progress()
    check("progress() returns dict", isinstance(p, dict))
    log_lines = runner.tail_log(50)
    check("tail_log returns a list", isinstance(log_lines, list))


def test_runner_stats_summary_shape():
    print("\n[runner — perception_stats_summary]")
    s = runner.perception_stats_summary()
    check("stats has labeled_frames key", "labeled_frames" in s)
    check("stats has by_version dict", isinstance(s.get("by_version"), dict))
    check("stats has active_model field", "active_model" in s)


# ─── Test runner ───────────────────────────────────────────────────────────
def main() -> int:
    tests = [
        test_train_no_data_yields_error,
        test_train_one_epoch_runs,
        test_checkpoint_round_trip,
        test_runner_lists_and_activates_checkpoint,
        test_runner_progress_and_log_helpers,
        test_runner_stats_summary_shape,
    ]
    print(f"Running {len(tests)} train/runner tests…")
    for t in tests:
        try:
            t()
        except Exception as e:
            global _FAIL
            _FAIL += 1
            print(f"  ✗ {t.__name__} crashed: {e}")
            traceback.print_exc()
    print(f"\n— passed: {_PASS}   failed: {_FAIL} —")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())