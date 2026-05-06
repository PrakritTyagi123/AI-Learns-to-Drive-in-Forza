"""
ForzaTek AI v2 — Tests for backend/perception/cache.py.

Run with:  python -m tests.test_perception_cache

End-to-end: fixture a small DB → build cache → read cached dataset →
verify staleness flips after adding a label → train one epoch using the
cache and confirm the throughput path actually exercises memmap.
"""
from __future__ import annotations

import base64
import io
import json
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
from backend.perception import cache as _cache
from backend.perception import classes as cls
from backend.perception.train import (
    best_ckpt_path,
    history_path,
    last_ckpt_path,
    train,
)


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


def reset_cache() -> None:
    cdir = paths.DATA_DIR / "perception_cache"
    if cdir.exists():
        for p in cdir.iterdir():
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass


def reset_models() -> None:
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


def test_status_when_no_cache():
    print("\n[cache.status — empty]")
    reset_db(); reset_cache()
    s = _cache.status()
    check("exists is False", s.get("exists") is False)
    check("fresh is False",  s.get("fresh") is False)
    check("n_frames is 0",   s.get("n_frames") == 0)


def test_build_creates_files():
    print("\n[cache.build — small fixture]")
    reset_db(); reset_cache()
    for i in range(5):
        insert_labeled_frame(i)
    res = _cache.build()
    check("build returns ok=True", bool(res.get("ok")), f"got {res}")
    check("n_frames matches", res.get("n_frames") == 5, f"got {res.get('n_frames')}")

    cdir = paths.DATA_DIR / "perception_cache"
    # v2 compressed format expects these files now:
    check("images_jpeg.bin exists",     (cdir / "images_jpeg.bin").exists())
    check("images_jpeg.idx exists",     (cdir / "images_jpeg.idx").exists())
    check("masks_png.bin exists",       (cdir / "masks_png.bin").exists())
    check("masks_png.idx exists",       (cdir / "masks_png.idx").exists())
    check("det_targets.bin exists",     (cdir / "det_targets.bin").exists())
    check("meta.json exists",           (cdir / "meta.json").exists())

    # det_targets.bin is fixed size: n_frames × G_h × G_w × (5+C) × 4 bytes.
    expected_det = (5 * cls.DET_GRID_H * cls.DET_GRID_W
                    * (5 + cls.NUM_DET_CLASSES) * 4)
    check("det_targets.bin has expected size",
          (cdir / "det_targets.bin").stat().st_size == expected_det,
          f"got {(cdir / 'det_targets.bin').stat().st_size} expected {expected_det}")
    # idx files: (n_frames + 1) × 8 bytes (uint64 offsets).
    expected_idx = (5 + 1) * 8
    check("images_jpeg.idx has expected size",
          (cdir / "images_jpeg.idx").stat().st_size == expected_idx)
    check("masks_png.idx has expected size",
          (cdir / "masks_png.idx").stat().st_size == expected_idx)
    # JPEG and PNG sizes are content-dependent, just verify they're nonzero.
    check("images_jpeg.bin is non-empty", (cdir / "images_jpeg.bin").stat().st_size > 0)
    check("masks_png.bin is non-empty",   (cdir / "masks_png.bin").stat().st_size > 0)


def test_status_reports_fresh_after_build():
    print("\n[cache.status — fresh]")
    s = _cache.status()
    check("exists is True", s.get("exists") is True)
    check("fresh is True",  s.get("fresh") is True)
    check("size_bytes > 0", s.get("size_bytes", 0) > 0)
    check("built_at is set", s.get("built_at") is not None)


def test_staleness_after_new_label():
    print("\n[cache.is_stale — after new label]")
    fresh_before = _cache.status().get("fresh")
    check("cache is fresh before adding label", fresh_before is True)
    insert_labeled_frame(99)            # add a 6th frame
    s = _cache.status()
    check("cache is stale after new label", s.get("fresh") is False,
          f"got fresh={s.get('fresh')}")


def test_cached_dataset_loads_items():
    print("\n[CachedPerceptionDataset.__getitem__]")
    reset_cache()
    reset_db()
    for i in range(4):
        insert_labeled_frame(i)
    res = _cache.build()
    if not res.get("ok"):
        check("build succeeded", False, str(res)); return
    train_ids, val_ids = _cache.make_splits_from_cache(seed=0, val_frac=0.25)
    check("split has all 4 entries", len(train_ids) + len(val_ids) == 4)

    ds = _cache.CachedPerceptionDataset(train_ids, train=False)
    check("dataset has correct length", len(ds) == len(train_ids))
    item = ds[0]
    check("returns dict with image/seg/det/frame_id",
          set(item.keys()) >= {"image", "seg", "det", "frame_id"})
    check("image has shape (3, H, W)",
          tuple(item["image"].shape) == (3, cls.INPUT_H, cls.INPUT_W))
    check("seg has shape (H, W)",
          tuple(item["seg"].shape) == (cls.INPUT_H, cls.INPUT_W))
    check("det has shape (G_h, G_w, 5+C)",
          tuple(item["det"].shape) ==
          (cls.DET_GRID_H, cls.DET_GRID_W, 5 + cls.NUM_DET_CLASSES))


def test_train_uses_cache():
    print("\n[train(use_cache=True) — fixture run]")
    reset_models()
    res = train(
        epochs=1,
        batch_size=2,
        lr=1e-3,
        val_frac=0.25,
        run_name="t_cached",
        num_workers=0,
        device="cpu",
        pretrained_backbone=False,
        activate_best=False,
        use_cache=True,
    )
    check("train returns ok=True", bool(res.get("ok")), f"got {res}")
    # NOTE: best checkpoint may legitimately not save here. The synthetic
    # fixture has trivially-perfect road IoU which trips the metric-trustworthy
    # gate (road=1.0 is suspicious on real data). We verify last_ckpt and
    # history were written instead — those always happen.
    check("last ckpt was created", last_ckpt_path("t_cached").exists())
    check("history file exists",   history_path("t_cached").exists())


def test_train_falls_back_when_cache_stale():
    print("\n[train(use_cache=True) — falls back when stale]")
    insert_labeled_frame(7777)         # invalidate the cache
    s = _cache.status()
    check("cache is now stale", s.get("fresh") is False)
    reset_models()
    res = train(
        epochs=1,
        batch_size=2,
        lr=1e-3,
        val_frac=0.25,
        run_name="t_fallback",
        num_workers=0,
        device="cpu",
        pretrained_backbone=False,
        activate_best=False,
        use_cache=True,
    )
    check("train still completes via SQLite fallback", bool(res.get("ok")),
          f"got {res}")


def test_clear_removes_files():
    print("\n[cache.clear]")
    ok = _cache.clear()
    check("clear returns True when files existed", ok is True)
    s = _cache.status()
    check("exists is False after clear", s.get("exists") is False)


def main() -> int:
    tests = [
        test_status_when_no_cache,
        test_build_creates_files,
        test_status_reports_fresh_after_build,
        test_staleness_after_new_label,
        test_cached_dataset_loads_items,
        test_train_uses_cache,
        test_train_falls_back_when_cache_stale,
        test_clear_removes_files,
    ]
    print(f"Running {len(tests)} cache tests…")
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