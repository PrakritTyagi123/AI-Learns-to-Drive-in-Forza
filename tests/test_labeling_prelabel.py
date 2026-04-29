"""
Module 4 — Prelabeler interface tests
=====================================
We can't reliably load YOLO/SegFormer in CI (huge weights, GPU-dependent
behavior, network downloads). So this test file checks:

    1. is_available() returns the right answer based on the import state.
    2. prelabel_frame validates its input.
    3. prelabel_frame applies the HUD mask before running models —
       verified by stubbing the inner _run_yolo and _run_seg functions
       and asserting they receive the masked image.
    4. The COCO->ours and Cityscapes->ours mappings are correct.

Run from project root:
    python -m tests.test_labeling_prelabel
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# ─── Path bootstrap ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMPDIR = Path(tempfile.mkdtemp(prefix="ftk_m4_pre_"))
_TMP_DB = _TMPDIR / "forzatek.db"
os.environ["FORZATEK_DATA_DIR"] = str(_TMPDIR)

import numpy as np  # noqa: E402

from backend.core import database, paths  # noqa: E402

paths.DB_PATH = _TMP_DB
if hasattr(paths, "DATA_DIR"):
    paths.DATA_DIR = _TMPDIR
if hasattr(database, "DB_PATH"):
    database.DB_PATH = _TMP_DB

_resolved = Path(getattr(database, "DB_PATH", paths.DB_PATH)).resolve()
if str(_TMPDIR) not in str(_resolved):
    print(f"REFUSING TO RUN: test would write to {_resolved}")
    sys.exit(2)

database.init_db()

from backend.hud_mask import service as hud_service  # noqa: E402
from backend.labeling import prelabeler  # noqa: E402


# ─── Test runner ────────────────────────────────────────────────────────────
_PASSED = 0
_FAILED = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASSED, _FAILED
    if cond:
        _PASSED += 1
        print(f"✓ {label}")
    else:
        _FAILED += 1
        print(f"✗ {label}  {detail}")


def reset_db() -> None:
    """Wipe via DELETE (Windows-safe) and reset autoincrement counters."""
    database.init_db()
    with database.write_conn() as conn:
        for tbl in ("labels", "proposals", "active_queue", "hud_masks", "frames", "sources"):
            try:
                conn.execute(f"DELETE FROM {tbl}")
            except Exception:
                pass
        try:
            conn.execute("DELETE FROM sqlite_sequence")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# 1. is_available — accurate based on import state
# ═══════════════════════════════════════════════════════════════════════════
ok, msg = prelabeler.is_available()
# We can't assert ok=True or False; just that it's a sane (bool, str).
check("is_available returns (bool, str)",
      isinstance(ok, bool) and isinstance(msg, str))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Input validation
# ═══════════════════════════════════════════════════════════════════════════
try:
    prelabeler.prelabel_frame(None, "FH4")
    check("prelabel_frame rejects None", False)
except ValueError:
    check("prelabel_frame rejects None", True)

try:
    prelabeler.prelabel_frame(np.zeros((10, 10), dtype=np.uint8), "FH4")
    check("prelabel_frame rejects 2D array", False)
except ValueError:
    check("prelabel_frame rejects 2D array", True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. ROAD-ONLY MODE: HUD mask is NOT applied to inference inputs.
#    Both YOLO and SegFormer see the raw frame. The HUD mask is repurposed
#    as a YOLO box-suppression region (verified in test 3b below).
# ═══════════════════════════════════════════════════════════════════════════
reset_db()

# Save a HUD mask covering the bottom-right quadrant for FH4 — this
# represents "your car's screen area" in road-only mode.
hud_service.save_mask("FH4", [{"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5}], None)

captured = {"yolo_in": None, "seg_in": None}

def _fake_yolo(img):
    captured["yolo_in"] = img.copy()
    return {"boxes": [], "min_confidence": 1.0}

def _fake_seg(img):
    captured["seg_in"] = img.copy()
    return {"mask_png_b64": "STUB", "mean_entropy": 0.0}

# Patch the inner runners.
prelabeler._run_yolo = _fake_yolo  # type: ignore
prelabeler._run_seg  = _fake_seg   # type: ignore

img = np.full((100, 200, 3), 255, dtype=np.uint8)
out = prelabeler.prelabel_frame(img, "FH4")

# Both pixels should be preserved — input frame is NOT masked anymore.
check("road-only: top-left preserved",
      captured["yolo_in"][10, 10, 0] == 255 and captured["seg_in"][10, 10, 0] == 255)
check("road-only: bottom-right preserved for YOLO (no input masking)",
      captured["yolo_in"][90, 190, 0] == 255,
      f"got {captured['yolo_in'][90, 190, 0]}")
check("road-only: bottom-right preserved for SegFormer (no input masking)",
      captured["seg_in"][90, 190, 0] == 255,
      f"got {captured['seg_in'][90, 190, 0]}")

# Original image obviously unmodified — we never copy and zero.
check("road-only: original image unmodified", img[90, 190, 0] == 255)


# ═══════════════════════════════════════════════════════════════════════════
# 3b. YOLO boxes inside the HUD region get filtered out.
# ═══════════════════════════════════════════════════════════════════════════
def _fake_yolo_with_boxes(img):
    # Two boxes: one inside HUD region (bottom-right), one outside (top-left).
    return {
        "boxes": [
            # Inside the masked region — should be filtered out as "your car".
            {"cls": "vehicle", "x": 0.75, "y": 0.75, "w": 0.20, "h": 0.20, "confidence": 0.9},
            # Outside the masked region — should be kept as real traffic.
            {"cls": "vehicle", "x": 0.25, "y": 0.25, "w": 0.10, "h": 0.10, "confidence": 0.8},
        ],
        "min_confidence": 0.8,
    }

prelabeler._run_yolo = _fake_yolo_with_boxes  # type: ignore
out_filt = prelabeler.prelabel_frame(img, "FH4")
kept = out_filt["det"]["boxes"]
check("box-filter: kept exactly 1 box", len(kept) == 1, f"got {len(kept)}")
check("box-filter: kept the OUTSIDE box (real traffic)",
      len(kept) == 1 and abs(kept[0]["x"] - 0.25) < 0.01)
check("box-filter: dropped the INSIDE box (your car)",
      all(abs(b["x"] - 0.75) > 0.01 for b in kept))


# Restore the no-box stub for subsequent tests.
prelabeler._run_yolo = _fake_yolo  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# 4. Output shape contract
# ═══════════════════════════════════════════════════════════════════════════
check("output has 'seg' key", "seg" in out)
check("output has 'det' key", "det" in out)
check("seg has 'mask_png_b64' key", "mask_png_b64" in out["seg"])
check("seg has 'mean_entropy' key", "mean_entropy" in out["seg"])
check("det has 'boxes' key",        "boxes" in out["det"])
check("det has 'min_confidence' key", "min_confidence" in out["det"])


# ═══════════════════════════════════════════════════════════════════════════
# 5. No game_version → no mask applied (passthrough)
# ═══════════════════════════════════════════════════════════════════════════
captured["yolo_in"] = None
captured["seg_in"]  = None
img2 = np.full((50, 50, 3), 200, dtype=np.uint8)
prelabeler.prelabel_frame(img2, None)
check("no game_version: bottom-right preserved (no mask)",
      captured["yolo_in"][40, 40, 0] == 200)


# ═══════════════════════════════════════════════════════════════════════════
# 6. COCO→ours mapping
# ═══════════════════════════════════════════════════════════════════════════
m = prelabeler._COCO_TO_OURS
check("COCO 2 (car) → vehicle",        m.get(2) == "vehicle")
check("COCO 3 (motorcycle) → vehicle", m.get(3) == "vehicle")
check("COCO 5 (bus) → vehicle",        m.get(5) == "vehicle")
check("COCO 7 (truck) → vehicle",      m.get(7) == "vehicle")
check("COCO 11 (stop sign) → sign",    m.get(11) == "sign")
check("COCO 0 (person) is dropped",    m.get(0) is None)
check("COCO 16 (dog) is dropped",      m.get(16) is None)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Cityscapes→ours mapping
# ═══════════════════════════════════════════════════════════════════════════
m = prelabeler._CITYSCAPES_TO_OURS
check("Cityscapes 0 (road) → 1 road",      m.get(0) == 1)
check("Cityscapes 1 (sidewalk) → 2 curb",  m.get(1) == 2)
check("Cityscapes 2 (building) → 3 wall",  m.get(2) == 3)
check("Cityscapes 3 (wall) → 3 wall",      m.get(3) == 3)
check("Cityscapes 4 (fence) → 3 wall",     m.get(4) == 3)
check("Cityscapes 8 (vegetation) → 0 offroad", m.get(8) == 0)
check("Cityscapes 9 (terrain) → 0 offroad",    m.get(9) == 0)
check("Cityscapes 10 (sky) defaults to offroad (not in map)",
      m.get(10) is None)


# ─── Done ──────────────────────────────────────────────────────────────────
print()
if _FAILED == 0:
    print(f"All {_PASSED} Module 4 prelabeler tests passed.")
else:
    print(f"FAILED: {_FAILED} of {_PASSED + _FAILED} tests")
    sys.exit(1)