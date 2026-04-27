"""
Module 3 — HUD Mask Tests
=========================
Run with:
    python -m tests.test_hud_mask

Style matches Modules 0/1/2: plain script, no pytest required, prints ✓ for
every passing check. Exits 1 on any failure.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Make `backend` importable when run from project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import cv2

# Point Module 0 at a temp DB BEFORE importing anything that opens a connection.
# We do this three ways belt-and-suspenders, because different versions of
# paths.py resolve DB_PATH at different times:
#   1. env var (read by paths.py if it supports it)
#   2. monkey-patch paths.DB_PATH after import (catches import-time resolution)
#   3. monkey-patch database.DB_PATH if the database module cached its own copy
_TMPDIR = tempfile.mkdtemp(prefix="ftk_m3_test_")
_TMP_DB = Path(_TMPDIR) / "forzatek.db"
os.environ["FORZATEK_DATA_DIR"] = _TMPDIR

from backend.core import database, paths  # noqa: E402

# Force the temp path regardless of how paths.py was written.
paths.DB_PATH = _TMP_DB
if hasattr(paths, "DATA_DIR"):
    paths.DATA_DIR = Path(_TMPDIR)
if hasattr(database, "DB_PATH"):
    database.DB_PATH = _TMP_DB

# Hard guardrail: if for any reason DB_PATH still points at a real-looking
# location, refuse to run rather than pollute production data.
_resolved = Path(getattr(database, "DB_PATH", paths.DB_PATH)).resolve()
if "forzatek_v2" in str(_resolved) or _TMPDIR not in str(_resolved):
    if _TMPDIR not in str(_resolved):
        print(f"REFUSING TO RUN: test would write to {_resolved}")
        print(f"  expected path under: {_TMPDIR}")
        sys.exit(2)

database.init_db()

from backend.hud_mask import service, auto_propagate  # noqa: E402


_FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _FAILED
    if condition:
        print(f"✓ {label}")
    else:
        _FAILED += 1
        print(f"✗ {label}" + (f"  ({detail})" if detail else ""))


def reset_db() -> None:
    """Wipe both tables between tests so nothing leaks state."""
    with database.write_conn() as conn:
        conn.execute("DELETE FROM hud_masks")
        conn.execute("DELETE FROM frames")
    auto_propagate.invalidate_cache()


def make_jpeg(h: int = 100, w: int = 200, color: tuple = (0, 200, 0)) -> bytes:
    img = np.full((h, w, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return buf.tobytes()


def insert_frame(game_version: str = "FH4", w: int = 200, h: int = 100) -> int:
    """Insert a synthetic frame and return its id."""
    with database.write_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO frames (
                ts, source_type, game_version, phash, frame_jpeg, width, height
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (time.time(), "test", game_version, 0, make_jpeg(h, w), w, h),
        )
        return cur.lastrowid


# ─── 1. Validation ─────────────────────────────────────────────────────────
reset_db()
try:
    service.save_mask("", [], None)
    check("empty game_version is rejected", False, "no error raised")
except ValueError:
    check("empty game_version is rejected", True)

try:
    service.save_mask("FH4", "not a list", None)  # type: ignore[arg-type]
    check("non-list rects is rejected", False)
except ValueError:
    check("non-list rects is rejected", True)

try:
    service.save_mask("FH4", [{"x": 0.1, "y": 0.1}], None)  # missing w,h
    check("rect missing keys is rejected", False)
except ValueError:
    check("rect missing keys is rejected", True)

# Empty/zero-area rects are silently dropped, not rejected.
saved = service.save_mask("FH4", [{"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.0}], None)
check(
    "zero-area rects are silently dropped",
    saved["rects"] == [],
    f"got {saved['rects']!r}",
)


# ─── 2. Save / load round-trip ─────────────────────────────────────────────
reset_db()
fid = insert_frame("FH4")
rects = [
    {"x": 0.02, "y": 0.78, "w": 0.18, "h": 0.20},
    {"x": 0.40, "y": 0.86, "w": 0.20, "h": 0.10},
]
saved = service.save_mask("FH4", rects, fid)
check("save_mask returns the saved row", saved["game_version"] == "FH4")
check("save_mask preserves rect count", len(saved["rects"]) == 2)
check("save_mask records sample_frame", saved["sample_frame"] == fid)

loaded = service.get_mask("FH4")
check("get_mask returns saved row", loaded is not None)
check(
    "round-tripped rects match",
    loaded["rects"] == rects,
    f"expected {rects}, got {loaded['rects']}",
)


# ─── 3. UPSERT semantics ───────────────────────────────────────────────────
new_rects = [{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]
service.save_mask("FH4", new_rects, fid)
loaded = service.get_mask("FH4")
check("re-saving replaces (does not append)", loaded["rects"] == new_rects)

# Different game_version coexists.
service.save_mask("FH5", [{"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}], None)
all_masks = service.list_masks()
check("two distinct versions coexist", len(all_masks) == 2)


# ─── 4. Clamping ───────────────────────────────────────────────────────────
reset_db()
out_of_bounds = [
    {"x": 0.9, "y": 0.9, "w": 0.5, "h": 0.5},   # extends past 1.0
    {"x": -0.2, "y": -0.1, "w": 0.5, "h": 0.5}, # negative origin
]
saved = service.save_mask("FH4", out_of_bounds, None)
r0, r1 = saved["rects"]
check("rect clamped to right edge",  abs((r0["x"] + r0["w"]) - 1.0) < 1e-9)
check("rect clamped to bottom edge", abs((r0["y"] + r0["h"]) - 1.0) < 1e-9)
check("negative origin clamped to 0", r1["x"] == 0.0 and r1["y"] == 0.0)


# ─── 5. delete_mask ────────────────────────────────────────────────────────
check("delete returns True for existing", service.delete_mask("FH4") is True)
check("delete returns False if absent",   service.delete_mask("FH4") is False)
check("get_mask returns None after delete", service.get_mask("FH4") is None)


# ─── 6. auto_propagate.apply_mask ──────────────────────────────────────────
reset_db()
# A mask that covers the bottom-right quarter.
service.save_mask("FH4", [{"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5}], None)

img = np.full((100, 200, 3), 255, dtype=np.uint8)  # white image
masked = auto_propagate.apply_mask(img, "FH4")

# Top-left should be untouched.
check(
    "apply_mask preserves non-HUD region",
    masked[10, 10, 0] == 255,
    f"got {masked[10, 10, 0]}",
)
# Bottom-right should be zeroed.
check(
    "apply_mask zeros HUD region",
    masked[90, 190, 0] == 0,
    f"got {masked[90, 190, 0]}",
)
# Original is unmodified.
check("apply_mask does not mutate input", img[90, 190, 0] == 255)

# No mask configured → returns a copy unchanged.
unmasked = auto_propagate.apply_mask(img, "FH99_unknown")
check(
    "apply_mask passes through when no mask exists",
    np.array_equal(unmasked, img),
)
check("returned image is a copy, not the same object", unmasked is not img)


# ─── 7. Resolution independence ────────────────────────────────────────────
reset_db()
# Same mask coordinates, two different resolutions.
service.save_mask("FH4", [{"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5}], None)

m_small = auto_propagate.get_mask_array("FH4", 100, 200)
m_large = auto_propagate.get_mask_array("FH4", 400, 800)

# In both, the top-left quadrant should be zero (HUD), the rest one.
def check_quadrant(mask: np.ndarray, label: str) -> None:
    h, w = mask.shape
    tl_zero  = mask[: h // 2,  : w // 2].sum()
    rest_one = mask[h // 2:, w // 2:].sum()
    check(f"{label}: top-left masked",     tl_zero == 0)
    check(f"{label}: bottom-right kept",   rest_one == (h // 2) * (w // 2))

check_quadrant(m_small, "small res")
check_quadrant(m_large, "large res")


# ─── 8. Cache invalidation ─────────────────────────────────────────────────
reset_db()
service.save_mask("FH4", [{"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5}], None)
m1 = auto_propagate.get_mask_array("FH4", 100, 200)
check("first get_mask_array returns array", m1 is not None)

# Re-save with a different mask. Cache should be busted automatically.
service.save_mask("FH4", [{"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5}], None)
m2 = auto_propagate.get_mask_array("FH4", 100, 200)
check(
    "save_mask invalidates auto_propagate cache",
    not np.array_equal(m1, m2),
)


# ─── 9. Sample frame picker ────────────────────────────────────────────────
reset_db()
check(
    "get_sample_frame returns None for empty DB",
    service.get_sample_frame("FH4") is None,
)
fid = insert_frame("FH4")
sample = service.get_sample_frame("FH4")
check("get_sample_frame returns a frame after insert", sample is not None)
check("sample contains JPEG bytes", isinstance(sample["jpeg_bytes"], bytes))
check("sample reports correct dimensions", sample["width"] == 200 and sample["height"] == 100)


# ─── 10. list_known_game_versions ──────────────────────────────────────────
reset_db()
insert_frame("FH4"); insert_frame("FH4"); insert_frame("FH5")
versions = service.list_known_game_versions()
check(
    "list_known_game_versions returns distinct versions",
    versions == ["FH4", "FH5"],
    f"got {versions}",
)


# ─── Done ──────────────────────────────────────────────────────────────────
print()
if _FAILED == 0:
    print("All Module 3 HUD mask tests passed.")
    sys.exit(0)
else:
    print(f"{_FAILED} test(s) failed.")
    sys.exit(1)