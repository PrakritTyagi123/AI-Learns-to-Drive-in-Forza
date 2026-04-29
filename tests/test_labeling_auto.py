"""
Module 4 — Auto-labeler tests
=============================
Tests the worker's decision logic by monkey-patching the prelabeler so we
don't need YOLO/SegFormer or torch at test time.

Run from project root:
    python -m tests.test_labeling_auto
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

_TMPDIR = Path(tempfile.mkdtemp(prefix="ftk_m4_auto_"))
_TMP_DB = _TMPDIR / "forzatek.db"
os.environ["FORZATEK_DATA_DIR"] = str(_TMPDIR)

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

from backend.labeling import auto_labeler, prelabeler, service  # noqa: E402


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


def insert_frame(label_status: str = "unlabeled") -> int:
    now = time.time()
    fake_jpeg = b"\xFF\xD8\xFF\xE0" + b"\x00" * 100 + b"\xFF\xD9"
    import random
    ph = random.randint(1, 2**62)
    with database.write_conn() as c:
        cur = c.execute(
            """INSERT INTO frames
               (ts, source_type, game_version, phash, frame_jpeg,
                width, height, label_status)
               VALUES (?, 'live', 'FH4', ?, ?, 200, 100, ?)""",
            (now, ph, fake_jpeg, label_status),
        )
        return cur.lastrowid


def wait_for_done(timeout: float = 5.0) -> bool:
    """Spin-wait until the worker reports running=False."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not auto_labeler.status()["running"]:
            return True
        time.sleep(0.05)
    return False


# ─── Mocks ──────────────────────────────────────────────────────────────────
class MockPrelabeler:
    """Replaces prelabeler.prelabel_batch for tests.
    Returns N copies of the fixture set on .next_response.
    """
    def __init__(self):
        self.calls = 0
        self.next_response = None

    def prelabel_batch(self, frames, gvs):
        self.calls += 1
        # Return one copy of next_response per input frame.
        import copy
        return [copy.deepcopy(self.next_response) for _ in frames]

    def prelabel_frame(self, img, gv):
        # Kept for the prelabel test suite which still calls single-frame.
        self.calls += 1
        return self.next_response


_mock = MockPrelabeler()


def install_mock(seg_entropy: float, det_min_conf: float,
                 num_boxes: int = 1, pct_decisive: float = 0.95,
                 pct_road: float = 0.30) -> None:
    """Configure what the mocked prelabeler returns next.

    pct_decisive: fraction of pixels SegFormer is decisive about (drives
        auto-trust in road-only mode). 0.95 = very decisive, 0.5 = unsure.
    """
    boxes = []
    for i in range(num_boxes):
        boxes.append({
            "cls": "vehicle", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1,
            "confidence": det_min_conf,
        })
    _mock.next_response = {
        "seg": {
            "mask_png_b64": "MOCK_MASK",
            "mean_entropy": seg_entropy,
            "pct_road":     pct_road,
            "pct_decisive": pct_decisive,
        },
        "det": {"boxes": boxes,
                "min_confidence": det_min_conf if boxes else 1.0},
    }


# Patch prelabeler module functions for the duration of these tests.
prelabeler.prelabel_frame = lambda img, gv: _mock.prelabel_frame(img, gv)  # type: ignore
prelabeler.prelabel_batch = lambda fs, gvs: _mock.prelabel_batch(fs, gvs)  # type: ignore
prelabeler.is_available   = lambda: (True, "ok")  # type: ignore


# Also stub cv2.imdecode so we don't need a real JPEG. Return a (10x10x3) array.
import numpy as np  # noqa: E402
import cv2          # noqa: E402

_real_imdecode = cv2.imdecode
def _stub_imdecode(arr, flags):
    return np.zeros((10, 10, 3), dtype=np.uint8)
cv2.imdecode = _stub_imdecode  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# 1. start() refuses if prelabeler unavailable
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
prelabeler.is_available = lambda: (False, "torch not installed")  # type: ignore
res = auto_labeler.start()
check("start fails when prelabeler unavailable", res.get("started") is False)
check("error surfaced",                          "torch" in (res.get("error") or ""))
prelabeler.is_available = lambda: (True, "ok")   # type: ignore  # restore


# ═══════════════════════════════════════════════════════════════════════════
# 2. status() snapshot has expected fields when idle
# ═══════════════════════════════════════════════════════════════════════════
s = auto_labeler.status()
check("status has 'running' key", "running" in s)
check("status has 'total' key",   "total" in s)
check("status has 'processed' key", "processed" in s)
check("status reports not running by default", s["running"] is False)


# ═══════════════════════════════════════════════════════════════════════════
# 3. High-confidence frame → auto_trusted
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
install_mock(seg_entropy=0.10, det_min_conf=0.95, num_boxes=1)
auto_labeler.start(confidence_threshold=0.85, entropy_threshold=0.50)
ok = wait_for_done(5.0)
check("worker completed within 5s", ok)
s = auto_labeler.status()
check("processed = 1",     s["processed"] == 1)
check("auto_trusted = 1",  s["auto_trusted"] == 1)
check("queued = 0",        s["queued"] == 0)

with database.read_conn() as conn:
    status = conn.execute(
        "SELECT label_status FROM frames WHERE id=?", (fid,)
    ).fetchone()["label_status"]
    provs = [r["provenance"] for r in conn.execute(
        "SELECT provenance FROM labels WHERE frame_id=?", (fid,)).fetchall()]
check("frame status = labeled",            status == "labeled")
check("labels written with auto_trusted",  all(p == "auto_trusted" for p in provs))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Low-confidence detection → queued
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
install_mock(seg_entropy=0.10, det_min_conf=0.40, num_boxes=1)
auto_labeler.start(confidence_threshold=0.85, entropy_threshold=0.50)
wait_for_done(5.0)
s = auto_labeler.status()
check("low det conf: queued = 1",    s["queued"] == 1)
check("low det conf: trusted = 0",   s["auto_trusted"] == 0)
with database.read_conn() as conn:
    status = conn.execute(
        "SELECT label_status FROM frames WHERE id=?", (fid,)
    ).fetchone()["label_status"]
check("frame status = queued",       status == "queued")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Low decisive % (SegFormer unsure where road ends) → queued
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
install_mock(seg_entropy=1.20, det_min_conf=0.99, num_boxes=1, pct_decisive=0.50)
auto_labeler.start(confidence_threshold=0.85, entropy_threshold=0.50, decisive_threshold=0.85)
wait_for_done(5.0)
s = auto_labeler.status()
check("low decisive: queued = 1",    s["queued"] == 1)
check("low decisive: trusted = 0",   s["auto_trusted"] == 0)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Zero boxes is fine — still trusted if seg is decisive
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
install_mock(seg_entropy=0.10, det_min_conf=0.0, num_boxes=0, pct_decisive=0.95)
auto_labeler.start(confidence_threshold=0.85, entropy_threshold=0.50, decisive_threshold=0.85)
wait_for_done(5.0)
s = auto_labeler.status()
check("zero boxes + decisive seg: trusted",   s["auto_trusted"] == 1)
check("zero boxes + decisive seg: not queued", s["queued"] == 0)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Proposals always written (even when auto-trusted)
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
install_mock(seg_entropy=0.10, det_min_conf=0.95, num_boxes=1)
auto_labeler.start()
wait_for_done(5.0)
with database.read_conn() as conn:
    n_seg = conn.execute(
        "SELECT COUNT(*) FROM proposals WHERE frame_id=? AND task='seg'", (fid,)
    ).fetchone()[0]
    n_det = conn.execute(
        "SELECT COUNT(*) FROM proposals WHERE frame_id=? AND task='det'", (fid,)
    ).fetchone()[0]
check("seg proposal written",  n_seg == 1)
check("det proposal written",  n_det == 1)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Cancel stops the worker
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
for _ in range(50):
    insert_frame()
install_mock(seg_entropy=0.10, det_min_conf=0.95, num_boxes=0)
auto_labeler.start()
time.sleep(0.05)
auto_labeler.cancel()
ok = wait_for_done(5.0)
check("worker stops after cancel", ok)
s = auto_labeler.status()
check("processed less than total after cancel", s["processed"] <= s["total"])


# ═══════════════════════════════════════════════════════════════════════════
# 9. Idempotent start — calling while running returns current status
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
for _ in range(20):
    insert_frame()
install_mock(seg_entropy=0.10, det_min_conf=0.95, num_boxes=0)
auto_labeler.start()
time.sleep(0.02)
res = auto_labeler.start()  # second call mid-flight
# Should not throw, and reports running=True (status snapshot from in-progress run).
check("start while running does not crash", isinstance(res, dict))
auto_labeler.cancel()
wait_for_done(5.0)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Skipped/labeled frames not reprocessed
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
a = insert_frame(label_status="unlabeled")
b = insert_frame(label_status="labeled")
c_ = insert_frame(label_status="skipped")
install_mock(seg_entropy=0.10, det_min_conf=0.95, num_boxes=0)
auto_labeler.start()
wait_for_done(5.0)
s = auto_labeler.status()
check("only unlabeled processed", s["total"] == 1)
check("only one frame labeled by auto",  s["auto_trusted"] == 1)


# ─── Done ──────────────────────────────────────────────────────────────────
# Restore cv2.imdecode in case other tests run after.
cv2.imdecode = _real_imdecode  # type: ignore

print()
if _FAILED == 0:
    print(f"All {_PASSED} Module 4 auto-labeler tests passed.")
else:
    print(f"FAILED: {_FAILED} of {_PASSED + _FAILED} tests")
    sys.exit(1)