"""
Module 4 — Labeling service tests
=================================
Pure DB tests. No YOLO, no SegFormer, no torch. Verifies the next-frame
picker, accept/edit/manual/unlabel/skip mutators, queue management, and
proposal write/read paths.

Run from project root:
    python -m tests.test_labeling_service
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# ─── Path bootstrap ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Point Module 0 at a temp DB BEFORE importing anything that opens a connection.
# Belt-and-suspenders: env var + monkey-patch paths.DB_PATH + monkey-patch
# database.DB_PATH (different paths.py revisions resolve at different times).
_TMPDIR = Path(tempfile.mkdtemp(prefix="ftk_m4_svc_"))
_TMP_DB = _TMPDIR / "forzatek.db"
os.environ["FORZATEK_DATA_DIR"] = str(_TMPDIR)

from backend.core import database, paths  # noqa: E402

paths.DB_PATH = _TMP_DB
if hasattr(paths, "DATA_DIR"):
    paths.DATA_DIR = _TMPDIR
if hasattr(database, "DB_PATH"):
    database.DB_PATH = _TMP_DB

# Hard guardrail — refuse to run if we'd write to anything but the temp dir.
_resolved = Path(getattr(database, "DB_PATH", paths.DB_PATH)).resolve()
if str(_TMPDIR) not in str(_resolved):
    print(f"REFUSING TO RUN: test would write to {_resolved}")
    print(f"  expected path under: {_TMPDIR}")
    sys.exit(2)

database.init_db()

from backend.labeling import service  # noqa: E402


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
    """Wipe the test DB clean. We DELETE rows rather than unlink the file
    because Windows may hold the file open via a cached connection.
    Also resets sqlite_sequence so AUTOINCREMENT IDs restart from 1.
    """
    database.init_db()  # ensure tables exist
    with database.write_conn() as conn:
        for tbl in ("labels", "proposals", "active_queue", "hud_masks", "frames", "sources"):
            try:
                conn.execute(f"DELETE FROM {tbl}")
            except Exception:
                pass  # table may not exist in this stub schema
        # Reset autoincrement counters so frame IDs start at 1 again.
        try:
            conn.execute("DELETE FROM sqlite_sequence")
        except Exception:
            pass


def insert_frame(
    game_version: str = "FH4",
    label_status: str = "unlabeled",
    biome: str = None,
    weather: str = None,
    time_of_day: str = None,
) -> int:
    """Insert a minimal frame row, return its id."""
    now = time.time()
    fake_jpeg = b"\xFF\xD8\xFF\xE0" + b"\x00" * 60 + b"\xFF\xD9"
    # phash must be INTEGER per real schema. Use a counter + random low bits
    # so we never collide and never overflow signed int64.
    import random
    ph = random.randint(1, 2**62)
    with database.write_conn() as c:
        cur = c.execute(
            """INSERT INTO frames
               (ts, source_type, game_version, phash, frame_jpeg,
                width, height, label_status, biome, weather, time_of_day)
               VALUES (?, 'live', ?, ?, ?, 200, 100, ?, ?, ?, ?)""",
            (now, game_version, ph, fake_jpeg,
             label_status, biome, weather, time_of_day),
        )
        return cur.lastrowid


def insert_proposal(frame_id: int, task: str, payload: dict,
                    confidence: float = 0.9, uncertainty: float = 0.1) -> None:
    service.write_proposal(frame_id, task, payload, confidence, uncertainty)


# ═══════════════════════════════════════════════════════════════════════════
# 1. next_frame — empty DB
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
check("next_frame returns None on empty DB", service.next_frame() is None)


# ═══════════════════════════════════════════════════════════════════════════
# 2. next_frame — picks unlabeled in id order
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
a = insert_frame()
b = insert_frame()
res = service.next_frame()
check("next_frame returns oldest unlabeled first", res is not None and res["frame_id"] == a)
check("next_frame.from_queue is False for unlabeled", res["from_queue"] is False)


# ═══════════════════════════════════════════════════════════════════════════
# 3. next_frame — queue beats unlabeled
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
a = insert_frame()
b = insert_frame()
service.enqueue_uncertain(b, uncertainty=0.7)
res = service.next_frame()
check("queued frame served before unlabeled", res["frame_id"] == b)
check("from_queue is True", res["from_queue"] is True)
check("uncertainty surfaced", res["uncertainty"] == 0.7)


# ═══════════════════════════════════════════════════════════════════════════
# 4. next_frame — queue ordered by uncertainty DESC
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
a = insert_frame()
b = insert_frame()
c_ = insert_frame()
service.enqueue_uncertain(a, uncertainty=0.3)
service.enqueue_uncertain(b, uncertainty=0.9)
service.enqueue_uncertain(c_, uncertainty=0.5)
res = service.next_frame()
check("highest uncertainty served first", res["frame_id"] == b)


# ═══════════════════════════════════════════════════════════════════════════
# 5. next_frame — stratification filters
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
a = insert_frame(game_version="FH4")
b = insert_frame(game_version="FH5")
c_ = insert_frame(game_version="FH5", biome="desert")
res = service.next_frame(game_version="FH5")
check("game_version filter restricts results", res["frame_id"] in (b, c_))
res2 = service.next_frame(game_version="FH5", biome="desert")
check("game_version + biome filter both apply", res2["frame_id"] == c_)
res3 = service.next_frame(game_version="FH99")
check("filter returns None when nothing matches", res3 is None)


# ═══════════════════════════════════════════════════════════════════════════
# 6. write_proposal — round trip
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
seg_payload = {"mask_png_b64": "abc", "classes": ["offroad", "road", "curb", "wall"]}
det_payload = {"boxes": [{"cls": "vehicle", "x": 0.5, "y": 0.5,
                          "w": 0.1, "h": 0.1, "confidence": 0.8}]}
service.write_proposal(fid, "seg", seg_payload, 0.92, 0.08)
service.write_proposal(fid, "det", det_payload, 0.80, 0.20)
props = service.get_proposals(fid)
check("get_proposals reports has_proposals", props["has_proposals"] is True)
check("seg proposal round-trips", props["seg"]["mask_png_b64"] == "abc")
check("det proposal round-trips",
      props["det"]["boxes"][0]["cls"] == "vehicle",
      f"got {props['det']}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. write_proposal — invalid task rejected
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
try:
    service.write_proposal(fid, "BOGUS", {}, 1.0, 0.0)
    check("invalid task raises ValueError", False, "no error raised")
except ValueError:
    check("invalid task raises ValueError", True)


# ═══════════════════════════════════════════════════════════════════════════
# 8. accept_proposal — copies proposals into labels
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
service.write_proposal(fid, "seg", {"mask_png_b64": "x"}, 0.9, 0.1)
service.write_proposal(fid, "det", {"boxes": []}, 0.95, 0.05)
ok = service.accept_proposal(fid)
check("accept_proposal returns True", ok)
with database.read_conn() as conn:
    rows = conn.execute(
        "SELECT task, provenance FROM labels WHERE frame_id = ?", (fid,)
    ).fetchall()
labels_by_task = {r["task"]: r["provenance"] for r in rows}
check("accept created seg label", labels_by_task.get("seg") == "human_accepted")
check("accept created det label", labels_by_task.get("det") == "human_accepted")
with database.read_conn() as conn:
    status = conn.execute(
        "SELECT label_status FROM frames WHERE id = ?", (fid,)
    ).fetchone()["label_status"]
check("accept set label_status to 'labeled'", status == "labeled")


# ═══════════════════════════════════════════════════════════════════════════
# 9. accept_proposal — fails with no proposals
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
ok = service.accept_proposal(fid)
check("accept with no proposals returns False", ok is False)


# ═══════════════════════════════════════════════════════════════════════════
# 10. submit_edit — saves edited data with right provenance
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
ok = service.submit_edit(fid, "EDITED_B64",
                         [{"cls": "vehicle", "x": 0.1, "y": 0.1,
                           "w": 0.1, "h": 0.1, "confidence": 1.0}])
check("submit_edit returns True", ok)
with database.read_conn() as conn:
    seg = conn.execute(
        "SELECT data_json, provenance FROM labels WHERE frame_id=? AND task='seg'",
        (fid,),
    ).fetchone()
check("submit_edit provenance is human_edited", seg["provenance"] == "human_edited")
check("submit_edit saved the b64", json.loads(seg["data_json"])["mask_png_b64"] == "EDITED_B64")


# ═══════════════════════════════════════════════════════════════════════════
# 11. submit_manual — provenance is manual_from_scratch
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
ok = service.submit_manual(fid, "MANUAL_B64", [])
check("submit_manual returns True", ok)
with database.read_conn() as conn:
    seg = conn.execute(
        "SELECT provenance FROM labels WHERE frame_id=? AND task='seg'", (fid,),
    ).fetchone()
check("submit_manual provenance correct", seg["provenance"] == "manual_from_scratch")


# ═══════════════════════════════════════════════════════════════════════════
# 12. auto_trust_proposal — provenance is auto_trusted
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
service.write_proposal(fid, "seg", {"mask_png_b64": "z"}, 0.95, 0.03)
service.write_proposal(fid, "det", {"boxes": []}, 0.99, 0.01)
ok = service.auto_trust_proposal(fid)
check("auto_trust_proposal returns True", ok)
with database.read_conn() as conn:
    provs = [r["provenance"] for r in conn.execute(
        "SELECT provenance FROM labels WHERE frame_id=?", (fid,)).fetchall()]
check("auto_trust marks both labels auto_trusted", all(p == "auto_trusted" for p in provs))


# ═══════════════════════════════════════════════════════════════════════════
# 13. unlabel_frame — removes labels and resets status
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
service.submit_manual(fid, "X", [])
service.unlabel_frame(fid)
with database.read_conn() as conn:
    n = conn.execute("SELECT COUNT(*) FROM labels WHERE frame_id=?", (fid,)).fetchone()[0]
    s = conn.execute("SELECT label_status FROM frames WHERE id=?", (fid,)).fetchone()["label_status"]
check("unlabel removes all labels", n == 0)
check("unlabel resets status", s == "unlabeled")


# ═══════════════════════════════════════════════════════════════════════════
# 14. unlabel_frame — also clears active_queue
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
service.enqueue_uncertain(fid, 0.5)
service.unlabel_frame(fid)
with database.read_conn() as conn:
    n = conn.execute("SELECT COUNT(*) FROM active_queue WHERE frame_id=?", (fid,)).fetchone()[0]
check("unlabel removes from queue", n == 0)


# ═══════════════════════════════════════════════════════════════════════════
# 15. skip_frame — sets status=skipped
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
service.skip_frame(fid)
with database.read_conn() as conn:
    s = conn.execute("SELECT label_status FROM frames WHERE id=?", (fid,)).fetchone()["label_status"]
check("skip sets status=skipped", s == "skipped")


# ═══════════════════════════════════════════════════════════════════════════
# 16. enqueue_uncertain — sets status=queued
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
service.enqueue_uncertain(fid, 0.7)
with database.read_conn() as conn:
    s = conn.execute("SELECT label_status FROM frames WHERE id=?", (fid,)).fetchone()["label_status"]
    u = conn.execute("SELECT uncertainty FROM active_queue WHERE frame_id=?", (fid,)).fetchone()["uncertainty"]
check("enqueue sets status=queued", s == "queued")
check("enqueue stores uncertainty", abs(u - 0.7) < 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 17. enqueue_uncertain — upsert (no duplicate rows)
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
service.enqueue_uncertain(fid, 0.7)
service.enqueue_uncertain(fid, 0.3)
with database.read_conn() as conn:
    n = conn.execute("SELECT COUNT(*) FROM active_queue WHERE frame_id=?", (fid,)).fetchone()[0]
    u = conn.execute("SELECT uncertainty FROM active_queue WHERE frame_id=?", (fid,)).fetchone()["uncertainty"]
check("enqueue upserts (no duplicate)", n == 1)
check("enqueue overwrites uncertainty", abs(u - 0.3) < 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 18. progress — counts and provenance breakdown
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
a = insert_frame()
b = insert_frame()
c_ = insert_frame()
d = insert_frame()
service.submit_manual(a, "x", [])     # labeled (manual_from_scratch)
service.write_proposal(b, "seg", {"mask_png_b64": "y"}, 0.9, 0.1)
service.write_proposal(b, "det", {"boxes": []}, 0.9, 0.1)
service.auto_trust_proposal(b)        # labeled (auto_trusted)
service.enqueue_uncertain(c_, 0.6)    # queued
# d stays unlabeled
p = service.progress()
check("progress.total counts every frame", p["total"] == 4)
check("progress.unlabeled correct", p["unlabeled"] == 1)
check("progress.queued correct",    p["queued"] == 1)
check("progress.labeled correct",   p["labeled"] == 2)
check("progress.by_provenance.manual_from_scratch", p["by_provenance"].get("manual_from_scratch") == 1)
check("progress.by_provenance.auto_trusted",        p["by_provenance"].get("auto_trusted") == 1)


# ═══════════════════════════════════════════════════════════════════════════
# 19. list_proposals_summary — returns queued frames sorted by uncertainty
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
a = insert_frame()
b = insert_frame()
c_ = insert_frame()
service.enqueue_uncertain(a, 0.2)
service.enqueue_uncertain(b, 0.9)
service.enqueue_uncertain(c_, 0.5)
summary = service.list_proposals_summary(limit=10)
check("queue summary returns 3 items", len(summary) == 3)
check("queue summary sorted DESC by uncertainty", summary[0]["frame_id"] == b)
check("queue summary 2nd item", summary[1]["frame_id"] == c_)


# ═══════════════════════════════════════════════════════════════════════════
# 20. list_unlabeled_ids — only returns unlabeled
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
a = insert_frame(label_status="unlabeled")
b = insert_frame(label_status="labeled")
c_ = insert_frame(label_status="unlabeled")
ids = service.list_unlabeled_ids(limit=100)
check("list_unlabeled_ids correct count", len(ids) == 2)
check("list_unlabeled_ids excludes labeled", b not in ids)


# ═══════════════════════════════════════════════════════════════════════════
# 21. get_frame_for_inference — returns bytes + game_version
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame(game_version="FH5")
fr = service.get_frame_for_inference(fid)
check("get_frame_for_inference returns dict",  fr is not None)
check("returns jpeg bytes",                    isinstance(fr["jpeg_bytes"], bytes))
check("returns game_version",                  fr["game_version"] == "FH5")


# ═══════════════════════════════════════════════════════════════════════════
# 22. get_proposals — returns most recent when multiple exist
# ═══════════════════════════════════════════════════════════════════════════
reset_db()
fid = insert_frame()
service.write_proposal(fid, "seg", {"mask_png_b64": "OLD"}, 0.5, 0.5)
service.write_proposal(fid, "seg", {"mask_png_b64": "NEW"}, 0.9, 0.1)
props = service.get_proposals(fid)
check("get_proposals returns latest", props["seg"]["mask_png_b64"] == "NEW")


# ─── Done ──────────────────────────────────────────────────────────────────
print()
if _FAILED == 0:
    print(f"All {_PASSED} Module 4 service tests passed.")
else:
    print(f"FAILED: {_FAILED} of {_PASSED + _FAILED} tests")
    sys.exit(1)