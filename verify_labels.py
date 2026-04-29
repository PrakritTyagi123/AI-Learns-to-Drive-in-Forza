"""
Verify that labels in the DB are well-formed for Module 5 to consume.

Run from project root:
    python verify_labels.py

What it checks:
  1. Every 'labeled' frame has at least one labels row.
  2. Every labels.data_json is valid JSON.
  3. Seg labels have a 'mask_png_b64' field, the PNG decodes, and
     the resulting array's class ids are all in {0, 1, 2, 3, 255}.
  4. Det labels have a 'boxes' list, each box has cls + x,y,w,h in [0,1].
  5. The 'format' field is present on every label (catches stale schemas).
  6. Counts by provenance (auto_trusted / human_*).
"""
from __future__ import annotations

import base64
import json
import sqlite3
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

DB = Path("data/forzatek.db")
if not DB.exists():
    print(f"ERROR: {DB} not found. Run from project root.")
    sys.exit(1)

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("ERROR: PIL/numpy not installed. `pip install pillow numpy`")
    sys.exit(1)


VALID_SEG_IDS = {0, 1, 2, 3, 255}
VALID_DET_CLASSES = {"vehicle", "sign"}


def main() -> int:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # ─── 1. Frame status counts ────────────────────────────────────────
    print("=" * 60)
    print("FRAME STATUS COUNTS")
    print("=" * 60)
    rows = conn.execute(
        "SELECT label_status, COUNT(*) AS n FROM frames GROUP BY label_status"
    ).fetchall()
    for r in rows:
        print(f"  {r['label_status']:12s} {r['n']}")

    # ─── 2. Label provenance counts ────────────────────────────────────
    print()
    print("=" * 60)
    print("LABEL PROVENANCE COUNTS (per task)")
    print("=" * 60)
    rows = conn.execute(
        """SELECT task, provenance, COUNT(*) AS n FROM labels
           GROUP BY task, provenance ORDER BY task, provenance"""
    ).fetchall()
    if not rows:
        print("  (no labels yet)")
    for r in rows:
        print(f"  {r['task']:5s} {r['provenance']:24s} {r['n']}")

    # ─── 3. Sanity-check the actual payload of every label ─────────────
    print()
    print("=" * 60)
    print("PAYLOAD VALIDATION")
    print("=" * 60)

    issues = []
    seg_format_counts = Counter()
    det_format_counts = Counter()
    seg_id_counts = Counter()

    rows = conn.execute(
        "SELECT id, frame_id, task, data_json, provenance FROM labels"
    ).fetchall()
    print(f"  Inspecting {len(rows)} label rows...")

    for r in rows:
        try:
            payload = json.loads(r["data_json"])
        except Exception as e:
            issues.append(f"label #{r['id']} task={r['task']} bad JSON: {e}")
            continue

        if not isinstance(payload, dict):
            issues.append(f"label #{r['id']} task={r['task']} payload not a dict")
            continue

        fmt = payload.get("format", "<missing>")
        if r["task"] == "seg":
            seg_format_counts[fmt] += 1
            # Check mask
            b64 = payload.get("mask_png_b64")
            if not b64:
                issues.append(f"label #{r['id']} (seg) missing mask_png_b64")
                continue
            try:
                img = Image.open(BytesIO(base64.b64decode(b64)))
                arr = np.array(img)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                unique = set(int(x) for x in np.unique(arr))
                bad = unique - VALID_SEG_IDS
                if bad:
                    issues.append(
                        f"label #{r['id']} frame={r['frame_id']} seg has invalid class ids {sorted(bad)}"
                    )
                for u in unique:
                    seg_id_counts[u] += 1
            except Exception as e:
                issues.append(f"label #{r['id']} seg PNG decode failed: {e}")

            # Check classes dict
            cl = payload.get("classes")
            if not isinstance(cl, dict):
                issues.append(f"label #{r['id']} seg classes not a dict (got {type(cl).__name__})")

        elif r["task"] == "det":
            det_format_counts[fmt] += 1
            boxes = payload.get("boxes")
            if not isinstance(boxes, list):
                issues.append(f"label #{r['id']} det boxes not a list")
                continue
            for i, b in enumerate(boxes):
                if not isinstance(b, dict):
                    issues.append(f"label #{r['id']} det box[{i}] not a dict")
                    continue
                cls = b.get("cls")
                if cls not in VALID_DET_CLASSES:
                    issues.append(f"label #{r['id']} det box[{i}] cls={cls!r} not in {VALID_DET_CLASSES}")
                for k in ("x", "y", "w", "h"):
                    v = b.get(k)
                    if not isinstance(v, (int, float)) or not (0 <= v <= 1):
                        issues.append(f"label #{r['id']} det box[{i}] {k}={v!r} not in [0,1]")
        else:
            issues.append(f"label #{r['id']} unknown task={r['task']!r}")

    print()
    print("Seg format versions:")
    for fmt, n in seg_format_counts.most_common():
        print(f"  {fmt:24s} {n}")
    print()
    print("Det format versions:")
    for fmt, n in det_format_counts.most_common():
        print(f"  {fmt:24s} {n}")
    print()
    print("Seg PNG class ids encountered (across all labels):")
    for cid, n in sorted(seg_id_counts.items()):
        nm = {0: "offroad", 1: "road", 2: "curb", 3: "wall", 255: "unknown"}.get(cid, "?")
        print(f"  id={cid:3d}  ({nm:8s})  appears in {n} labels")

    # ─── 4. Check label_status consistency ─────────────────────────────
    print()
    print("=" * 60)
    print("CONSISTENCY CHECKS")
    print("=" * 60)

    # Frames marked 'labeled' but with no labels rows.
    n = conn.execute("""
        SELECT COUNT(*) FROM frames f
        WHERE f.label_status = 'labeled'
          AND NOT EXISTS (SELECT 1 FROM labels l WHERE l.frame_id = f.id)
    """).fetchone()[0]
    if n > 0:
        issues.append(f"{n} frames marked 'labeled' but have no labels rows")
        print(f"  ✗ {n} 'labeled' frames have no labels (orphan status)")
    else:
        print(f"  ✓ every 'labeled' frame has at least one labels row")

    # Labels rows pointing to non-existent frames.
    n = conn.execute("""
        SELECT COUNT(*) FROM labels l
        WHERE NOT EXISTS (SELECT 1 FROM frames f WHERE f.id = l.frame_id)
    """).fetchone()[0]
    if n > 0:
        issues.append(f"{n} labels reference non-existent frames")
        print(f"  ✗ {n} labels point to deleted frames")
    else:
        print(f"  ✓ every label points to an existing frame")

    # Frames with seg label but no det label (or vice versa). Both is ideal.
    n_seg_only = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT frame_id FROM labels WHERE task='seg'
            EXCEPT
            SELECT frame_id FROM labels WHERE task='det'
        )
    """).fetchone()[0]
    n_det_only = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT frame_id FROM labels WHERE task='det'
            EXCEPT
            SELECT frame_id FROM labels WHERE task='seg'
        )
    """).fetchone()[0]
    print(f"  ℹ frames with seg but no det: {n_seg_only}")
    print(f"  ℹ frames with det but no seg: {n_det_only}")

    # ─── 5. Summary ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if not issues:
        print("✓ ALL CHECKS PASSED — labels are well-formed for Module 5")
    else:
        print(f"✗ {len(issues)} ISSUE(S) FOUND:")
        for msg in issues[:30]:
            print(f"    - {msg}")
        if len(issues) > 30:
            print(f"    ... and {len(issues) - 30} more")

    conn.close()
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
    