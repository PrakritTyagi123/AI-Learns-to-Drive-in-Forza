"""ForzaTek AI v2 — Module 4: Labeling.

The annotation tool. Reads `frames`, writes `labels`, writes `proposals`.

Pipeline:
    1. Module 2 fills `frames` with raw JPEGs (HUD intact).
    2. Module 3 stores a HUD mask per game_version.
    3. Module 4 (this module):
         a. `prelabeler.py` runs YOLOv8 + SegFormer on each frame
            AFTER `hud_mask.auto_propagate.apply_mask()` zeros out
            the HUD region.
         b. `auto_labeler.py` walks every unlabeled frame in a
            background thread. High-confidence proposals → straight
            into `labels` (provenance='auto_trusted'). Low-confidence
            ones → `active_queue` for human review.
         c. The user opens `label.html`, gets served the worst frames
            first (lowest confidence), accepts / edits / skips.

Provenance values written to the `labels` table:
    'auto_trusted'        — auto-labeler accepted (no human saw it)
    'human_accepted'      — user pressed A on a proposal as-is
    'human_edited'        — user modified the proposal then saved
    'manual_from_scratch' — user pressed R, painted from blank

NEVER do screen capture, gamepad output, or yt-dlp here. Those are
Module 2 / Module 8 territory. Module 4 is pure data work.
"""