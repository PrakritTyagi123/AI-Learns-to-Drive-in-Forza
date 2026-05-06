"""ForzaTek AI v2 — Module 5: Perception (training).

Trains the custom multi-task model on labels written by Module 4.

Public surface (cross-module imports allowed):
    perception.classes  — INPUT_W, INPUT_H, NUM_SEG_CLASSES, NUM_DET_CLASSES,
                          SEG_CLASSES, DET_CLASSES, palette, IGNORE_INDEX
    perception.runtime  — fast inference wrapper (loads checkpoint, fp16)

Internal-only (do not import from sibling modules):
    perception.model     — nn.Module definition
    perception.dataset   — DB-backed Dataset
    perception.train     — training loop
    perception.predict   — active-learning batch inference
    perception.metrics   — IoU, mAP, confusion matrix
    perception.runner    — subprocess manager for the UI

The Eel + FastAPI surface is in `eel_api.py` and `routes.py`, registered
once from backend/main.py.
"""