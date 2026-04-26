"""
ForzaTek AI v2 — Project Paths
==============================
Every path used by every module is computed here, once, from this file's
location. No module hardcodes a path; they all import from `backend.core.paths`.

This means moving the project folder doesn't break anything, and tests can
swap out paths cleanly by patching this module.

Layout:
    forzatek/
    ├── backend/
    │   ├── core/
    │   │   └── paths.py      ← THIS FILE
    │   └── ...
    ├── frontend/
    └── data/
        ├── forzatek.db
        ├── settings.json
        └── models/
"""
from __future__ import annotations

from pathlib import Path

# ────────── Project root ──────────
# This file lives at <root>/backend/core/paths.py, so the project root is
# three parents up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ────────── Top-level directories ──────────
BACKEND_DIR:  Path = PROJECT_ROOT / "backend"
FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
DATA_DIR:     Path = PROJECT_ROOT / "data"

# ────────── Subdirectories of data/ ──────────
MODELS_DIR:      Path = DATA_DIR / "models"           # perception .pt / .onnx checkpoints
PPO_DIR:         Path = DATA_DIR / "ppo"              # PPO checkpoints
VIDEOS_DIR:      Path = DATA_DIR / "videos"           # ingested local / downloaded videos
LOGS_DIR:        Path = DATA_DIR / "logs"             # training + runtime logs

# ────────── Specific files ──────────
DB_PATH:         Path = DATA_DIR / "forzatek.db"
SETTINGS_PATH:   Path = DATA_DIR / "settings.json"
SCHEMA_PATH:     Path = Path(__file__).parent / "schema.sql"


def ensure_dirs() -> None:
    """Create every project directory if it doesn't already exist.

    Called once at app startup (from `main.py`) and again at the top of
    `init_db()` so tests that point at a temp DB still work.
    """
    for d in (DATA_DIR, MODELS_DIR, PPO_DIR, VIDEOS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "BACKEND_DIR",
    "FRONTEND_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "PPO_DIR",
    "VIDEOS_DIR",
    "LOGS_DIR",
    "DB_PATH",
    "SETTINGS_PATH",
    "SCHEMA_PATH",
    "ensure_dirs",
]