"""
ForzaTek AI v2 — Module 0: Core
================================
Foundation layer. Database, paths, and connections.

Every other module imports from here. This package imports from no
sibling module — it's the bottom of the dependency graph.

Public surface:
    from backend.core import (
        init_db, read_conn, write_conn,
        count_frames, count_labels,
        get_active_model, set_active_model, overall_stats,
        DB_PATH, DATA_DIR, MODELS_DIR, FRONTEND_DIR,
    )
"""
from backend.core.database import (
    SCHEMA_VERSION,
    count_frames,
    count_labels,
    get_active_model,
    init_db,
    overall_stats,
    read_conn,
    set_active_model,
    write_conn,
)
from backend.core.paths import (
    BACKEND_DIR,
    DATA_DIR,
    DB_PATH,
    FRONTEND_DIR,
    LOGS_DIR,
    MODELS_DIR,
    PPO_DIR,
    PROJECT_ROOT,
    SCHEMA_PATH,
    SETTINGS_PATH,
    VIDEOS_DIR,
    ensure_dirs,
)

__all__ = [
    # database
    "SCHEMA_VERSION",
    "init_db",
    "read_conn",
    "write_conn",
    "count_frames",
    "count_labels",
    "get_active_model",
    "set_active_model",
    "overall_stats",
    # paths
    "PROJECT_ROOT",
    "BACKEND_DIR",
    "FRONTEND_DIR",
    "DATA_DIR",
    "DB_PATH",
    "MODELS_DIR",
    "PPO_DIR",
    "VIDEOS_DIR",
    "LOGS_DIR",
    "SETTINGS_PATH",
    "SCHEMA_PATH",
    "ensure_dirs",
]