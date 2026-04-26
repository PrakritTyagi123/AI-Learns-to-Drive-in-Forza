-- ForzaTek AI v2 — Database Schema
-- =================================
-- Single source of truth for every SQLite table in the project.
-- Every other module agrees on this contract.
--
-- Tables
-- ------
--   meta              — schema version + tiny key/value config that doesn't deserve its own table
--   frames            — raw captured frames (live Forza or ingested videos)
--   labels            — one row per (frame, task) pair, the ground truth
--   proposals         — model-generated label guesses awaiting human review
--   models            — trained perception checkpoints + their metadata
--   sources           — videos / URLs we've ingested
--   active_queue      — frames the active learner wants reviewed next
--   world_map_cells   — persistent map of where the car has driven (Module 7)
--   hud_masks         — per-game-version HUD rectangles to ignore (Module 3)
--   ppo_checkpoints   — RL training checkpoints (Module 8)

-- ─────────── Meta ───────────
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- ─────────── Frames: the raw data ───────────
CREATE TABLE IF NOT EXISTS frames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    source_id       INTEGER,                      -- FK to sources (NULL for live capture)
    source_type     TEXT NOT NULL,                -- 'live' | 'video'
    game_version    TEXT NOT NULL,                -- 'fh4' | 'fh5' | 'fh6'
    biome           TEXT,
    weather         TEXT,
    time_of_day     TEXT,
    phash           INTEGER NOT NULL,             -- perceptual hash for de-duplication
    frame_jpeg      BLOB NOT NULL,                -- 720p JPEG, quality 85
    width           INTEGER,                      -- original capture width
    height          INTEGER,                      -- original capture height
    telemetry_json  TEXT,                         -- live-only, full Forza Data Out snapshot
    video_time_sec  REAL,                         -- video-only, seconds into the source
    label_status    TEXT DEFAULT 'unlabeled'      -- unlabeled|proposed|reviewed|labeled|skipped
);
CREATE INDEX IF NOT EXISTS idx_frame_status   ON frames(label_status);
CREATE INDEX IF NOT EXISTS idx_frame_version  ON frames(game_version);
CREATE INDEX IF NOT EXISTS idx_frame_bucket   ON frames(game_version, biome, weather, time_of_day);
CREATE INDEX IF NOT EXISTS idx_frame_source   ON frames(source_id);
CREATE INDEX IF NOT EXISTS idx_frame_phash    ON frames(phash);

-- ─────────── Labels: the ground truth ───────────
-- One row per (frame, task). task in {'seg', 'det'}.
-- data_json shape depends on task:
--   seg -> {"mask_png_b64": "...", "classes": ["offroad","road","curb","wall"]}
--   det -> {"boxes": [{"cls":"vehicle","x":0.1,"y":0.2,"w":0.1,"h":0.15}, ...]}
CREATE TABLE IF NOT EXISTS labels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id    INTEGER NOT NULL,
    task        TEXT NOT NULL,
    data_json   TEXT NOT NULL,
    provenance  TEXT NOT NULL,    -- 'manual' | 'proposed_accepted' | 'proposed_edited' | 'auto_trusted'
    model_id    INTEGER,          -- which model made the proposal (NULL if manual)
    round_num   INTEGER DEFAULT 0,
    created_at  REAL NOT NULL,
    UNIQUE(frame_id, task),
    FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_labels_frame ON labels(frame_id);
CREATE INDEX IF NOT EXISTS idx_labels_task  ON labels(task);

-- ─────────── Proposals: model guesses awaiting review ───────────
-- Kept separate from labels so we never confuse "model said so" with "human confirmed".
CREATE TABLE IF NOT EXISTS proposals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     INTEGER NOT NULL,
    task         TEXT NOT NULL,
    data_json    TEXT NOT NULL,
    confidence   REAL,
    uncertainty  REAL,              -- higher = model less sure (drives the active-learning queue)
    model_id     INTEGER,
    created_at   REAL NOT NULL,
    FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_proposals_frame ON proposals(frame_id);
CREATE INDEX IF NOT EXISTS idx_proposals_unc   ON proposals(uncertainty DESC);

-- ─────────── Models: perception training checkpoints ───────────
CREATE TABLE IF NOT EXISTS models (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    round_num       INTEGER NOT NULL,
    path            TEXT NOT NULL,        -- relative path to the .pt file
    onnx_path       TEXT,                 -- optional exported ONNX
    trained_on      INTEGER NOT NULL,     -- number of frames in the train set
    metrics_json    TEXT,                 -- {seg_iou, det_map, val_loss, ...}
    game_versions   TEXT,                 -- CSV of versions covered ('fh4,fh5')
    is_active       INTEGER DEFAULT 0,    -- 1 = current production model (only one row at a time)
    created_at      REAL NOT NULL
);

-- ─────────── Sources: videos and URLs we've ingested ───────────
CREATE TABLE IF NOT EXISTS sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL,        -- 'video_file' | 'youtube_url'
    uri             TEXT NOT NULL,
    title           TEXT,
    game_version    TEXT,
    biome_override  TEXT,
    duration_sec    REAL,
    frames_sampled  INTEGER DEFAULT 0,
    frames_accepted INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'pending',  -- pending|processing|done|failed
    created_at      REAL NOT NULL
);

-- ─────────── Active learning queue ───────────
CREATE TABLE IF NOT EXISTS active_queue (
    frame_id     INTEGER PRIMARY KEY,
    uncertainty  REAL NOT NULL,
    queued_at    REAL NOT NULL,
    round_num    INTEGER NOT NULL,
    FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_queue_unc ON active_queue(uncertainty DESC);

-- ─────────── World map: persistent grid of where the car has driven ───────────
-- (game_version, bin_x, bin_z) is the natural key. Bin size lives in settings.json.
CREATE TABLE IF NOT EXISTS world_map_cells (
    game_version  TEXT NOT NULL,
    bin_x         INTEGER NOT NULL,
    bin_z         INTEGER NOT NULL,
    visits        INTEGER NOT NULL DEFAULT 0,
    mean_speed    REAL,
    last_seen     REAL,
    PRIMARY KEY (game_version, bin_x, bin_z)
);
CREATE INDEX IF NOT EXISTS idx_map_version ON world_map_cells(game_version);

-- ─────────── HUD masks: per-game-version rectangles to ignore ───────────
-- One row per game version. mask_json is a list of {x,y,w,h} in normalized 0-1 coords.
CREATE TABLE IF NOT EXISTS hud_masks (
    game_version  TEXT PRIMARY KEY,
    mask_json     TEXT NOT NULL,
    sample_frame  INTEGER,                   -- frame_id used to author the mask
    updated_at    REAL NOT NULL
);

-- ─────────── PPO checkpoints: RL training artifacts ───────────
CREATE TABLE IF NOT EXISTS ppo_checkpoints (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    iteration     INTEGER NOT NULL,
    path          TEXT NOT NULL,
    perception_id INTEGER,                   -- which perception model it was trained against
    metrics_json  TEXT,                      -- {mean_reward, episode_len, kl, ...}
    is_active     INTEGER DEFAULT 0,
    created_at    REAL NOT NULL,
    FOREIGN KEY(perception_id) REFERENCES models(id)
);