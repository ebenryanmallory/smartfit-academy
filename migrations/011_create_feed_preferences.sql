-- Migration: explicit feed preferences (one row per user).
-- Topics/types are stored as JSON arrays in TEXT columns, matching the
-- existing convention (feed_interactions.tags, user_progress.additional_data).
-- NULL or empty array means "no restriction" (all topics / all types).
CREATE TABLE IF NOT EXISTS feed_preferences (
  user_id       TEXT      PRIMARY KEY,
  topics        TEXT,                -- JSON array of catalog topic ids, e.g. ["c-intro-ai"]
  custom_topics TEXT,                -- JSON array of user-entered topic strings (max 5, 2-60 chars each)
  post_types    TEXT,                -- JSON array of enabled post types
  updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);
