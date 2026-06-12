-- Migration: interaction metadata for the AI feed.
-- Posts themselves are ephemeral and never stored; only high-level
-- interaction signals are kept to personalize future generation.
CREATE TABLE IF NOT EXISTS feed_interactions (
  id          INTEGER   PRIMARY KEY AUTOINCREMENT,
  user_id     TEXT      NOT NULL,
  post_type   TEXT      NOT NULL,  -- 'did_you_know' | 'quiz' | 'concept' | 'code_snippet'
  topic       TEXT      NOT NULL,  -- lesson topic id, e.g. 'c-python-fundamentals'
  tags        TEXT,                -- JSON array of concept tags
  action      TEXT      NOT NULL,  -- 'viewed' | 'liked' | 'more_like_this' | 'quiz_correct' | 'quiz_incorrect'
  difficulty  TEXT,                -- 'intro' | 'core' | 'stretch'
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feed_interactions_user_created
  ON feed_interactions(user_id, created_at DESC);
