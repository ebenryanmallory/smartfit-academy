-- Migration: Create page_views table for first-party analytics

CREATE TABLE IF NOT EXISTS page_views (
  id           INTEGER   PRIMARY KEY AUTOINCREMENT,
  path         TEXT      NOT NULL,
  title        TEXT,
  referrer     TEXT,
  country      TEXT,        -- 2-letter ISO code from CF request metadata
  device_type  TEXT,        -- mobile / tablet / desktop (parsed server-side from UA)
  browser      TEXT,        -- chrome / safari / firefox / edge / other
  os           TEXT,        -- ios / android / windows / macos / linux / other
  language     TEXT,        -- navigator.language, e.g. "en-US"
  screen_width INTEGER,     -- window.screen.width in px
  utm_source   TEXT,
  utm_medium   TEXT,
  utm_campaign TEXT,
  utm_term     TEXT,
  utm_content  TEXT,
  viewed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_page_views_viewed_at ON page_views(viewed_at);
CREATE INDEX IF NOT EXISTS idx_page_views_path ON page_views(path);
CREATE INDEX IF NOT EXISTS idx_page_views_utm_source ON page_views(utm_source)
  WHERE utm_source IS NOT NULL;
