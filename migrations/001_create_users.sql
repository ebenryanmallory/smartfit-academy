CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,         -- Clerk user ID
  email TEXT NOT NULL UNIQUE,  -- Clerk email
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
