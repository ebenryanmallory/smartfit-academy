-- Migration: add education_level to users.
-- Migration 003 was committed as an empty file, so the column it was meant to
-- add never existed in any environment. Added here under a new name because
-- 003 is already recorded as applied and would not re-run.
ALTER TABLE users ADD COLUMN education_level TEXT;
