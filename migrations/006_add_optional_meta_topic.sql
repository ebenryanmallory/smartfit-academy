-- Migration: Add optional meta_topic field to lesson_plans table

-- Add completely optional meta_topic field (nullable by default)
ALTER TABLE lesson_plans ADD COLUMN meta_topic TEXT DEFAULT NULL;

-- Optional index for performance when meta_topic is used
-- Only indexes non-null, non-empty values for efficiency
CREATE INDEX IF NOT EXISTS idx_lesson_plans_meta_topic 
ON lesson_plans(meta_topic) 
WHERE meta_topic IS NOT NULL AND meta_topic != '';

-- No data migration required - all existing records will have NULL meta_topic
-- This ensures zero breaking changes for existing functionality 