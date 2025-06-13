-- Migration: Add UUID columns for user-facing identifiers

-- Add UUID column to lesson_plans table (nullable, will be populated for new records)
ALTER TABLE lesson_plans ADD COLUMN uuid TEXT;

-- Add UUID column to lessons table (nullable, will be populated for new records)
ALTER TABLE lessons ADD COLUMN uuid TEXT;

-- Create indexes for UUID lookups (these will be used frequently)
CREATE INDEX IF NOT EXISTS idx_lesson_plans_uuid ON lesson_plans(uuid);
CREATE INDEX IF NOT EXISTS idx_lessons_uuid ON lessons(uuid); 