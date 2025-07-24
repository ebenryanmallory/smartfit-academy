-- Migration: Create lesson_plans and lessons tables for AI-generated content

-- Table for storing lesson plans (collections of lessons)
CREATE TABLE IF NOT EXISTS lesson_plans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  topic TEXT NOT NULL,
  title TEXT NOT NULL,
  total_estimated_time TEXT,
  difficulty TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Table for storing individual lessons within lesson plans
CREATE TABLE IF NOT EXISTS lessons (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lesson_plan_id INTEGER NOT NULL,
  lesson_order INTEGER NOT NULL, -- Order within the lesson plan
  title TEXT NOT NULL,
  description TEXT,
  content TEXT, -- Markdown content for the lesson
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (lesson_plan_id) REFERENCES lesson_plans(id) ON DELETE CASCADE
);

-- Index for better query performance
CREATE INDEX IF NOT EXISTS idx_lesson_plans_user_id ON lesson_plans(user_id);
CREATE INDEX IF NOT EXISTS idx_lessons_lesson_plan_id ON lessons(lesson_plan_id); 