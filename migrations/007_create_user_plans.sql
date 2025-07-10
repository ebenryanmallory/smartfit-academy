-- Add plan_type column to users table for subscription management
ALTER TABLE users ADD COLUMN plan_type TEXT DEFAULT 'free';

-- Create index for efficient plan lookups
CREATE INDEX IF NOT EXISTS idx_users_plan_type ON users(plan_type);

-- Update existing users to have 'free' plan explicitly
UPDATE users SET plan_type = 'free' WHERE plan_type IS NULL; 