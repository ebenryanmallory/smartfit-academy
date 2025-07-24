import { Hono } from 'hono'
import { getAuth } from '@hono/clerk-auth'
import type { AppContext } from './types'

const userRoutes = new Hono<AppContext>()

// D1 example route: get all users from the 'users' table
userRoutes.get('/users', async (c) => {
  const db = c.env.DB;
  const result = await db.prepare('SELECT * FROM users').all();
  return c.json({ users: result.results });
});

// Protected: Initialize user in D1 from Clerk session
userRoutes.post('/user/init', async (c) => {
  console.log('User init endpoint called');
  const auth = getAuth(c);
  
  console.log('Auth object:', JSON.stringify(auth, null, 2));
  
  if (!auth?.userId) {
    console.error('No userId in auth - authentication required');
    console.error('Auth details:', auth);
    return c.json({ 
      error: 'Unauthorized', 
      details: 'Valid authentication required',
      timestamp: new Date().toISOString()
    }, 401);
  }
  
  const db = c.env.DB;
  const userId = auth.userId;
  const email = auth.sessionClaims?.email || `${userId}@unknown.com`;
  
  console.log(`Initializing user: ${userId} with email: ${email}`);

  try {
    // Check if user exists
    console.log('Checking if user exists...');
    const existing = await db.prepare('SELECT * FROM users WHERE id = ?').bind(userId).first();
    if (!existing) {
      // Insert user
      console.log('Creating new user...');
      await db.prepare('INSERT INTO users (id, email) VALUES (?, ?)').bind(userId, email).run();
      console.log(`Created new user: ${userId} with email: ${email}`);
    } else {
      console.log(`User already exists: ${userId}`);
    }
    
    // Return user row
    console.log('Fetching user record...');
    const user = await db.prepare('SELECT * FROM users WHERE id = ?').bind(userId).first();
    console.log('User record:', user);
    return c.json({ user });
  } catch (error) {
    console.error('Error in user init:', error);
    return c.json({ error: 'Failed to initialize user' }, 500);
  }
});

// Protected: Get current user from D1
userRoutes.get('/user', async (c) => {
  const auth = getAuth(c);
  if (!auth?.userId) {
    return c.json({ error: 'Unauthorized' }, 401);
  }
  const db = c.env.DB;
  const userId = auth.userId;
  const user = await db.prepare('SELECT * FROM users WHERE id = ?').bind(userId).first();
  if (!user) {
    return c.json({ error: 'User not found' }, 404);
  }
  return c.json({ user });
});

// Protected: Update user education level
userRoutes.post('/user/education-level', async (c) => {
  const auth = getAuth(c);
  if (!auth?.userId) {
    return c.json({ error: 'Unauthorized' }, 401);
  }

  // Parse body
  let body;
  try {
    body = await c.req.json();
  } catch (e) {
    return c.json({ error: 'Invalid JSON body' }, 400);
  }
  
  const { educationLevel } = body;
  if (!educationLevel || typeof educationLevel !== 'string') {
    return c.json({ error: 'Missing or invalid educationLevel' }, 400);
  }

  // Validate education level
  const validLevels = ['elementary', 'highschool', 'undergrad', 'grad'];
  if (!validLevels.includes(educationLevel)) {
    return c.json({ error: 'Invalid education level. Must be one of: elementary, highschool, undergrad, grad' }, 400);
  }

  const db = c.env.DB;
  const userId = auth.userId;
  const email = auth.sessionClaims?.email || `${userId}@unknown.com`;

  try {
    // Ensure user exists in database (auto-initialize if needed)
    await db.prepare('INSERT OR IGNORE INTO users (id, email) VALUES (?, ?)').bind(userId, email).run();
    
    // Update education level
    await db.prepare('UPDATE users SET education_level = ? WHERE id = ?').bind(educationLevel, userId).run();
    
    // Return updated user
    const user = await db.prepare('SELECT * FROM users WHERE id = ?').bind(userId).first();
    return c.json({ user });
  } catch (error) {
    console.error('Error updating education level:', error);
    return c.json({ error: 'Failed to update education level' }, 500);
  }
});

// ---------------- User Progress Routes ----------------

// Protected: Upsert (insert or update) user progress for a lesson
userRoutes.post('/user/progress', async (c) => {
  const auth = getAuth(c);
  if (!auth?.userId) {
    return c.json({ error: 'Unauthorized' }, 401);
  }

  // Parse body
  let body;
  try {
    body = await c.req.json();
  } catch (e) {
    return c.json({ error: 'Invalid JSON body' }, 400);
  }
  const { lessonId, completed, score, additionalData } = body;
  if (!lessonId) {
    return c.json({ error: 'Missing lessonId' }, 400);
  }

  const db = c.env.DB;
  const userId = auth.userId;

  // Upsert user progress
  await db.prepare(
    `INSERT INTO user_progress (user_id, lesson_id, completed, score, additional_data) VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(user_id, lesson_id) DO UPDATE SET completed = excluded.completed, score = excluded.score, additional_data = excluded.additional_data, updated_at = CURRENT_TIMESTAMP`
  ).bind(userId, lessonId, completed ? 1 : 0, score ?? null, additionalData ? JSON.stringify(additionalData) : null).run();

  const progress = await db.prepare('SELECT * FROM user_progress WHERE user_id = ? AND lesson_id = ?').bind(userId, lessonId).first();
  return c.json({ progress });
});

// Protected: Get progress for current user (optionally filter by lesson)
userRoutes.get('/user/progress', async (c) => {
  const auth = getAuth(c);
  if (!auth?.userId) {
    return c.json({ error: 'Unauthorized' }, 401);
  }
  const db = c.env.DB;
  const userId = auth.userId;
  const lessonId = c.req.query('lessonId');

  let result;
  if (lessonId) {
    result = await db.prepare('SELECT * FROM user_progress WHERE user_id = ? AND lesson_id = ?').bind(userId, lessonId).first();
    return c.json({ progress: result });
  }
  result = await db.prepare('SELECT * FROM user_progress WHERE user_id = ?').bind(userId).all();
  return c.json({ progress: result.results });
});

// ---------------- User Topics Routes ----------------

// Protected: Add a topic to user's saved topics
userRoutes.post('/user/topics', async (c) => {
  const auth = getAuth(c);
  
  if (!auth?.userId) {
    console.error('No userId in POST topics route - authentication required');
    return c.json({ 
      error: 'Unauthorized', 
      details: 'Valid authentication required',
      timestamp: new Date().toISOString()
    }, 401);
  }

  // Parse body
  let body;
  try {
    body = await c.req.json();
  } catch (e) {
    return c.json({ error: 'Invalid JSON body' }, 400);
  }
  const { topic } = body;
  if (!topic || typeof topic !== 'string' || topic.trim().length === 0) {
    return c.json({ error: 'Missing or invalid topic' }, 400);
  }

  const db = c.env.DB;
  const userId = auth.userId;
  const email = auth.sessionClaims?.email || `${userId}@unknown.com`;
  const cleanTopic = topic.trim();

  try {
    // Ensure user exists in database (auto-initialize if needed)
    await db.prepare('INSERT OR IGNORE INTO users (id, email) VALUES (?, ?)').bind(userId, email).run();
    
    // Insert topic (will fail silently if duplicate due to UNIQUE constraint)
    await db.prepare('INSERT OR IGNORE INTO user_topics (user_id, topic) VALUES (?, ?)').bind(userId, cleanTopic).run();
    
    // Get the topic record
    const savedTopic = await db.prepare('SELECT * FROM user_topics WHERE user_id = ? AND topic = ?').bind(userId, cleanTopic).first();
    return c.json({ topic: savedTopic });
  } catch (error) {
    console.error('Error saving topic:', error);
    return c.json({ error: 'Failed to save topic' }, 500);
  }
});

// Protected: Get all topics for current user
userRoutes.get('/user/topics', async (c) => {
  const auth = getAuth(c);
  
  if (!auth?.userId) {
    console.error('No userId in topics GET route - authentication required');
    return c.json({ 
      error: 'Unauthorized', 
      details: 'Valid authentication required',
      timestamp: new Date().toISOString()
    }, 401);
  }
  
  const db = c.env.DB;
  const userId = auth.userId;
  
  try {
    // Ensure user exists in database (auto-initialize if needed)
    const email = auth.sessionClaims?.email || `${userId}@unknown.com`;
    await db.prepare('INSERT OR IGNORE INTO users (id, email) VALUES (?, ?)').bind(userId, email).run();
    
    const result = await db.prepare('SELECT * FROM user_topics WHERE user_id = ? ORDER BY created_at DESC').bind(userId).all();
    return c.json({ topics: result.results });
  } catch (error) {
    console.error('Error fetching topics:', error);
    return c.json({ error: 'Failed to fetch topics' }, 500);
  }
});

// Protected: Remove a topic from user's saved topics
userRoutes.delete('/user/topics/:topic', async (c) => {
  const auth = getAuth(c);
  
  if (!auth?.userId) {
    console.error('No userId in DELETE topics route - authentication required');
    return c.json({ 
      error: 'Unauthorized', 
      details: 'Valid authentication required',
      timestamp: new Date().toISOString()
    }, 401);
  }
  
  const topic = c.req.param('topic');
  if (!topic) {
    return c.json({ error: 'Missing topic parameter' }, 400);
  }
  
  const db = c.env.DB;
  const userId = auth.userId;
  
  try {
    const result = await db.prepare('DELETE FROM user_topics WHERE user_id = ? AND topic = ?').bind(userId, decodeURIComponent(topic)).run();
    return c.json({ success: true, deleted: result.meta.changes > 0 });
  } catch (error) {
    return c.json({ error: 'Failed to delete topic' }, 500);
  }
});

export default userRoutes 