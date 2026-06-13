import { Hono } from 'hono'
import { getAuth } from '@clerk/hono'
import type { AppContext } from './types'
import { FEED_TOPICS, FEED_POST_TYPES } from '../instructions/index'

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

// ---------------- Feed Interaction Routes ----------------

// Protected: Record feed interaction events (batched). Feed posts themselves are
// ephemeral and never stored; only this metadata persists to personalize future batches.
userRoutes.post('/user/feed/interactions', async (c) => {
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

  const VALID_ACTIONS = ['viewed', 'liked', 'more_like_this', 'quiz_correct', 'quiz_incorrect'];
  const events = Array.isArray(body.events) ? body.events.slice(0, 20) : [];
  const valid = events.filter((e: any) =>
    e &&
    typeof e.postType === 'string' &&
    typeof e.topic === 'string' &&
    VALID_ACTIONS.includes(e.action)
  );
  if (valid.length === 0) {
    return c.json({ error: 'No valid events' }, 400);
  }

  const db = c.env.DB;
  const userId = auth.userId;

  try {
    const stmt = db.prepare(
      'INSERT INTO feed_interactions (user_id, post_type, topic, tags, action, difficulty) VALUES (?, ?, ?, ?, ?, ?)'
    );
    await db.batch(valid.map((e: any) => stmt.bind(
      userId,
      e.postType,
      e.topic,
      Array.isArray(e.tags) ? JSON.stringify(e.tags.slice(0, 6).map(String)) : null,
      e.action,
      typeof e.difficulty === 'string' ? e.difficulty : null
    )));
    return c.json({ success: true, recorded: valid.length });
  } catch (error) {
    console.error('Error recording feed interactions:', error);
    return c.json({ error: 'Failed to record feed interactions' }, 500);
  }
});

// ---------------- Feed Preferences Routes ----------------

const FEED_PREF_TOPIC_IDS: string[] = FEED_TOPICS.map(t => t.id)
const MAX_CUSTOM_TOPICS = 5

function parseJsonArray(value: unknown): string[] {
  if (typeof value !== 'string') return []
  try {
    const parsed = JSON.parse(value)
    return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === 'string') : []
  } catch {
    return []
  }
}

// Free-text custom topics go into an LLM prompt: collapse whitespace, strip
// control characters, enforce 2-60 chars. Entries that fail are dropped, not rejected.
function cleanCustomTopics(input: unknown): string[] {
  if (!Array.isArray(input)) return []
  const seen = new Set<string>()
  const out: string[] = []
  for (const entry of input) {
    if (typeof entry !== 'string') continue
    // eslint-disable-next-line no-control-regex
    const cleaned = entry.replace(/[\x00-\x1f\x7f]/g, '').replace(/\s+/g, ' ').trim()
    if (cleaned.length < 2 || cleaned.length > 60) continue
    const key = cleaned.toLowerCase()
    if (seen.has(key) || FEED_PREF_TOPIC_IDS.includes(key)) continue
    seen.add(key)
    out.push(cleaned)
    if (out.length >= MAX_CUSTOM_TOPICS) break
  }
  return out
}

// Protected: Get explicit feed preferences. Empty arrays mean "no restriction".
userRoutes.get('/user/feed/preferences', async (c) => {
  const auth = getAuth(c);
  if (!auth?.userId) {
    return c.json({ error: 'Unauthorized' }, 401);
  }
  const db = c.env.DB;

  try {
    const row = await db.prepare(
      'SELECT topics, custom_topics, post_types FROM feed_preferences WHERE user_id = ?'
    ).bind(auth.userId).first();
    return c.json({
      preferences: {
        topics: parseJsonArray(row?.topics).filter(t => FEED_PREF_TOPIC_IDS.includes(t)),
        customTopics: cleanCustomTopics(parseJsonArray(row?.custom_topics)),
        postTypes: parseJsonArray(row?.post_types).filter(t => (FEED_POST_TYPES as readonly string[]).includes(t))
      }
    });
  } catch (error) {
    console.error('Error fetching feed preferences:', error);
    return c.json({ error: 'Failed to fetch feed preferences' }, 500);
  }
});

// Protected: Save explicit feed preferences (full replace).
userRoutes.post('/user/feed/preferences', async (c) => {
  const auth = getAuth(c);
  if (!auth?.userId) {
    return c.json({ error: 'Unauthorized' }, 401);
  }

  // Parse body
  let body;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: 'Invalid JSON body' }, 400);
  }

  if ((body.topics !== undefined && !Array.isArray(body.topics)) ||
      (body.customTopics !== undefined && !Array.isArray(body.customTopics)) ||
      (body.postTypes !== undefined && !Array.isArray(body.postTypes))) {
    return c.json({ error: 'topics, customTopics and postTypes must be arrays' }, 400);
  }

  const topics = (Array.isArray(body.topics) ? body.topics : [])
    .filter((t: unknown): t is string => typeof t === 'string' && FEED_PREF_TOPIC_IDS.includes(t));
  const customTopics = cleanCustomTopics(body.customTopics);
  // Empty post_types means "all" — never persist a state that can generate nothing.
  const postTypes = (Array.isArray(body.postTypes) ? body.postTypes : [])
    .filter((t: unknown): t is string => typeof t === 'string' && (FEED_POST_TYPES as readonly string[]).includes(t));

  const db = c.env.DB;
  const userId = auth.userId;
  const email = auth.sessionClaims?.email || `${userId}@unknown.com`;

  try {
    // Ensure user exists in database (auto-initialize if needed)
    await db.prepare('INSERT OR IGNORE INTO users (id, email) VALUES (?, ?)').bind(userId, email).run();

    await db.prepare(
      `INSERT INTO feed_preferences (user_id, topics, custom_topics, post_types) VALUES (?, ?, ?, ?)
       ON CONFLICT(user_id) DO UPDATE SET topics = excluded.topics, custom_topics = excluded.custom_topics, post_types = excluded.post_types, updated_at = CURRENT_TIMESTAMP`
    ).bind(userId, JSON.stringify(topics), JSON.stringify(customTopics), JSON.stringify(postTypes)).run();

    return c.json({ preferences: { topics, customTopics, postTypes } });
  } catch (error) {
    console.error('Error saving feed preferences:', error);
    return c.json({ error: 'Failed to save feed preferences' }, 500);
  }
});

export default userRoutes