import { Hono } from 'hono'
import { clerkMiddleware, getAuth } from '@hono/clerk-auth'

// Define the user type
interface User {
  id: string;
  emailAddresses: Array<{ emailAddress: string }>;
  firstName: string | null;
  lastName: string | null;
}

// Extend the Hono context type
type Variables = {
  user: User;
}

import type { D1Database } from '@cloudflare/workers-types';

const app = new Hono<{ Variables: Variables; Bindings: { DB: D1Database; WORKERS_AI_TOKEN: string; WORKERS_AI_ACCOUNT_ID: string } }>()

// Apply Clerk middleware to all API routes
app.use('/api/*', clerkMiddleware())

// Public routes
app.get('/hello', (c) => {
  return c.json({ message: 'Hello from Hono!' })
})

// Protected routes
app.get('/api/protected/user', (c) => {
  const auth = getAuth(c)
  if (!auth?.userId) {
    return c.json({ error: 'Unauthorized' }, 401)
  }
  return c.json({
    message: 'Protected route accessed successfully',
    userId: auth.userId,
  })
})

// Admin routes (protected + role check)
app.use('/api/admin/*', clerkMiddleware())

// Example admin route
app.get('/api/admin/stats', (c) => {
  return c.json({ 
    message: 'Admin route accessed successfully',
    stats: {
      totalUsers: 100,
      activeUsers: 75,
      totalLessons: 50,
    }
  })
})

// D1 example route: get all users from the 'users' table
app.get('/api/d1/users', async (c) => {
  const db = c.env.DB;
  const result = await db.prepare('SELECT * FROM users').all();
  return c.json({ users: result.results });
});

// Protected: Initialize user in D1 from Clerk session
app.post('/api/d1/user/init', async (c) => {
  const auth = getAuth(c);
  if (!auth?.userId || !auth.sessionClaims?.email) {
    return c.json({ error: 'Unauthorized or missing email' }, 401);
  }
  const db = c.env.DB;
  const userId = auth.userId;
  const email = auth.sessionClaims.email;

  // Check if user exists
  const existing = await db.prepare('SELECT * FROM users WHERE id = ?').bind(userId).first();
  if (!existing) {
    // Insert user
    await db.prepare('INSERT INTO users (id, email) VALUES (?, ?)').bind(userId, email).run();
  }
  // Return user row
  const user = await db.prepare('SELECT * FROM users WHERE id = ?').bind(userId).first();
  return c.json({ user });
});

// Protected: Get current user from D1
app.get('/api/d1/user', async (c) => {
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

// ---------------- User Progress Routes ----------------

// Protected: Upsert (insert or update) user progress for a lesson
app.post('/api/d1/user/progress', async (c) => {
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
app.get('/api/d1/user/progress', async (c) => {
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

// ------------------------------------------------------

// Llama3 LLM endpoint
app.post('/llm/llama3', async (c) => {

  const WORKERS_AI_TOKEN = c.env.WORKERS_AI_TOKEN;
  const ACCOUNT_ID = c.env.WORKERS_AI_ACCOUNT_ID;

  if (!WORKERS_AI_TOKEN || !ACCOUNT_ID) {
    return c.json({ error: 'The assistant is currently unavailable. Please try again later.' }, 500);
  }

  // Parse messages from the request body
  let body;
  try {
    body = await c.req.json();
  } catch (e) {
    return c.json({ error: 'Invalid JSON body' }, 400);
  }

  const { messages } = body;
  if (!messages || !Array.isArray(messages)) {
    return c.json({ error: 'Missing or invalid messages array' }, 400);
  }

  // Call Cloudflare Workers AI REST API
  const aiUrl = `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct`;
  const aiRes = await fetch(aiUrl, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${WORKERS_AI_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ messages }),
  });

  const aiData = await aiRes.json();
  if (!aiRes.ok) {
    return c.json({ error: aiData.error || 'AI request failed' }, aiRes.status as any);
  }

  return c.json(aiData);
});

export default app
