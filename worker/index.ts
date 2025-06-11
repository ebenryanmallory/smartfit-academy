import { Hono } from 'hono'
import { clerkMiddleware, getAuth } from '@hono/clerk-auth'
import { educationalAssistantInstructions } from './instructions/educational-assistant.js'

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

// Debug logging middleware
app.use('*', async (c, next) => {
  console.log(`[DEBUG] ${c.req.method} ${c.req.path}`)
  await next()
})

// Apply Clerk middleware to all API routes
app.use('/api/*', async (c, next) => {
  try {
    console.log(`[CLERK] Processing ${c.req.method} ${c.req.path}`)
    await clerkMiddleware()(c, next)
  } catch (error) {
    console.error('[CLERK] Authentication middleware failed:', error)
    console.error('[CLERK] Error details:', JSON.stringify(error, null, 2))
    console.error('[CLERK] Request headers:', JSON.stringify(c.req.header(), null, 2))
    return c.json({ 
      error: 'Authentication failed', 
      details: 'Clerk middleware error - check server logs',
      timestamp: new Date().toISOString()
    }, 500)
  }
})

// Debug route to test logging
app.get('/debug', (c) => {
  console.log('[DEBUG-ROUTE] This route was hit')
  return c.json({ message: 'Debug route working', timestamp: new Date().toISOString() })
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

// ---------------- User Topics Routes ----------------

// Protected: Add a topic to user's saved topics
app.post('/api/d1/user/topics', async (c) => {
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
app.get('/api/d1/user/topics', async (c) => {
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
app.delete('/api/d1/user/topics/:topic', async (c) => {
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

// ------------------------------------------------------

// Llama3 LLM endpoint
app.post('/llm/llama3', async (c) => {
  console.log('Llama3 endpoint called');

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

  const { messages, useCustomInstructions } = body;
  if (!messages || !Array.isArray(messages)) {
    return c.json({ error: 'Missing or invalid messages array' }, 400);
  }

  // Prepare messages array with optional custom instructions
  let processedMessages = [...messages];
  
  if (useCustomInstructions) {
    const systemInstruction = {
      role: 'system',
      content: educationalAssistantInstructions
    };
    
    // Insert system instruction at the beginning, or replace existing system message
    if (processedMessages.length > 0 && processedMessages[0].role === 'system') {
      processedMessages[0] = systemInstruction;
    } else {
      processedMessages.unshift(systemInstruction);
    }
  }

  // Call Cloudflare Workers AI REST API
  const aiUrl = `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct`;
  const aiRes = await fetch(aiUrl, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${WORKERS_AI_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ messages: processedMessages }),
  });

  const aiData = await aiRes.json();
  if (!aiRes.ok) {
    return c.json({ error: aiData.error || 'AI request failed' }, aiRes.status as any);
  }

  return c.json(aiData);
});

// Define valid SPA routes based on App.tsx routes
const validSpaRoutes = [
  '/',
  '/onboarding',
  '/sample-lesson',
  '/lessons', // This covers /lessons/:id pattern
  '/style-guide',
  '/dashboard',
  '/dashboard/lessons' // This covers /dashboard/lessons and /dashboard/lessons/:id patterns
];

// Helper function to check if a path matches valid SPA routes
function isValidSpaRoute(path: string): boolean {
  // Exact match
  if (validSpaRoutes.includes(path)) {
    return true;
  }
  
  // Check for dynamic routes
  // /lessons/:id pattern
  if (path.startsWith('/lessons/') && path.split('/').length === 3) {
    return true;
  }
  
  // /dashboard/lessons/:id pattern
  if (path.startsWith('/dashboard/lessons/') && path.split('/').length === 4) {
    return true;
  }
  
  return false;
}

// Catch-all for non-GET requests that don't match any routes
app.all('*', async (c) => {
  const path = c.req.path;
  const method = c.req.method;
  
  console.log(`Unmatched route: ${method} ${path}`);
  
  // For non-GET requests to API or LLM routes, return 404
  if (method !== 'GET' && (path.startsWith('/api/') || path.startsWith('/llm/'))) {
    console.log(`Returning 404 for unmatched ${method} ${path}`);
    return c.notFound();
  }
  
  // For GET requests, handle SPA routing
  if (method === 'GET') {
    // Don't serve SPA for API routes
    if (path.startsWith('/api/') || path.startsWith('/llm/')) {
      return c.notFound();
    }
    
    // Don't serve SPA for static assets (files with extensions)
    if (path.includes('.') && !path.endsWith('/')) {
      return c.notFound();
    }
    
    // Check if this is a valid SPA route
    if (!isValidSpaRoute(path)) {
      // Serve a proper 404 page
      return c.html(`<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/ring.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>404 - Page Not Found | SmartFit Academy</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Geist:wght@100..900&display=swap" rel="stylesheet">
    <style>
      body { font-family: 'Geist', sans-serif; margin: 0; padding: 2rem; background: #f9fafb; }
      .container { max-width: 42rem; margin: 0 auto; text-align: center; }
      h1 { font-size: 3rem; font-weight: bold; margin-bottom: 1.5rem; color: #111827; }
      p { font-size: 1.125rem; color: #6b7280; margin-bottom: 2rem; }
      a { color: #3b82f6; text-decoration: none; font-weight: 500; }
      a:hover { text-decoration: underline; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>404 - Page Not Found</h1>
      <p>The page you're looking for doesn't exist.</p>
      <a href="/">← Back to Home</a>
    </div>
  </body>
</html>`, 404);
    }
    
    // For valid SPA routes, redirect to root
    // This ensures users get to a working page rather than a 404
    // React Router can then handle navigation from there
    return c.redirect('/', 302);
  }
  
  // For any other method, return 404
  return c.notFound();
});

export default app
