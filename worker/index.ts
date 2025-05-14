declare const __STATIC_CONTENT_MANIFEST: string | undefined;

import { Hono } from 'hono'
import { serveStatic } from 'hono/cloudflare-workers'
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

const app = new Hono<{ Variables: Variables; Bindings: { DB: D1Database } }>()

// Serve static assets using Wrangler's [site] integration
app.use('*', serveStatic({
  root: '/',
  rewriteRequestPath: (path) => {
    // Handle root path
    if (path === '/') return '/index.html'
    // Handle paths without extension (SPA routes)
    if (!path.includes('.')) return '/index.html'
    return path
  },
  manifest: {}
}))

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

export default app
