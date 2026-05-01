import { Hono } from 'hono'
import { clerkMiddleware } from '@hono/clerk-auth'
import {
  authRoutes,
  userRoutes,
  lessonRoutes,
  llmRoutes,
  claudeRoutes,
  analyticsRoutes,
  type AppContext
} from './routes'

const app = new Hono<AppContext>()

// Debug logging middleware
app.use('*', async (c, next) => {
  console.log(`[DEBUG] ${c.req.method} ${c.req.path}`)
  await next()
})

// Apply Clerk middleware to all API routes and Claude routes
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

// Apply Clerk middleware to Claude routes (premium features)
app.use('/claude/*', async (c, next) => {
  try {
    console.log(`[CLERK] Processing Claude ${c.req.method} ${c.req.path}`)
    await clerkMiddleware()(c, next)
  } catch (error) {
    console.error('[CLERK] Claude authentication middleware failed:', error)
    console.error('[CLERK] Error details:', JSON.stringify(error, null, 2))
    console.error('[CLERK] Request headers:', JSON.stringify(c.req.header(), null, 2))
    return c.json({ 
      error: 'Authentication failed', 
      details: 'Clerk middleware error - check server logs',
      timestamp: new Date().toISOString()
    }, 500)
  }
})

// Mount route modules
app.route('/api', authRoutes)
app.route('/api/d1', userRoutes)
app.route('/api/d1/user', lessonRoutes)
app.route('/llm', llmRoutes)
app.route('/claude', claudeRoutes)
app.route('/track', analyticsRoutes)

// Serve index.html for all GET requests — React Router handles client-side routing
app.get('*', async (c) => {
  const url = new URL('/', c.req.url);
  return c.env.ASSETS.fetch(url.toString()) as unknown as Response;
});

export default app
