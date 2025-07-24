import { Hono } from 'hono'
import { clerkMiddleware } from '@hono/clerk-auth'
import { 
  authRoutes,
  userRoutes,
  lessonRoutes,
  llmRoutes,
  claudeRoutes,
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

// Define valid SPA routes based on App.tsx routes
const validSpaRoutes = [
  '/',
  '/onboarding',
  '/lessons', // This covers /lessons/:id pattern
  '/pricing',
  '/style-guide',
  '/modern-relevance',
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
  // /lessons/:id pattern (including saved lessons like /lessons/saved-123)
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
