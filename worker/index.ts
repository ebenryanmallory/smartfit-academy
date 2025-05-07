declare const __STATIC_CONTENT_MANIFEST: string | undefined;

import { Hono } from 'hono'
import { serveStatic } from 'hono/cloudflare-workers'

const app = new Hono()

// Serve static assets using Wrangler's [site] integration
app.use('*', serveStatic({
  root: '/',
  rewriteRequestPath: (path) => (path === '/' ? '/index.html' : path),
  manifest: typeof __STATIC_CONTENT_MANIFEST !== 'undefined' ? __STATIC_CONTENT_MANIFEST : {},
}))

// Example API route (optional)
app.get('/api/hello', (c) => c.json({ message: 'Hello from Hono!' }))

export default app
