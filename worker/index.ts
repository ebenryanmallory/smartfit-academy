import { Hono } from 'hono'
import { serveStatic } from 'hono/cloudflare-workers'

const app = new Hono()

// Serve static assets from the Vite build output directory
app.use(
  '*',
  serveStatic({
    root: '../dist', // Adjust if your build output is elsewhere
    rewriteRequestPath: (path) => (path === '/' ? '/index.html' : path),
  })
)

// Example API route (optional)
app.get('/api/hello', (c) => c.json({ message: 'Hello from Hono!' }))

export default app
