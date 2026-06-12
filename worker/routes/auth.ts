import { Hono } from 'hono'
import { getAuth } from '@clerk/hono'
import type { AppContext } from './types'

const authRoutes = new Hono<AppContext>()

// Protected routes
authRoutes.get('/protected/user', (c) => {
  const auth = getAuth(c)
  if (!auth?.userId) {
    return c.json({ error: 'Unauthorized' }, 401)
  }
  return c.json({
    message: 'Protected route accessed successfully',
    userId: auth.userId,
  })
})

// Example admin route
authRoutes.get('/admin/stats', (c) => {
  const auth = getAuth(c)
  if (!auth?.userId) {
    return c.json({ error: 'Unauthorized' }, 401)
  }
  return c.json({
    message: 'Admin route accessed successfully',
    stats: {
      totalUsers: 100,
      activeUsers: 75,
      totalLessons: 50,
    }
  })
})

export default authRoutes 