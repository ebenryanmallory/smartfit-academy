import { Hono } from 'hono';
import { clerkMiddleware, getAuth } from '@hono/clerk-auth';
const authRoutes = new Hono();
// Protected routes
authRoutes.get('/protected/user', (c) => {
    const auth = getAuth(c);
    if (!auth?.userId) {
        return c.json({ error: 'Unauthorized' }, 401);
    }
    return c.json({
        message: 'Protected route accessed successfully',
        userId: auth.userId,
    });
});
// Admin routes (protected + role check)
authRoutes.use('/admin/*', clerkMiddleware());
// Example admin route
authRoutes.get('/admin/stats', (c) => {
    return c.json({
        message: 'Admin route accessed successfully',
        stats: {
            totalUsers: 100,
            activeUsers: 75,
            totalLessons: 50,
        }
    });
});
export default authRoutes;
