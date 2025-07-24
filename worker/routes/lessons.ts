import { Hono } from 'hono'
import { getAuth } from '@hono/clerk-auth'
import type { AppContext } from './types'

const lessonRoutes = new Hono<AppContext>()

// ---------------- User Lesson Plans Routes ----------------

// Protected: Save a lesson plan with its lessons
lessonRoutes.post('/lesson-plans', async (c) => {
  const auth = getAuth(c);
  
  if (!auth?.userId) {
    console.error('No userId in POST lesson plans route - authentication required');
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
    console.log('Received lesson plan data:', JSON.stringify(body, null, 2));
  } catch (e) {
    console.error('Failed to parse JSON body:', e);
    return c.json({ error: 'Invalid JSON body' }, 400);
  }
  
  const { topic, title, totalEstimatedTime, lessons, uuid, meta_topic } = body;
  console.log('Extracted fields:', { topic, title, totalEstimatedTime, lessonsCount: lessons?.length, uuid, meta_topic });
  
  if (!topic || !title || !lessons || !Array.isArray(lessons)) {
    console.error('Missing required fields:', { topic: !!topic, title: !!title, lessons: !!lessons, isArray: Array.isArray(lessons) });
    return c.json({ error: 'Missing required fields: topic, title, lessons' }, 400);
  }

  const db = c.env.DB;
  const userId = auth.userId;
  const email = auth.sessionClaims?.email || `${userId}@unknown.com`;

  console.log('Starting lesson plan save for user:', userId);

  try {
    // Ensure user exists in database (auto-initialize if needed)
    console.log('Ensuring user exists in database...');
    await db.prepare('INSERT OR IGNORE INTO users (id, email) VALUES (?, ?)').bind(userId, email).run();
    console.log('User initialization complete');
    
    // Insert lesson plan
    console.log('Inserting lesson plan with data:', { userId, topic, title, totalEstimatedTime, uuid, meta_topic });
    const lessonPlanResult = await db.prepare(
      'INSERT INTO lesson_plans (user_id, topic, title, total_estimated_time, uuid, meta_topic) VALUES (?, ?, ?, ?, ?, ?)'
    ).bind(userId, topic, title, totalEstimatedTime || null, uuid || null, meta_topic || null).run();
    
    const lessonPlanId = lessonPlanResult.meta.last_row_id;
    console.log('Lesson plan inserted with ID:', lessonPlanId);
    
    // Insert individual lessons
    console.log('Inserting', lessons.length, 'lessons...');
    for (let i = 0; i < lessons.length; i++) {
      const lesson = lessons[i];
      console.log(`Inserting lesson ${i + 1}:`, { 
        title: lesson.title, 
        description: lesson.description?.substring(0, 50) + '...', 
        hasContent: !!lesson.content,
        uuid: lesson.uuid,
        lesson_order: lesson.lesson_order || i + 1
      });
      
      try {
        await db.prepare(
          'INSERT INTO lessons (lesson_plan_id, lesson_order, title, description, content, uuid) VALUES (?, ?, ?, ?, ?, ?)'
        ).bind(lessonPlanId, lesson.lesson_order || i + 1, lesson.title, lesson.description || null, lesson.content || null, lesson.uuid || null).run();
        console.log(`Lesson ${i + 1} inserted successfully`);
      } catch (lessonError) {
        console.error(`Error inserting lesson ${i + 1}:`, lessonError);
        console.error('Lesson data that failed:', lesson);
        throw lessonError;
      }
    }
    
    console.log('All lessons inserted successfully');
    
    // Return the complete lesson plan with lessons
    console.log('Fetching saved lesson plan...');
    const savedLessonPlan = await db.prepare('SELECT * FROM lesson_plans WHERE id = ?').bind(lessonPlanId).first();
    const savedLessons = await db.prepare('SELECT * FROM lessons WHERE lesson_plan_id = ? ORDER BY lesson_order').bind(lessonPlanId).all();
    
    console.log('Lesson plan save completed successfully');
    return c.json({ 
      lessonPlan: {
        ...savedLessonPlan,
        lessons: savedLessons.results
      }
    });
  } catch (error) {
    console.error('Error saving lesson plan - Full error details:', error);
    console.error('Error name:', error instanceof Error ? error.name : 'Unknown');
    console.error('Error message:', error instanceof Error ? error.message : 'Unknown error');
    console.error('Error stack:', error instanceof Error ? error.stack : 'No stack trace');
    
    // Log the specific data that caused the error
    console.error('Request data that caused error:', {
      userId,
      topic,
      title,
      totalEstimatedTime,
      uuid,
      lessonsCount: lessons?.length,
      firstLesson: lessons?.[0]
    });
    
    return c.json({ 
      error: 'Failed to save lesson plan', 
      details: error instanceof Error ? error.message : 'Unknown error',
      errorType: error instanceof Error ? error.name : 'Unknown',
      timestamp: new Date().toISOString()
    }, 500);
  }
});

// Protected: Get all lesson plans for current user
lessonRoutes.get('/lesson-plans', async (c) => {
  const auth = getAuth(c);
  
  if (!auth?.userId) {
    console.error('No userId in lesson plans GET route - authentication required');
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
    
    const lessonPlans = await db.prepare('SELECT * FROM lesson_plans WHERE user_id = ? ORDER BY created_at DESC').bind(userId).all();
    
    // Get lessons for each lesson plan
    const lessonPlansWithLessons = await Promise.all(
      lessonPlans.results.map(async (plan: any) => {
        const lessons = await db.prepare('SELECT * FROM lessons WHERE lesson_plan_id = ? ORDER BY lesson_order').bind(plan.id).all();
        return {
          ...plan,
          lessons: lessons.results
        };
      })
    );
    
    return c.json({ lessonPlans: lessonPlansWithLessons });
  } catch (error) {
    console.error('Error fetching lesson plans:', error);
    return c.json({ error: 'Failed to fetch lesson plans' }, 500);
  }
});

// Protected: Get a specific lesson plan with its lessons
lessonRoutes.get('/lesson-plans/:id', async (c) => {
  const auth = getAuth(c);
  
  if (!auth?.userId) {
    return c.json({ 
      error: 'Unauthorized', 
      details: 'Valid authentication required',
      timestamp: new Date().toISOString()
    }, 401);
  }
  
  const lessonPlanId = c.req.param('id');
  if (!lessonPlanId) {
    return c.json({ error: 'Missing lesson plan ID parameter' }, 400);
  }
  
  const db = c.env.DB;
  const userId = auth.userId;
  
  try {
    // Get lesson plan (ensure it belongs to the user)
    const lessonPlan = await db.prepare('SELECT * FROM lesson_plans WHERE id = ? AND user_id = ?').bind(lessonPlanId, userId).first();
    
    if (!lessonPlan) {
      return c.json({ error: 'Lesson plan not found' }, 404);
    }
    
    // Get lessons for this lesson plan
    const lessons = await db.prepare('SELECT * FROM lessons WHERE lesson_plan_id = ? ORDER BY lesson_order').bind(lessonPlanId).all();
    
    return c.json({ 
      lessonPlan: {
        ...lessonPlan,
        lessons: lessons.results
      }
    });
  } catch (error) {
    console.error('Error fetching lesson plan:', error);
    return c.json({ error: 'Failed to fetch lesson plan' }, 500);
  }
});

// Protected: Delete a lesson plan and all its lessons
lessonRoutes.delete('/lesson-plans/:id', async (c) => {
  const auth = getAuth(c);
  
  if (!auth?.userId) {
    return c.json({ 
      error: 'Unauthorized', 
      details: 'Valid authentication required',
      timestamp: new Date().toISOString()
    }, 401);
  }
  
  const lessonPlanId = c.req.param('id');
  if (!lessonPlanId) {
    return c.json({ error: 'Missing lesson plan ID parameter' }, 400);
  }
  
  const db = c.env.DB;
  const userId = auth.userId;
  
  try {
    // Delete lesson plan (CASCADE will delete associated lessons)
    const result = await db.prepare('DELETE FROM lesson_plans WHERE id = ? AND user_id = ?').bind(lessonPlanId, userId).run();
    
    return c.json({ success: true, deleted: result.meta.changes > 0 });
  } catch (error) {
    console.error('Error deleting lesson plan:', error);
    return c.json({ error: 'Failed to delete lesson plan' }, 500);
  }
});

// Protected: Get a specific lesson by ID
lessonRoutes.get('/lessons/:id', async (c) => {
  const auth = getAuth(c);
  
  if (!auth?.userId) {
    return c.json({ 
      error: 'Unauthorized', 
      details: 'Valid authentication required',
      timestamp: new Date().toISOString()
    }, 401);
  }
  
  const lessonId = c.req.param('id');
  if (!lessonId) {
    return c.json({ error: 'Missing lesson ID parameter' }, 400);
  }
  
  const db = c.env.DB;
  const userId = auth.userId;
  
  try {
    // Get lesson with its lesson plan (ensure it belongs to the user)
    // Try UUID lookup first, then fall back to numeric ID for backward compatibility
    let lesson = await db.prepare(`
      SELECT l.*, lp.topic, lp.meta_topic, lp.title as plan_title, lp.total_estimated_time
      FROM lessons l 
      JOIN lesson_plans lp ON l.lesson_plan_id = lp.id 
      WHERE l.uuid = ? AND lp.user_id = ?
    `).bind(lessonId, userId).first();
    
    // If UUID lookup failed, try numeric ID lookup for backward compatibility
    if (!lesson && /^\d+$/.test(lessonId)) {
      lesson = await db.prepare(`
        SELECT l.*, lp.topic, lp.meta_topic, lp.title as plan_title, lp.total_estimated_time
        FROM lessons l 
        JOIN lesson_plans lp ON l.lesson_plan_id = lp.id 
        WHERE l.id = ? AND lp.user_id = ?
      `).bind(lessonId, userId).first();
    }
    
    if (!lesson) {
      return c.json({ error: 'Lesson not found or access denied' }, 404);
    }
    
    return c.json({ lesson });
  } catch (error) {
    console.error('Error fetching lesson:', error);
    return c.json({ error: 'Failed to fetch lesson' }, 500);
  }
});

// Protected: Update lesson content within a lesson plan
lessonRoutes.put('/lesson-plans/:planId/lessons/:lessonId', async (c) => {
  const auth = getAuth(c);
  
  if (!auth?.userId) {
    return c.json({ 
      error: 'Unauthorized', 
      details: 'Valid authentication required',
      timestamp: new Date().toISOString()
    }, 401);
  }
  
  const lessonPlanId = c.req.param('planId');
  const lessonId = c.req.param('lessonId');
  
  if (!lessonPlanId || !lessonId) {
    return c.json({ error: 'Missing lesson plan ID or lesson ID parameter' }, 400);
  }
  
  // Parse body
  let body;
  try {
    body = await c.req.json();
  } catch (e) {
    return c.json({ error: 'Invalid JSON body' }, 400);
  }
  
  const { content } = body;
  if (content === undefined) {
    return c.json({ error: 'Missing content field' }, 400);
  }
  
  const db = c.env.DB;
  const userId = auth.userId;
  
  try {
    // Verify lesson plan belongs to user and lesson belongs to plan
    const verification = await db.prepare(`
      SELECT l.id 
      FROM lessons l 
      JOIN lesson_plans lp ON l.lesson_plan_id = lp.id 
      WHERE l.id = ? AND lp.id = ? AND lp.user_id = ?
    `).bind(lessonId, lessonPlanId, userId).first();
    
    if (!verification) {
      return c.json({ error: 'Lesson not found or access denied' }, 404);
    }
    
    // Update lesson content
    await db.prepare('UPDATE lessons SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?').bind(content, lessonId).run();
    
    // Return updated lesson
    const updatedLesson = await db.prepare('SELECT * FROM lessons WHERE id = ?').bind(lessonId).first();
    return c.json({ lesson: updatedLesson });
  } catch (error) {
    console.error('Error updating lesson content:', error);
    return c.json({ error: 'Failed to update lesson content' }, 500);
  }
});

export default lessonRoutes 