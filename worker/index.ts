import { Hono } from 'hono'
import { clerkMiddleware, getAuth } from '@hono/clerk-auth'
import { 
  educationalAssistantInstructions,
  lessonPlanGeneratorInstructions,
  lessonContentGeneratorInstructions
} from './instructions/index.js'

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

// Protected: Update user education level
app.post('/api/d1/user/education-level', async (c) => {
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
  
  const { educationLevel } = body;
  if (!educationLevel || typeof educationLevel !== 'string') {
    return c.json({ error: 'Missing or invalid educationLevel' }, 400);
  }

  // Validate education level
  const validLevels = ['elementary', 'highschool', 'undergrad', 'grad'];
  if (!validLevels.includes(educationLevel)) {
    return c.json({ error: 'Invalid education level. Must be one of: elementary, highschool, undergrad, grad' }, 400);
  }

  const db = c.env.DB;
  const userId = auth.userId;
  const email = auth.sessionClaims?.email || `${userId}@unknown.com`;

  try {
    // Ensure user exists in database (auto-initialize if needed)
    await db.prepare('INSERT OR IGNORE INTO users (id, email) VALUES (?, ?)').bind(userId, email).run();
    
    // Update education level
    await db.prepare('UPDATE users SET education_level = ? WHERE id = ?').bind(educationLevel, userId).run();
    
    // Return updated user
    const user = await db.prepare('SELECT * FROM users WHERE id = ?').bind(userId).first();
    return c.json({ user });
  } catch (error) {
    console.error('Error updating education level:', error);
    return c.json({ error: 'Failed to update education level' }, 500);
  }
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

// ---------------- User Lesson Plans Routes ----------------

// Protected: Save a lesson plan with its lessons
app.post('/api/d1/user/lesson-plans', async (c) => {
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
  
  const { topic, title, totalEstimatedTime, lessons, uuid } = body;
  console.log('Extracted fields:', { topic, title, totalEstimatedTime, lessonsCount: lessons?.length, uuid });
  
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
    console.log('Inserting lesson plan with data:', { userId, topic, title, totalEstimatedTime, uuid });
    const lessonPlanResult = await db.prepare(
      'INSERT INTO lesson_plans (user_id, topic, title, total_estimated_time, uuid) VALUES (?, ?, ?, ?, ?)'
    ).bind(userId, topic, title, totalEstimatedTime || null, uuid || null).run();
    
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
    console.error('Error name:', error.name);
    console.error('Error message:', error.message);
    console.error('Error stack:', error.stack);
    
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
      details: error.message,
      errorType: error.name,
      timestamp: new Date().toISOString()
    }, 500);
  }
});

// Protected: Get all lesson plans for current user
app.get('/api/d1/user/lesson-plans', async (c) => {
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
app.get('/api/d1/user/lesson-plans/:id', async (c) => {
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
app.delete('/api/d1/user/lesson-plans/:id', async (c) => {
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
app.get('/api/d1/user/lessons/:id', async (c) => {
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
      SELECT l.*, lp.topic, lp.title as plan_title, lp.total_estimated_time
      FROM lessons l 
      JOIN lesson_plans lp ON l.lesson_plan_id = lp.id 
      WHERE l.uuid = ? AND lp.user_id = ?
    `).bind(lessonId, userId).first();
    
    // If UUID lookup failed, try numeric ID lookup for backward compatibility
    if (!lesson && /^\d+$/.test(lessonId)) {
      lesson = await db.prepare(`
        SELECT l.*, lp.topic, lp.title as plan_title, lp.total_estimated_time
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
app.put('/api/d1/user/lesson-plans/:planId/lessons/:lessonId', async (c) => {
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

  const { messages, useCustomInstructions, instructionType, educationLevel } = body;
  if (!messages || !Array.isArray(messages)) {
    return c.json({ error: 'Missing or invalid messages array' }, 400);
  }

  // Prepare messages array with optional custom instructions
  let processedMessages = [...messages];
  
  if (useCustomInstructions || instructionType) {
    let instructionContent = educationalAssistantInstructions; // default
    
    // Select instruction type if specified
    if (instructionType) {
      switch (instructionType) {
        case 'educationalAssistant':
          instructionContent = educationalAssistantInstructions;
          break;
        case 'lessonPlanGenerator':
          // Use education level if provided, default to 'undergrad'
          const validEducationLevelsForPlan = ['elementary', 'highschool', 'undergrad', 'grad'];
          const targetEducationLevelForPlan = validEducationLevelsForPlan.includes(educationLevel) ? educationLevel : 'undergrad';
          instructionContent = lessonPlanGeneratorInstructions(targetEducationLevelForPlan);
          console.log(`Using lesson plan generator for education level: ${targetEducationLevelForPlan}`);
          break;
        case 'lessonContentGenerator':
          // Use education level if provided, default to 'undergrad'
          const validEducationLevels = ['elementary', 'highschool', 'undergrad', 'grad'];
          const targetEducationLevel = validEducationLevels.includes(educationLevel) ? educationLevel : 'undergrad';
          instructionContent = lessonContentGeneratorInstructions(targetEducationLevel);
          console.log(`Using lesson content generator for education level: ${targetEducationLevel}`);
          break;
        default:
          console.warn(`Unknown instruction type: ${instructionType}, using default`);
          instructionContent = educationalAssistantInstructions;
      }
    }
    
    const systemInstruction = {
      role: 'system',
      content: instructionContent
    };
    
    // Insert system instruction at the beginning, or replace existing system message
    if (processedMessages.length > 0 && processedMessages[0].role === 'system') {
      processedMessages[0] = systemInstruction;
    } else {
      processedMessages.unshift(systemInstruction);
    }
  }

  try {
    // Call Cloudflare Workers AI REST API
    const aiUrl = `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct`;
    
    // Prepare request body with appropriate max_tokens based on instruction type
    const requestBody: any = { messages: processedMessages };
    
    // Set higher token limits for lesson plan generation to prevent truncation
    if (instructionType === 'lessonPlanGenerator') {
      requestBody.max_tokens = 4096; // Much higher limit for comprehensive lesson plans
      requestBody.temperature = 0.7; // Slightly higher creativity for educational content
    } else if (instructionType === 'lessonContentGenerator') {
      requestBody.max_tokens = 2048; // Higher limit for detailed lesson content
      requestBody.temperature = 0.7;
    } else {
      requestBody.max_tokens = 1024; // Higher than default for general educational assistance
      requestBody.temperature = 0.6;
    }
    
    // Check if streaming is requested
    const isStreaming = requestBody.stream === true;

    // Prepare request body with streaming parameter
    const requestBodyForAI = {
      messages: processedMessages,
      stream: isStreaming,
      max_tokens: requestBody.max_tokens,
      temperature: requestBody.temperature
    };

    if (isStreaming) {
      // Handle streaming response
      const aiRes = await fetch(aiUrl, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${WORKERS_AI_TOKEN}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBodyForAI),
      });

      if (!aiRes.ok) {
        const errorText = await aiRes.text();
        console.error('Cloudflare Workers AI API error:', aiRes.status, errorText);
        return c.json({ error: `AI API error: ${aiRes.status}` }, 500);
      }

      // Return the streaming response directly
      return new Response(aiRes.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        },
      });
    } else {
      // Handle non-streaming response (existing code)
      const aiRes = await fetch(aiUrl, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${WORKERS_AI_TOKEN}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const aiData = await aiRes.json();
      if (!aiRes.ok) {
        console.error('AI API error:', aiData);
        return c.json({ error: aiData.error || 'AI request failed' }, aiRes.status as any);
      }

      // Extract the response content
      const responseContent = aiData.result?.response || aiData.response || '';
      console.log('Raw AI response length:', responseContent.length);
      console.log('Raw AI response preview:', responseContent.substring(0, 200) + '...');
      
      // Log token usage if available for debugging
      if (aiData.result?.usage) {
        console.log('Token usage:', aiData.result.usage);
      }

      // For lesson plan generation, validate the JSON structure
      if (instructionType === 'lessonPlanGenerator') {
        try {
          // Check if response looks like it might be truncated
          if (responseContent.length < 50) {
            throw new Error('Response appears to be too short or empty');
          }

          // Try to find JSON boundaries
          const jsonStart = responseContent.indexOf('{');
          const jsonEnd = responseContent.lastIndexOf('}');
          
          if (jsonStart === -1 || jsonEnd === -1 || jsonEnd <= jsonStart) {
            throw new Error('No valid JSON structure found in response');
          }

          const jsonContent = responseContent.substring(jsonStart, jsonEnd + 1);
          console.log('Extracted JSON content length:', jsonContent.length);
          console.log('JSON content preview:', jsonContent.substring(0, 300) + '...');
          
          // Check for potential truncation indicators
          const lastChar = responseContent.trim().slice(-1);
          if (lastChar !== '}' && lastChar !== ']') {
            console.warn('Response may be truncated - does not end with } or ]');
            console.warn('Last 100 characters:', responseContent.slice(-100));
          }

          // Parse and validate the JSON structure
          let parsedData;
          try {
            parsedData = JSON.parse(jsonContent);
          } catch (parseError) {
            // If JSON parsing fails, try to repair common truncation issues
            console.log('Initial JSON parse failed, attempting repair...');
            
            let repairedJson = jsonContent;
            
            // Try to close incomplete objects/arrays
            const openBraces = (repairedJson.match(/\{/g) || []).length;
            const closeBraces = (repairedJson.match(/\}/g) || []).length;
            const openBrackets = (repairedJson.match(/\[/g) || []).length;
            const closeBrackets = (repairedJson.match(/\]/g) || []).length;
            
            // Add missing closing braces
            for (let i = 0; i < openBraces - closeBraces; i++) {
              repairedJson += '}';
            }
            
            // Add missing closing brackets
            for (let i = 0; i < openBrackets - closeBrackets; i++) {
              repairedJson += ']';
            }
            
            // Remove trailing commas that might cause issues
            repairedJson = repairedJson.replace(/,(\s*[}\]])/g, '$1');
            
            // Try to parse the repaired JSON
            try {
              parsedData = JSON.parse(repairedJson);
              console.log('Successfully repaired and parsed JSON');
            } catch (repairError) {
              console.error('JSON repair failed:', repairError);
              throw parseError; // Throw original error
            }
          }
          
          // Validate lesson plan structure
          if (!parsedData.lessonPlan) {
            throw new Error('Missing lessonPlan object in response');
          }
          
          if (!parsedData.lessonPlan.lessons || !Array.isArray(parsedData.lessonPlan.lessons)) {
            throw new Error('Missing or invalid lessons array in response');
          }
          
          if (parsedData.lessonPlan.lessons.length === 0) {
            throw new Error('Lessons array is empty');
          }

          // Validate each lesson has required fields
          for (let i = 0; i < parsedData.lessonPlan.lessons.length; i++) {
            const lesson = parsedData.lessonPlan.lessons[i];
            if (!lesson.title || typeof lesson.title !== 'string') {
              throw new Error(`Lesson ${i + 1} is missing a valid title`);
            }
            if (!lesson.description || typeof lesson.description !== 'string') {
              throw new Error(`Lesson ${i + 1} is missing a valid description`);
            }
            
            // Check for truncated lessons (common issue)
            if (lesson.title.length < 5 || lesson.description.length < 10) {
              throw new Error(`Lesson ${i + 1} appears to be truncated or incomplete`);
            }
            
            // Validate sections array
            if (!lesson.sections || !Array.isArray(lesson.sections)) {
              throw new Error(`Lesson ${i + 1} is missing a valid sections array`);
            }
            
            if (lesson.sections.length === 0) {
              throw new Error(`Lesson ${i + 1} has no sections`);
            }
            
            // Validate each section
            for (let j = 0; j < lesson.sections.length; j++) {
              const section = lesson.sections[j];
              if (!section.title || typeof section.title !== 'string') {
                throw new Error(`Lesson ${i + 1}, Section ${j + 1} is missing a valid title`);
              }
              if (!section.content || typeof section.content !== 'string') {
                throw new Error(`Lesson ${i + 1}, Section ${j + 1} is missing valid content`);
              }
              
              // Check for truncated sections
              if (section.title.length < 3) {
                throw new Error(`Lesson ${i + 1}, Section ${j + 1} title appears truncated`);
              }
              if (section.content.length < 50) {
                throw new Error(`Lesson ${i + 1}, Section ${j + 1} content appears truncated or too short`);
              }
            }
          }

          // Ensure we have reasonable metadata
          if (!parsedData.lessonPlan.totalEstimatedTime) {
            parsedData.lessonPlan.totalEstimatedTime = 'Not specified';
          }

          console.log('Validated lesson plan structure successfully');
          
          // Return the validated response with the original structure
          return c.json({
            ...aiData,
            result: {
              ...aiData.result,
              response: JSON.stringify(parsedData)
            }
          });

        } catch (validationError) {
          console.error('Lesson plan validation failed:', validationError);
          console.error('Raw response that failed validation:', responseContent);
          
          // Return a structured error that the frontend can handle
          return c.json({
            error: 'Invalid lesson plan format',
            details: validationError.message,
            rawResponse: responseContent.substring(0, 500), // First 500 chars for debugging
            validationFailed: true
          }, 422); // Unprocessable Entity
        }
      }

      // For other instruction types, return as-is
      return c.json(aiData);
    }

  } catch (error) {
    console.error('LLM endpoint error:', error);
    return c.json({ 
      error: 'Failed to process AI request', 
      details: error.message 
    }, 500);
  }
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
