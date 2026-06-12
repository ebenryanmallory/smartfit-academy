/**
 * Claude Opus API Route with Tool Use for Structured Responses
 * 
 * This route provides access to Claude Opus with Tool Use for guaranteed structured output.
 * Features:
 * - Authentication required (Clerk)
 * - Plan verification (premium plans only)
 * - Structured JSON responses via Anthropic Tool Use
 * - No manual validation needed (Tool Use guarantees schema compliance)
 * 
 * FRONTEND USAGE:
 * 
 * const response = await fetch('/claude/opus', {
 *   method: 'POST',
 *   headers: { 'Content-Type': 'application/json' },
 *   body: JSON.stringify({
 *     messages: [{ role: 'user', content: 'Generate a lesson plan about AI ethics' }],
 *     instructionType: 'lessonPlanGenerator',
 *     educationLevel: 'undergrad'
 *   })
 * })
 * 
 * const data = await response.json()
 * 
 * SUCCESS RESPONSE FORMAT:
 * {
 *   success: true,
 *   instructionType: 'lessonPlanGenerator',
 *   data: { lessonPlan: { ... } }, // Structured data matching the tool schema
 *   model: 'claude-opus-4-8',
 *   usage: { input_tokens: 123, output_tokens: 456 },
 *   toolName: 'generate_lesson_plan'
 * }
 * 
 * ERROR RESPONSE FORMATS:
 * - 401: Handled by Clerk middleware (authentication required)
 * - 403: { error: 'Premium feature required', message: '...', upgradeRequired: true }
 * - 400: { error: 'Invalid JSON body' } or { error: 'Missing or invalid messages array' }
 * - 500: { error: 'Claude service temporarily unavailable' } or other server errors
 * 
 * DATA STRUCTURE BY INSTRUCTION TYPE:
 * 
 * 1. educationalAssistant -> data: { response: string, topics_covered: string[], difficulty_level: string }
 * 2. lessonPlanGenerator -> data: { lessonPlan: { title, description, totalEstimatedTime, lessons: [...] } }
 * 3. lessonContentGenerator -> data: { content: { title, introduction, sections: [...], conclusion, keyTakeaways: [...] } }
 * 4. historicalConnectionGenerator -> data: { connectionSummary: { topic, modernContext, historicalPattern, keyInsight, connections: [...] } }
 */

import { Hono } from 'hono'
import { stream } from 'hono/streaming'
import { getAuth } from '@clerk/hono'
import {
  educationalAssistantInstructions,
  lessonPlanGeneratorInstructions,
  lessonContentGeneratorInstructions,
  historicalConnectionGeneratorInstructions,
  feedPostGeneratorInstructions,
  FEED_TOPICS
} from '../instructions/index'
import type { AppContext } from './types'

const claudeRoutes = new Hono<AppContext>()

// Tool schema definitions for structured output
const educationalAssistantTool = {
  name: "provide_educational_response",
  description: "Provide an educational response to the user's question",
  input_schema: {
    type: "object",
    properties: {
      response: {
        type: "string",
        description: "The educational response to the user's question"
      },
      topics_covered: {
        type: "array",
        items: { type: "string" },
        description: "List of main topics covered in the response"
      },
      difficulty_level: {
        type: "string",
        enum: ["beginner", "intermediate", "advanced"],
        description: "The difficulty level of the response"
      }
    },
    required: ["response", "topics_covered", "difficulty_level"]
  }
}

const lessonPlanTool = {
  name: "generate_lesson_plan",
  description: "Generate a comprehensive lesson plan",
  input_schema: {
    type: "object",
    properties: {
      lessonPlan: {
        type: "object",
        properties: {
          title: {
            type: "string",
            description: "The title of the lesson plan"
          },
          description: {
            type: "string",
            description: "A brief description of the lesson plan"
          },
          totalEstimatedTime: {
            type: "string",
            description: "Total estimated time for the lesson plan"
          },
          lessons: {
            type: "array",
            items: {
              type: "object",
              properties: {
                title: {
                  type: "string",
                  description: "The title of the individual lesson"
                },
                description: {
                  type: "string",
                  description: "Detailed description of the lesson content"
                },
                estimatedTime: {
                  type: "string",
                  description: "Estimated time for this lesson"
                },
                objectives: {
                  type: "array",
                  items: { type: "string" },
                  description: "Learning objectives for this lesson"
                }
              },
              required: ["title", "description", "estimatedTime", "objectives"]
            },
            description: "Array of individual lessons in the plan"
          }
        },
        required: ["title", "description", "totalEstimatedTime", "lessons"]
      }
    },
    required: ["lessonPlan"]
  }
}

const lessonContentTool = {
  name: "generate_lesson_content",
  description: "Generate detailed lesson content",
  input_schema: {
    type: "object",
    properties: {
      content: {
        type: "object",
        properties: {
          title: {
            type: "string",
            description: "The lesson title"
          },
          introduction: {
            type: "string",
            description: "Introduction to the lesson"
          },
          sections: {
            type: "array",
            items: {
              type: "object",
              properties: {
                heading: { type: "string" },
                content: { type: "string" },
                examples: {
                  type: "array",
                  items: { type: "string" }
                }
              },
              required: ["heading", "content"]
            },
            description: "Main content sections of the lesson"
          },
          conclusion: {
            type: "string",
            description: "Lesson conclusion and summary"
          },
          keyTakeaways: {
            type: "array",
            items: { type: "string" },
            description: "Key takeaways from the lesson"
          }
        },
        required: ["title", "introduction", "sections", "conclusion", "keyTakeaways"]
      }
    },
    required: ["content"]
  }
}

const historicalConnectionTool = {
  name: "generate_historical_connections",
  description: "Generate historical connections to modern topics",
  input_schema: {
    type: "object",
    properties: {
      connectionSummary: {
        type: "object",
        properties: {
          topic: {
            type: "string",
            description: "The main topic being analyzed"
          },
          modernContext: {
            type: "string",
            description: "Description of the modern context"
          },
          historicalPattern: {
            type: "string",
            description: "Description of the historical pattern"
          },
          keyInsight: {
            type: "string",
            description: "The key insight connecting past and present"
          },
          overallTheme: {
            type: "string",
            description: "The overall theme connecting all historical periods"
          },
          connections: {
            type: "array",
            items: {
              type: "object",
              properties: {
                era: { type: "string", description: "Historical era" },
                year: { type: "string", description: "Specific year or time period" },
                event: { type: "string", description: "Historical event or development" },
                thinker: { type: "string", description: "Key historical figure (optional)" },
                connection: { type: "string", description: "How this connects to the modern topic" },
                relevance: { type: "string", description: "Why this connection is relevant today" }
              },
              required: ["era", "year", "event", "connection", "relevance"]
            },
            description: "Array of historical connections",
            minItems: 3,
            maxItems: 4
          }
        },
        required: ["topic", "modernContext", "historicalPattern", "keyInsight", "connections"]
      }
    },
    required: ["connectionSummary"]
  }
}

const FEED_POST_TYPES = ['did_you_know', 'quiz', 'concept', 'code_snippet']
const FEED_DIFFICULTIES = ['intro', 'core', 'stretch']
const FEED_TOPIC_IDS = FEED_TOPICS.map(t => t.id)
const FEED_BATCH_SIZE = 5
// Free users get a tight hourly cap on generation batches; paid users are uncapped.
// Each batch is a full Opus call, and infinite scroll triggers them automatically.
const FEED_FREE_BATCHES_PER_HOUR = 3

const feedPostsTool = {
  name: "generate_feed_posts",
  description: "Generate a batch of short educational feed posts",
  input_schema: {
    type: "object",
    properties: {
      posts: {
        type: "array",
        minItems: FEED_BATCH_SIZE,
        maxItems: FEED_BATCH_SIZE,
        items: {
          type: "object",
          properties: {
            type: {
              type: "string",
              enum: FEED_POST_TYPES,
              description: "The post type"
            },
            topic: {
              type: "string",
              enum: FEED_TOPIC_IDS,
              description: "The lesson topic this post ties to"
            },
            title: {
              type: "string",
              description: "Specific hook, max 80 characters"
            },
            body: {
              type: "string",
              description: "Markdown body, length per post-type rules, no headings"
            },
            tags: {
              type: "array",
              items: { type: "string" },
              description: "2-4 lowercase concept tags"
            },
            difficulty: {
              type: "string",
              enum: FEED_DIFFICULTIES,
              description: "Difficulty relative to the target audience"
            },
            code: {
              type: "object",
              properties: {
                language: { type: "string", enum: ["python"] },
                snippet: { type: "string", description: "Valid runnable Python, max 15 lines" },
                takeaway: { type: "string", description: "One-sentence takeaway" }
              },
              required: ["language", "snippet", "takeaway"],
              description: "Required when type is code_snippet"
            },
            quiz: {
              type: "object",
              properties: {
                question: { type: "string" },
                options: {
                  type: "array",
                  items: { type: "string" },
                  minItems: 4,
                  maxItems: 4
                },
                correctIndex: { type: "integer", description: "Index 0-3 of the correct option" },
                explanation: { type: "string", description: "Why the answer is right, 1-3 sentences" }
              },
              required: ["question", "options", "correctIndex", "explanation"],
              description: "Required when type is quiz"
            }
          },
          required: ["type", "topic", "title", "body", "tags", "difficulty"]
        }
      }
    },
    required: ["posts"]
  }
}

interface FeedInteractionRow {
  post_type: string
  topic: string
  tags: string | null
  action: string
  difficulty: string | null
}

interface FeedPersonalizationSummary {
  avoidTags: string[]
  preferredTypes: string[]
  preferredTags: string[]
  quizHint: string | null
}

function summarizeInteractions(rows: FeedInteractionRow[], sessionExclude: string[]): FeedPersonalizationSummary {
  const avoid = new Set<string>(sessionExclude)
  const typeCounts = new Map<string, number>()
  const tagCounts = new Map<string, number>()
  let quizCorrect = 0
  let quizTotal = 0

  for (const row of rows) {
    let tags: string[] = []
    try {
      const parsed = JSON.parse(row.tags || '[]')
      if (Array.isArray(parsed)) tags = parsed.filter(t => typeof t === 'string')
    } catch { /* malformed tags row — skip */ }

    if (row.action === 'viewed') {
      tags.forEach(t => avoid.add(t))
    } else if (row.action === 'liked' || row.action === 'more_like_this') {
      typeCounts.set(row.post_type, (typeCounts.get(row.post_type) || 0) + 1)
      tags.forEach(t => tagCounts.set(t, (tagCounts.get(t) || 0) + 1))
    } else if (row.action === 'quiz_correct') {
      quizCorrect++
      quizTotal++
    } else if (row.action === 'quiz_incorrect') {
      quizTotal++
    }
  }

  let quizHint: string | null = null
  if (quizTotal >= 4) {
    const ratio = quizCorrect / quizTotal
    if (ratio > 0.8) quizHint = `${quizCorrect}/${quizTotal} correct — lean slightly harder, within the audience level`
    else if (ratio < 0.4) quizHint = `${quizCorrect}/${quizTotal} correct — lean slightly gentler, reinforce fundamentals`
  }

  const topByCount = (m: Map<string, number>, limit: number) =>
    [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, limit).map(([k]) => k)

  return {
    avoidTags: [...avoid].slice(0, 30),
    preferredTypes: topByCount(typeCounts, 2),
    preferredTags: topByCount(tagCounts, 10),
    quizHint
  }
}

function buildFeedUserMessage(summary: FeedPersonalizationSummary): string {
  const lines = [`Generate the next batch of ${FEED_BATCH_SIZE} feed posts.`]
  if (summary.avoidTags.length > 0) {
    lines.push(`AVOID these recently-seen concept tags (pick different concepts, or a genuinely new angle): ${summary.avoidTags.join(', ')}`)
  }
  if (summary.preferredTypes.length > 0 || summary.preferredTags.length > 0) {
    const parts = []
    if (summary.preferredTypes.length > 0) parts.push(`post types: ${summary.preferredTypes.join(', ')}`)
    if (summary.preferredTags.length > 0) parts.push(`tags: ${summary.preferredTags.join(', ')}`)
    lines.push(`PREFERRED ${parts.join('. PREFERRED ')}`)
  }
  if (summary.quizHint) {
    lines.push(`Quiz performance: ${summary.quizHint}`)
  }
  return lines.join('\n')
}

function validateFeedPost(post: any): boolean {
  if (!post || typeof post !== 'object') return false
  if (!FEED_POST_TYPES.includes(post.type)) return false
  if (!FEED_TOPIC_IDS.includes(post.topic)) return false
  if (typeof post.title !== 'string' || typeof post.body !== 'string') return false
  if (!Array.isArray(post.tags)) return false
  if (!FEED_DIFFICULTIES.includes(post.difficulty)) return false
  if (post.type === 'quiz') {
    const q = post.quiz
    if (!q || typeof q.question !== 'string' || typeof q.explanation !== 'string') return false
    if (!Array.isArray(q.options) || q.options.length !== 4) return false
    if (!Number.isInteger(q.correctIndex) || q.correctIndex < 0 || q.correctIndex > 3) return false
  }
  if (post.type === 'code_snippet') {
    const code = post.code
    if (!code || typeof code.snippet !== 'string' || typeof code.takeaway !== 'string') return false
  }
  return true
}

// Incrementally scans streamed tool-input JSON ({"posts": [{...}, ...]}) and
// invokes onPost with each post object as soon as its closing brace arrives.
// Depth 1 is the outer object; each post object lives at depth 2. Braces inside
// strings (e.g. code snippets) are ignored via string/escape tracking.
function createFeedPostExtractor(onPost: (post: unknown) => void): (fragment: string) => void {
  let buf = ''
  let pos = 0 // next unscanned index — characters before pos are never re-scanned
  let depth = 0
  let inString = false
  let escaped = false
  let objStart = -1
  return (fragment: string) => {
    buf += fragment
    while (pos < buf.length) {
      const ch = buf[pos]
      if (inString) {
        if (escaped) escaped = false
        else if (ch === '\\') escaped = true
        else if (ch === '"') inString = false
      } else if (ch === '"') {
        inString = true
      } else if (ch === '{') {
        depth++
        if (depth === 2 && objStart === -1) objStart = pos
      } else if (ch === '}') {
        depth--
        if (depth === 1 && objStart !== -1) {
          try {
            onPost(JSON.parse(buf.slice(objStart, pos + 1)))
          } catch { /* malformed object — skip it */ }
          buf = buf.slice(pos + 1)
          pos = 0
          objStart = -1
          continue
        }
      }
      pos++
    }
    // Drop consumed text; keep only an in-progress post object, if any.
    if (objStart === -1) {
      buf = ''
      pos = 0
    } else if (objStart > 0) {
      buf = buf.slice(objStart)
      pos -= objStart
      objStart = 0
    }
  }
}

// Returns true if the authenticated user is on the "monthly" (paid) plan.
function isMonthlyPlan(c: any): boolean {
  const auth = getAuth(c)
  if (!auth?.userId) return false
  return auth.has?.({ plan: 'monthly' }) ?? false
}

// Helper function to get appropriate tool based on instruction type
function getToolForInstructionType(instructionType: string) {
  switch (instructionType) {
    case 'educationalAssistant':
      return educationalAssistantTool
    case 'lessonPlanGenerator':
    case 'relevanceEngine':
      return lessonPlanTool
    case 'lessonContentGenerator':
      return lessonContentTool
    case 'historicalConnectionGenerator':
      return historicalConnectionTool
    default:
      return educationalAssistantTool
  }
}

// Claude Opus endpoint with plan verification (auth handled by middleware)
claudeRoutes.post('/opus', async (c) => {
  console.log('Claude Opus endpoint called')

  try {
    if (!isMonthlyPlan(c)) {
      return c.json({
        error: 'Premium feature required',
        message: 'Claude Opus requires a monthly plan.',
        upgradeRequired: true
      }, 403)
    }

    // Get Claude API key from environment
    const CLAUDE_API_KEY = c.env.CLAUDE_API_KEY
    if (!CLAUDE_API_KEY) {
      console.error('Claude API key not configured')
      return c.json({ error: 'Claude service temporarily unavailable' }, 500)
    }

    // Parse request body
    let body
    try {
      body = await c.req.json()
    } catch (e) {
      return c.json({ error: 'Invalid JSON body' }, 400)
    }

    const { messages, instructionType, educationLevel } = body
    if (!messages || !Array.isArray(messages)) {
      return c.json({ error: 'Missing or invalid messages array' }, 400)
    }

    // Prepare system instruction based on type
    let systemInstruction = educationalAssistantInstructions // default
    
    if (instructionType) {
      switch (instructionType) {
        case 'educationalAssistant':
          systemInstruction = educationalAssistantInstructions
          break
        case 'lessonPlanGenerator':
          const validEducationLevelsForPlan = ['elementary', 'highschool', 'undergrad', 'grad']
          const targetEducationLevelForPlan = validEducationLevelsForPlan.includes(educationLevel) ? educationLevel : 'undergrad'
          systemInstruction = lessonPlanGeneratorInstructions(targetEducationLevelForPlan)
          break
        case 'lessonContentGenerator':
          const validEducationLevels = ['elementary', 'highschool', 'undergrad', 'grad']
          const targetEducationLevel = validEducationLevels.includes(educationLevel) ? educationLevel : 'undergrad'
          systemInstruction = lessonContentGeneratorInstructions(targetEducationLevel)
          break
        case 'relevanceEngine':
          const validEducationLevelsForRelevance = ['elementary', 'highschool', 'undergrad', 'grad']
          const targetEducationLevelForRelevance = validEducationLevelsForRelevance.includes(educationLevel) ? educationLevel : 'undergrad'
          systemInstruction = lessonPlanGeneratorInstructions(targetEducationLevelForRelevance)
          break
        case 'historicalConnectionGenerator':
          const validEducationLevelsForHistorical = ['elementary', 'highschool', 'undergrad', 'grad']
          const targetEducationLevelForHistorical = validEducationLevelsForHistorical.includes(educationLevel) ? educationLevel : 'undergrad'
          systemInstruction = historicalConnectionGeneratorInstructions(targetEducationLevelForHistorical)
          break
        default:
          console.warn(`Unknown instruction type: ${instructionType}, using default`)
      }
    }

    // Get appropriate tool for the instruction type
    const tool = getToolForInstructionType(instructionType || 'educationalAssistant')

    // Prepare Claude API request
    const claudeRequest = {
      model: "claude-opus-4-8",
      max_tokens: 4096,
      // No temperature — sampling params are rejected (400) on Opus 4.7+
      system: systemInstruction,
      messages: messages,
      tools: [tool],
      tool_choice: { type: "tool", name: tool.name }
    }

    console.log(`Making Claude API request with tool: ${tool.name}`)

    // Call Claude API
    const claudeResponse = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': CLAUDE_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify(claudeRequest)
    })

    if (!claudeResponse.ok) {
      const errorText = await claudeResponse.text()
      console.error('Claude API error:', claudeResponse.status, errorText)
      return c.json({ 
        error: `Claude API error: ${claudeResponse.status}`,
        details: 'Check server logs for details'
      }, 500)
    }

    const claudeData = await claudeResponse.json()
    console.log('Claude API response received')

    // Extract tool use result
    const toolUseContent = claudeData.content?.find((item: any) => item.type === 'tool_use')
    
    if (!toolUseContent) {
      console.error('No tool use found in Claude response')
      return c.json({ 
        error: 'Invalid response format from Claude',
        details: 'Expected tool use but got different response type'
      }, 500)
    }

    // Return structured response that matches the expected format
    return c.json({
      success: true,
      instructionType,
      data: toolUseContent.input,
      model: 'claude-opus-4-8',
      usage: claudeData.usage,
      toolName: toolUseContent.name
    })

  } catch (error) {
    console.error('Claude Opus endpoint error:', error)
    return c.json({ 
      error: 'Failed to process Claude request', 
      details: error instanceof Error ? error.message : 'Unknown error'
    }, 500)
  }
})

// Feed post generation endpoint — available to ALL signed-in users.
// Free users are rate-limited per hour; paid users are uncapped.
// The prompt is fully server-authored; clients may only send excludeTags.
claudeRoutes.post('/feed', async (c) => {
  try {
    const auth = getAuth(c)
    if (!auth?.userId) {
      return c.json({ error: 'Unauthorized' }, 401)
    }

    const CLAUDE_API_KEY = c.env.CLAUDE_API_KEY
    if (!CLAUDE_API_KEY) {
      console.error('Claude API key not configured')
      return c.json({ error: 'Claude service temporarily unavailable' }, 500)
    }

    if (!isMonthlyPlan(c)) {
      const batchCount = await c.env.DB
        .prepare("SELECT COUNT(*) AS n FROM feed_interactions WHERE user_id = ? AND action = 'generated' AND created_at >= datetime('now', '-1 hour')")
        .bind(auth.userId)
        .first<{ n: number }>()
      if ((batchCount?.n ?? 0) >= FEED_FREE_BATCHES_PER_HOUR) {
        return c.json({
          error: 'rate_limited',
          message: 'You have reached your hourly feed limit on the free plan.',
          upgradeRequired: true
        }, 429)
      }
    }

    // Empty body is fine — excludeTags is the only accepted input
    let body: { excludeTags?: unknown } = {}
    try {
      body = await c.req.json()
    } catch { /* no body */ }
    const sessionExclude = Array.isArray(body.excludeTags)
      ? body.excludeTags.slice(0, 60).filter((t): t is string => typeof t === 'string')
      : []

    const db = c.env.DB

    const userRow = await db
      .prepare('SELECT education_level FROM users WHERE id = ?')
      .bind(auth.userId)
      .first<{ education_level: string | null }>()
    const validLevels = ['elementary', 'highschool', 'undergrad', 'grad']
    const educationLevel = userRow?.education_level && validLevels.includes(userRow.education_level)
      ? userRow.education_level
      : 'undergrad'

    // Recent interaction metadata (~a few sessions worth), aggregated server-side
    // into a compact personalization summary — raw history never reaches the model.
    // 'generated' rows are server-side rate-limit bookkeeping, not signals.
    const interactionResult = await db
      .prepare("SELECT post_type, topic, tags, action, difficulty FROM feed_interactions WHERE user_id = ? AND action != 'generated' ORDER BY created_at DESC LIMIT 40")
      .bind(auth.userId)
      .all()
    const rows = (interactionResult.results || []) as unknown as FeedInteractionRow[]

    const summary = summarizeInteractions(rows, sessionExclude)

    const claudeRequest = {
      model: 'claude-opus-4-8',
      // 5 posts with quizzes/code can exceed 4k tokens; headroom avoids truncated tool JSON
      max_tokens: 8000,
      // Streamed so each post can be forwarded the moment its JSON completes
      stream: true,
      // No temperature (rejected on Opus 4.7+) and no thinking (incompatible with forced tool_choice)
      system: [
        {
          type: 'text',
          text: feedPostGeneratorInstructions(educationLevel),
          cache_control: { type: 'ephemeral' }
        }
      ],
      messages: [{ role: 'user', content: buildFeedUserMessage(summary) }],
      tools: [feedPostsTool],
      tool_choice: { type: 'tool', name: feedPostsTool.name }
    }

    const claudeResponse = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': CLAUDE_API_KEY,
        'anthropic-version': '2023-06-01',
        // Without this beta header the API buffers tool-input JSON for validation
        // and delivers ~all of it in the final ~80ms, defeating per-post streaming
        // (measured 2026-06: first post would otherwise wait the full 20-30s batch).
        // Tradeoff: streamed tool JSON is no longer guaranteed valid on truncation —
        // tolerated by createFeedPostExtractor's per-object try/catch.
        // If this causes problems (e.g. malformed batches), delete this header to
        // restore buffered-but-validated tool input; posts will then arrive all at once.
        'anthropic-beta': 'fine-grained-tool-streaming-2025-05-14'
      },
      body: JSON.stringify(claudeRequest)
    })

    if (!claudeResponse.ok || !claudeResponse.body) {
      const errorText = await claudeResponse.text()
      console.error('Claude API error (feed):', claudeResponse.status, errorText)
      return c.json({
        error: `Claude API error: ${claudeResponse.status}`,
        details: 'Check server logs for details'
      }, 500)
    }

    // NDJSON response: one {"post": ...} line per validated post, emitted as soon
    // as that post's JSON object completes in the model's tool-input stream.
    c.header('Content-Type', 'application/x-ndjson')
    return stream(c, async (out) => {
      // The extractor callback is sync, so it queues posts; the read loop flushes.
      const queue: object[] = []
      let parsed = 0
      const extractor = createFeedPostExtractor((post) => {
        parsed++
        if (validateFeedPost(post)) queue.push(post as object)
        else console.error('Feed post failed validation', JSON.stringify(post).slice(0, 200))
      })

      let emitted = 0
      try {
        const reader = claudeResponse.body!.getReader()
        const decoder = new TextDecoder()
        let sseBuf = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          sseBuf += decoder.decode(value, { stream: true })
          const lines = sseBuf.split('\n')
          sseBuf = lines.pop() ?? ''
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            let event: {
              type?: string
              delta?: { type?: string; partial_json?: string; stop_reason?: string }
              usage?: unknown
              error?: unknown
            }
            try {
              event = JSON.parse(line.slice(6))
            } catch { continue }
            if (event.type === 'content_block_delta' && event.delta?.type === 'input_json_delta') {
              extractor(event.delta.partial_json ?? '')
            } else if (event.type === 'message_delta' && event.delta?.stop_reason === 'max_tokens') {
              // Truncated tool input usually means malformed JSON for the last post(s);
              // the extractor salvages whatever parsed cleanly.
              console.warn('Feed generation hit max_tokens; batch may be partial. usage:', event.usage)
            } else if (event.type === 'error') {
              console.error('Claude stream error (feed):', JSON.stringify(event.error))
            }
          }
          while (queue.length > 0) {
            await out.write(JSON.stringify({ post: queue.shift() }) + '\n')
            emitted++
          }
        }
      } catch (err) {
        console.error('Feed stream read error:', err)
      }

      if (emitted === 0) {
        console.error(`Feed stream produced no valid posts (${parsed} parsed)`)
        await out.write(JSON.stringify({ error: 'Failed to generate valid feed posts' }) + '\n')
        return
      }

      console.log(`Feed batch streamed: ${emitted}/${parsed} valid posts`)

      // Rate-limit bookkeeping; one row per batch, excluded from personalization
      await db
        .prepare("INSERT INTO feed_interactions (user_id, post_type, topic, action) VALUES (?, 'batch', 'feed', 'generated')")
        .bind(auth.userId)
        .run()
    })
  } catch (error) {
    console.error('Claude feed endpoint error:', error)
    return c.json({
      error: 'Failed to generate feed posts',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, 500)
  }
})

export default claudeRoutes 