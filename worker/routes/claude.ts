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
 *   model: 'claude-3-opus-20240229',
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
import { getAuth } from '@hono/clerk-auth'
import { 
  educationalAssistantInstructions,
  lessonPlanGeneratorInstructions,
  lessonContentGeneratorInstructions,
  historicalConnectionGeneratorInstructions
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

// Helper function to check user's plan via Clerk user context
async function checkUserPlan(c: any): Promise<{ canUseClaudeOpus: boolean; planType: string }> {
  // Get authenticated user from middleware context
  const auth = getAuth(c)
  if (!auth?.userId) {
    throw new Error('User not authenticated')
  }

  try {
    // Access user data that's already available through Clerk middleware
    // The user object should be available in c.var.user from the middleware
    const user = c.var.user
    
    if (!user) {
      console.warn('User object not available from Clerk middleware')
      return { canUseClaudeOpus: false, planType: 'free' }
    }

    // Check for plan information in user metadata
    // This can be stored in public_metadata, private_metadata, or unsafe_metadata
    const publicMetadata = user.publicMetadata || {}
    const privateMetadata = user.privateMetadata || {}
    const unsafeMetadata = user.unsafeMetadata || {}
    
    // Look for plan information in metadata (adjust based on how you store plan data)
    const planType = 
      privateMetadata.subscription_plan || 
      privateMetadata.plan || 
      publicMetadata.plan || 
      unsafeMetadata.plan ||
      'free'

    // Define which plans can use Claude Opus (premium feature)
    const premiumPlans = ['premium', 'pro', 'enterprise', 'paid']
    const canUseClaudeOpus = premiumPlans.includes(planType.toLowerCase())
    
    console.log(`User ${auth.userId} has plan: ${planType}, can use Claude Opus: ${canUseClaudeOpus}`)
    
    return { canUseClaudeOpus, planType }
    
  } catch (error) {
    console.error('Error checking user plan from Clerk middleware:', error)
    // Default to free plan on error
    return { canUseClaudeOpus: false, planType: 'free' }
  }
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
    // Authentication is already handled by middleware
    // Check user's plan
    const { canUseClaudeOpus, planType } = await checkUserPlan(c)
    if (!canUseClaudeOpus) {
      return c.json({ 
        error: 'Premium feature required', 
        message: `Claude Opus requires a premium plan. Current plan: ${planType}`,
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
      model: "claude-3-opus-20240229",
      max_tokens: 4096,
      temperature: instructionType === 'historicalConnectionGenerator' ? 0.8 : 0.7,
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
      model: 'claude-3-opus-20240229',
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

export default claudeRoutes 