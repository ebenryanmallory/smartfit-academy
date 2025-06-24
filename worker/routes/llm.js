import { Hono } from 'hono';
import { educationalAssistantInstructions, lessonPlanGeneratorInstructions, lessonContentGeneratorInstructions, historicalConnectionGeneratorInstructions } from '../instructions/index';
const llmRoutes = new Hono();
// Llama3 LLM endpoint
llmRoutes.post('/llama3', async (c) => {
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
    }
    catch (e) {
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
                case 'relevanceEngine':
                    // Use education level if provided, default to 'undergrad'
                    const validEducationLevelsForRelevance = ['elementary', 'highschool', 'undergrad', 'grad'];
                    const targetEducationLevelForRelevance = validEducationLevelsForRelevance.includes(educationLevel) ? educationLevel : 'undergrad';
                    instructionContent = lessonPlanGeneratorInstructions(targetEducationLevelForRelevance);
                    console.log(`Using lesson plan generator (via relevance engine route) for education level: ${targetEducationLevelForRelevance}`);
                    break;
                case 'historicalConnectionGenerator':
                    // Use education level if provided, default to 'undergrad'
                    const validEducationLevelsForHistorical = ['elementary', 'highschool', 'undergrad', 'grad'];
                    const targetEducationLevelForHistorical = validEducationLevelsForHistorical.includes(educationLevel) ? educationLevel : 'undergrad';
                    instructionContent = historicalConnectionGeneratorInstructions(targetEducationLevelForHistorical);
                    console.log(`Using historical connection generator for education level: ${targetEducationLevelForHistorical}`);
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
        }
        else {
            processedMessages.unshift(systemInstruction);
        }
    }
    try {
        // Call Cloudflare Workers AI REST API
        const aiUrl = `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct`;
        // Prepare request body with appropriate max_tokens based on instruction type
        const requestBody = { messages: processedMessages };
        // Set higher token limits for lesson plan generation to prevent truncation
        if (instructionType === 'lessonPlanGenerator' || instructionType === 'relevanceEngine') {
            requestBody.max_tokens = 4096; // Much higher limit for comprehensive lesson plans
            requestBody.temperature = 0.7; // Slightly higher creativity for educational content
        }
        else if (instructionType === 'lessonContentGenerator') {
            requestBody.max_tokens = 2048; // Higher limit for detailed lesson content
            requestBody.temperature = 0.7;
        }
        else if (instructionType === 'historicalConnectionGenerator') {
            requestBody.max_tokens = 3072; // High limit for detailed historical connections
            requestBody.temperature = 0.8; // Higher creativity for finding historical parallels
        }
        else {
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
        }
        else {
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
                return c.json({ error: aiData.error || 'AI request failed' }, aiRes.status);
            }
            // Extract the response content
            const responseContent = aiData.result?.response || aiData.response || '';
            console.log('Raw AI response length:', responseContent.length);
            console.log('Raw AI response preview:', responseContent.substring(0, 200) + '...');
            // Log token usage if available for debugging
            if (aiData.result?.usage) {
                console.log('Token usage:', aiData.result.usage);
            }
            // For lesson plan generation, relevance engine (now using lesson plan), and historical connection generator, validate the JSON structure
            if (instructionType === 'lessonPlanGenerator' || instructionType === 'relevanceEngine' || instructionType === 'historicalConnectionGenerator') {
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
                    }
                    catch (parseError) {
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
                        }
                        catch (repairError) {
                            console.error('JSON repair failed:', repairError);
                            throw parseError; // Throw original error
                        }
                    }
                    // Validate structure based on instruction type
                    if (instructionType === 'lessonPlanGenerator' || instructionType === 'relevanceEngine') {
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
                            if (lesson.title.length < 10) {
                                throw new Error(`Lesson ${i + 1} title appears to be truncated or too short (minimum 10 characters)`);
                            }
                            if (lesson.description.length < 50) {
                                throw new Error(`Lesson ${i + 1} description appears to be truncated or too short (minimum 50 characters)`);
                            }
                            // Check for proper sentence structure in description (should end with punctuation)
                            if (!/[.!?]$/.test(lesson.description.trim())) {
                                throw new Error(`Lesson ${i + 1} description appears incomplete (missing ending punctuation)`);
                            }
                            // Check for reasonable description length (shouldn't be excessively long either)
                            if (lesson.description.length > 2000) {
                                console.warn(`Lesson ${i + 1} description is very long (${lesson.description.length} characters)`);
                            }
                        }
                        // Ensure we have reasonable metadata
                        if (!parsedData.lessonPlan.totalEstimatedTime) {
                            parsedData.lessonPlan.totalEstimatedTime = 'Not specified';
                        }
                        console.log('Validated lesson plan structure successfully');
                    }
                    else if (instructionType === 'historicalConnectionGenerator') {
                        // Validate historical connection structure
                        if (!parsedData.connectionSummary) {
                            throw new Error('Missing connectionSummary object in response');
                        }
                        const summary = parsedData.connectionSummary;
                        // Validate required top-level fields
                        if (!summary.topic || typeof summary.topic !== 'string') {
                            throw new Error('Missing or invalid topic in connectionSummary');
                        }
                        if (!summary.modernContext || typeof summary.modernContext !== 'string') {
                            throw new Error('Missing or invalid modernContext in connectionSummary');
                        }
                        if (!summary.historicalPattern || typeof summary.historicalPattern !== 'string') {
                            throw new Error('Missing or invalid historicalPattern in connectionSummary');
                        }
                        if (!summary.keyInsight || typeof summary.keyInsight !== 'string') {
                            throw new Error('Missing or invalid keyInsight in connectionSummary');
                        }
                        // Validate content quality of required fields
                        if (summary.topic.length < 3) {
                            throw new Error('Topic appears truncated or too short');
                        }
                        if (summary.modernContext.length < 20) {
                            throw new Error('Modern context description appears truncated or too short');
                        }
                        if (summary.historicalPattern.length < 20) {
                            throw new Error('Historical pattern description appears truncated or too short');
                        }
                        if (summary.keyInsight.length < 20) {
                            throw new Error('Key insight description appears truncated or too short');
                        }
                        // Validate optional overallTheme if present
                        if (summary.overallTheme && (typeof summary.overallTheme !== 'string' || summary.overallTheme.length < 10)) {
                            throw new Error('Overall theme is present but appears invalid or truncated');
                        }
                        // Validate connections array
                        if (!summary.connections || !Array.isArray(summary.connections)) {
                            throw new Error('Missing or invalid connections array in connectionSummary');
                        }
                        if (summary.connections.length === 0) {
                            throw new Error('Connections array is empty');
                        }
                        // Validate connection count matches instruction requirements (3-4 connections)
                        if (summary.connections.length < 3) {
                            throw new Error(`Insufficient connections: expected 3-4 connections, got ${summary.connections.length}`);
                        }
                        if (summary.connections.length > 5) {
                            throw new Error(`Too many connections: expected 3-4 connections, got ${summary.connections.length}`);
                        }
                        // Validate each connection thoroughly
                        const eras = new Set();
                        let hasClassicalThinker = false;
                        for (let i = 0; i < summary.connections.length; i++) {
                            const connection = summary.connections[i];
                            const connectionIndex = i + 1;
                            // Validate required fields
                            if (!connection.era || typeof connection.era !== 'string') {
                                throw new Error(`Connection ${connectionIndex} is missing a valid era`);
                            }
                            if (!connection.year || typeof connection.year !== 'string') {
                                throw new Error(`Connection ${connectionIndex} is missing a valid year`);
                            }
                            if (!connection.event || typeof connection.event !== 'string') {
                                throw new Error(`Connection ${connectionIndex} is missing a valid event`);
                            }
                            if (!connection.connection || typeof connection.connection !== 'string') {
                                throw new Error(`Connection ${connectionIndex} is missing a valid connection description`);
                            }
                            if (!connection.relevance || typeof connection.relevance !== 'string') {
                                throw new Error(`Connection ${connectionIndex} is missing a valid relevance description`);
                            }
                            // Validate optional thinker field
                            if (connection.thinker && (typeof connection.thinker !== 'string' || connection.thinker.length < 2)) {
                                throw new Error(`Connection ${connectionIndex} has invalid thinker field`);
                            }
                            // Track if we have classical thinkers
                            if (connection.thinker) {
                                hasClassicalThinker = true;
                            }
                            // Check for truncated content with appropriate minimums
                            if (connection.era.length < 3) {
                                throw new Error(`Connection ${connectionIndex} era appears truncated`);
                            }
                            if (connection.year.length < 2) {
                                throw new Error(`Connection ${connectionIndex} year appears truncated`);
                            }
                            if (connection.event.length < 15) {
                                throw new Error(`Connection ${connectionIndex} event description appears truncated or too brief`);
                            }
                            if (connection.connection.length < 50) {
                                throw new Error(`Connection ${connectionIndex} connection description appears truncated or insufficient detail`);
                            }
                            if (connection.relevance.length < 30) {
                                throw new Error(`Connection ${connectionIndex} relevance description appears truncated or insufficient detail`);
                            }
                            // Check for proper sentence structure (should end with punctuation)
                            if (!/[.!?]$/.test(connection.connection.trim())) {
                                throw new Error(`Connection ${connectionIndex} connection description appears incomplete (missing ending punctuation)`);
                            }
                            if (!/[.!?]$/.test(connection.relevance.trim())) {
                                throw new Error(`Connection ${connectionIndex} relevance description appears incomplete (missing ending punctuation)`);
                            }
                            // Track eras for diversity check
                            eras.add(connection.era.toLowerCase());
                            // Validate year format (should contain numbers or time indicators)
                            if (!/\d|century|era|period/i.test(connection.year)) {
                                throw new Error(`Connection ${connectionIndex} year format appears invalid or incomplete`);
                            }
                        }
                        // Validate historical period diversity (should span different eras)
                        if (eras.size < Math.min(3, summary.connections.length)) {
                            console.warn('Limited historical period diversity in connections, but continuing validation');
                        }
                        // Log if classical thinker requirement is met
                        if (!hasClassicalThinker && summary.connections.length >= 3) {
                            console.warn('No classical thinkers included in connections, but this is optional');
                        }
                        // Validate against common AI response artifacts
                        const fullResponse = JSON.stringify(parsedData);
                        if (fullResponse.includes('...') || fullResponse.includes('[truncated]') || fullResponse.includes('etc.')) {
                            throw new Error('Response contains truncation indicators or incomplete content');
                        }
                        // Check for reasonable total content length
                        const totalContentLength = summary.modernContext.length + summary.historicalPattern.length +
                            summary.keyInsight.length + summary.connections.reduce((sum, conn) => sum + conn.connection.length + conn.relevance.length, 0);
                        if (totalContentLength < 300) {
                            throw new Error('Response content appears too brief overall, possibly truncated');
                        }
                        console.log('Validated historical connection structure successfully');
                        console.log(`✓ ${summary.connections.length} connections spanning ${eras.size} different eras`);
                        console.log(`✓ Total content length: ${totalContentLength} characters`);
                        console.log(`✓ Classical thinker included: ${hasClassicalThinker}`);
                    }
                    else if (instructionType === 'lessonContentGenerator') {
                        // Validate lesson content structure - basic validation for now
                        if (typeof parsedData !== 'object' || parsedData === null) {
                            throw new Error('Invalid lesson content response: not a valid object');
                        }
                        console.log('Validated lesson content generator response structure');
                        console.log('Response keys:', Object.keys(parsedData));
                    }
                    else {
                        // For other types, perform basic validation
                        console.log(`Basic validation for instruction type: ${instructionType}`);
                        console.log('Response keys:', Object.keys(parsedData));
                    }
                    // Return the validated response with the original structure
                    return c.json({
                        ...aiData,
                        result: {
                            ...aiData.result,
                            response: JSON.stringify(parsedData)
                        }
                    });
                }
                catch (validationError) {
                    console.error(`${instructionType} validation failed:`, validationError);
                    console.error('Raw response that failed validation:', responseContent);
                    // Return a structured error that the frontend can handle
                    return c.json({
                        error: `Invalid ${instructionType} format`,
                        details: validationError.message,
                        rawResponse: responseContent.substring(0, 500), // First 500 chars for debugging
                        validationFailed: true
                    }, 422); // Unprocessable Entity
                }
            }
            // For other instruction types, return as-is
            return c.json(aiData);
        }
    }
    catch (error) {
        console.error('LLM endpoint error:', error);
        return c.json({
            error: 'Failed to process AI request',
            details: error.message
        }, 500);
    }
});
export default llmRoutes;
