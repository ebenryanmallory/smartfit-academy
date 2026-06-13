// Shared topic-suggestion flow: free-text interest goes to an LLM, which
// returns vetted educational topics the user can pick from. Used by the
// dashboard chat assistant and the feed preferences dialog.

export interface ParsedTopicsResponse {
  cleanResponse: string;
  topics: string[];
  hasFormatError?: boolean;
}

export const parseTopicsFromResponse = (response: string): ParsedTopicsResponse => {
  // Robust topic parsing with validation and error detection:
  // 1. Primary: TOPICS: at start, END_TOPICS to end
  // 2. Fallback: Extract any "- " prefixed lines as topics
  // 3. Error detection: Check for proper format compliance
  const lines = response.split('\n');
  const topics: string[] = [];
  let cleanResponse = response;
  let hasFormatError = false;

  // Check if response starts with TOPICS:
  const hasTopicsMarker = lines.length > 0 && lines[0].trim() === 'TOPICS:';

  if (hasTopicsMarker) {
    let endTopicsIndex = -1;
    let foundTopics = false;

    // Find END_TOPICS marker and extract topics
    for (let i = 1; i < lines.length; i++) {
      if (lines[i].trim() === 'END_TOPICS') {
        endTopicsIndex = i;
        break;
      }
      // Extract topics (lines starting with "- ")
      const line = lines[i].trim();
      if (line.startsWith('- ') && line.length > 2) {
        topics.push(line.substring(2).trim());
        foundTopics = true;
      }
    }

    // Validate format compliance
    if (!foundTopics) {
      hasFormatError = true;
      cleanResponse = "I apologize, but there was a formatting issue with my response. I should have provided topic suggestions for you to explore.";
    } else if (endTopicsIndex === -1) {
      // Missing END_TOPICS marker - this is a format error but we can still extract topics
      hasFormatError = true;

      // Find where topics likely end (first non-topic line)
      let topicEndIndex = 1;
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line.startsWith('- ') && line !== '') {
          topicEndIndex = i;
          break;
        }
      }
      cleanResponse = lines.slice(topicEndIndex).join('\n').trim();

      if (!cleanResponse) {
        cleanResponse = "I found some topics for you to explore, but there was a formatting issue with the rest of my response.";
      }
    } else {
      // Perfect format - extract clean response
      cleanResponse = lines.slice(endTopicsIndex + 1).join('\n').trim();

      if (!cleanResponse) {
        cleanResponse = "Great! I've identified some topics for you to explore.";
      }
    }
  } else {
    // No TOPICS: marker found - this is a format error
    hasFormatError = true;

    // Fallback: try to extract topics from content
    const potentialTopics = lines
      .map(line => line.trim())
      .filter(line => line.startsWith('- ') && line.length > 2)
      .map(line => line.substring(2).trim())
      .slice(0, 6); // Limit to 6 topics max

    if (potentialTopics.length > 0) {
      topics.push(...potentialTopics);
      // Remove topic lines from clean response
      cleanResponse = lines
        .filter(line => !line.trim().startsWith('- ') || line.trim().length <= 2)
        .join('\n')
        .trim();

      if (!cleanResponse) {
        cleanResponse = "I found some topics in my response, but there was a formatting issue. Let me know if you'd like to explore any of these topics further!";
      }
    } else {
      // No topics found at all
      cleanResponse = "I apologize, but there was a formatting issue with my response. I should have provided specific topic suggestions for you to explore. Please try asking your question again.";
    }
  }

  return { cleanResponse, topics, hasFormatError };
};

export interface TopicSuggestionResult {
  content: string;
  topics: string[];
}

// Claude Opus only (structured topics_covered); educationalAssistant is open
// to all signed-in users. Throws on server error or empty response.
export async function fetchTopicSuggestions(
  messages: { role: string; content: string }[]
): Promise<TopicSuggestionResult> {
  const res = await fetch('/claude/opus', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, instructionType: 'educationalAssistant' }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.message || data.error);
  if (data.success && data.data?.response) {
    return {
      content: data.data.response,
      topics: (data.data.topics_covered || []).slice(0, 6),
    };
  }
  throw new Error('No response from assistant');
}
