import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useUser, useAuth } from '@clerk/clerk-react';
import { toast } from 'sonner';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Bookmark, BookmarkCheck, ChevronDown, X } from 'lucide-react';

interface Message {
  role: string;
  content: string;
  topics?: string[];
  hasFormatError?: boolean;
}

interface BottomChatAssistantProps {
  onTopicSaved?: () => void;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
  onUserInput?: (userInput: string) => void;
}

export default function BottomChatAssistant({ onTopicSaved, isExpanded = false, onToggleExpanded, onUserInput }: BottomChatAssistantProps = {}) {
  const { isSignedIn } = useUser();
  const { getToken } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [savedTopics, setSavedTopics] = useState<Set<string>>(new Set());
  const [panelHeight, setPanelHeight] = useState(50); // Height as percentage of viewport
  const [isResizing, setIsResizing] = useState(false);
  const [collapsedHeight, setCollapsedHeight] = useState(120); // Track actual collapsed height
  const resizeRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  // Load user's existing saved topics on mount
  useEffect(() => {
    const loadSavedTopics = async () => {
      if (!isSignedIn) return;
      
      try {
        const token = await getToken();
        const response = await fetch('/api/d1/user/topics', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          credentials: 'include',
        });
        
        if (response.ok) {
          const data = await response.json();
          const topicNames = (data.topics || []).map((t: any) => t.topic);
          setSavedTopics(new Set(topicNames));
        }
      } catch (error) {
        // Error loading saved topics - handle silently
      }
    };

    loadSavedTopics();
  }, [isSignedIn, getToken]);

  // Track collapsed height for proper topic positioning
  useEffect(() => {
    if (!isExpanded && panelRef.current) {
      const height = panelRef.current.offsetHeight;
      if (height > 0) {
        setCollapsedHeight(height);
      }
    }
  }, [isExpanded, messages.length]);

  // Function to refresh saved topics (can be called after saving)
  const refreshSavedTopics = async () => {
    if (!isSignedIn) return;
    
    try {
      const token = await getToken();
      const response = await fetch('/api/d1/user/topics', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });
      
      if (response.ok) {
        const data = await response.json();
        const topicNames = (data.topics || []).map((t: any) => t.topic);
        setSavedTopics(new Set(topicNames));
      }
    } catch (error) {
      // Error refreshing saved topics - handle silently
    }
  };

  // Handle vertical resizing
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
    
    const handleMouseMove = (e: MouseEvent) => {
      const viewportHeight = window.innerHeight;
      const mouseY = e.clientY;
      const newHeight = Math.max(30, Math.min(80, ((viewportHeight - mouseY) / viewportHeight) * 100));
      setPanelHeight(newHeight);
    };
    
    const handleMouseUp = () => {
      setIsResizing(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, []);

  // Development test cases removed for production

  const parseTopicsFromResponse = (response: string): { cleanResponse: string; topics: string[]; hasFormatError?: boolean } => {
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

  const handleSaveTopic = async (topic: string) => {
    if (!isSignedIn) {
      toast.error("Sign in required", {
        description: "You need to sign in to save topics to your learning list. Sign in to track your interests and build your personalized curriculum.",
        action: {
          label: "Sign In",
          onClick: () => {
            // This will trigger the Clerk sign-in modal
            const signInButton = document.querySelector('[data-clerk-sign-in-button]') as HTMLElement;
            if (signInButton) {
              signInButton.click();
            } else {
              // Fallback: show a more detailed message
              toast.info("Click the Sign In button in the top navigation to get started!");
            }
          }
        },
        duration: 6000,
      });
      return;
    }

    // Check if already saved
    if (savedTopics.has(topic)) {
      toast.info("Topic already saved", {
        description: `"${topic}" is already in your learning topics.`,
        duration: 2000,
      });
      return;
    }

    try {
      const token = await getToken();
      const response = await fetch('/api/d1/user/topics', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ topic }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to save topic');
      }

      // Check if the topic was actually saved (response should contain the saved topic)
      if (data.topic) {
        // Refresh saved topics from server to ensure consistency
        await refreshSavedTopics();
        
        // Call callback to refresh parent components
        onTopicSaved?.();
        
        // Show success toast
        toast.success("Topic saved successfully!", {
          description: `"${topic}" has been added to your learning topics.`,
          duration: 2500,
        });
      } else {
        throw new Error('Topic was not saved properly');
      }
    } catch (error) {
      toast.error("Failed to save topic", {
        description: error instanceof Error ? error.message : "There was an error saving your topic. Please try again.",
        duration: 4000,
      });
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const userInput = input.trim(); // Store user input before clearing
    const newMessages = [...messages, { role: 'user', content: userInput }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);
    
    // Call the onUserInput callback with the user's input
    onUserInput?.(userInput);

    try {
      const res = await fetch('/llm/llama3', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          messages: newMessages.map(msg => ({ role: msg.role, content: msg.content })),
          useCustomInstructions: true 
        }),
      });
      const data = await res.json();
      if (data.error) {
        setMessages([...newMessages, { role: 'assistant', content: `Error: ${data.error}` }]);
      } else if (data.result?.response) {
        const { cleanResponse, topics, hasFormatError } = parseTopicsFromResponse(data.result.response);
        
        // Debug logging removed for production
        
        // Format errors handled silently in production
        
        setMessages([...newMessages, { 
          role: 'assistant', 
          content: cleanResponse || 'I\'d be happy to help you explore educational topics!',
          topics: topics,
          hasFormatError: hasFormatError
        }]);
      } else {
        setMessages([...newMessages, { role: 'assistant', content: 'Sorry, no response.' }]);
      }
    } catch (err) {
      setMessages([...newMessages, { role: 'assistant', content: 'Error contacting assistant.' }]);
    } finally {
      setLoading(false);
    }
  };

  const latestAssistantMessage = messages.filter(msg => msg.role === 'assistant').slice(-1)[0];

  return (
    <>
      {/* Topic Links - Above the chat panel */}
      {latestAssistantMessage?.topics && latestAssistantMessage.topics.length > 0 && (
        <div className="fixed bottom-0 left-0 right-0 z-40 bg-gradient-to-t from-blue-50 to-blue-100 border-t border-blue-200 shadow-sm"
             style={{ 
               bottom: isExpanded ? `${panelHeight}vh` : `${collapsedHeight}px`,
               transition: 'bottom 0.3s ease-in-out'
             }}>
          <div className="max-w-6xl mx-auto p-4">
            <h4 className="text-sm font-medium text-blue-900 mb-3">💡 Explore these topics:</h4>
            <div className="flex flex-wrap gap-2">
              {latestAssistantMessage.topics.map((topic, idx) => {
                const isSaved = savedTopics.has(topic);
                return (
                  <div 
                    key={idx} 
                    className="flex items-center border border-blue-300 rounded-full overflow-hidden hover:border-blue-400 transition-colors bg-white shadow-sm"
                  >
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleSaveTopic(topic)}
                      className="text-blue-700 hover:bg-blue-100 text-xs px-3 py-1 h-auto rounded-none border-none flex-1"
                    >
                      {topic}
                    </Button>
                    <div className="w-px bg-blue-300 h-6"></div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleSaveTopic(topic)}
                      className={`flex items-center gap-1 px-2 py-1 h-auto text-xs rounded-none border-none ${
                        isSaved 
                          ? 'text-green-600 hover:text-green-700 hover:bg-green-50' 
                          : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'
                      }`}
                      title={isSaved ? 'Topic saved to your list' : 'Add topic to your learning list'}
                    >
                      {isSaved ? (
                        <>
                          <BookmarkCheck className="h-3 w-3" />
                          <span className="sr-only">Added</span>
                        </>
                      ) : (
                        <>
                          <Bookmark className="h-3 w-3" />
                          <span className="sr-only">Add</span>
                        </>
                      )}
                    </Button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Main Chat Panel */}
      <div 
        ref={panelRef}
        className={`fixed bottom-0 left-0 right-0 bg-white border-t shadow-lg z-50 transition-all duration-300 ${
          isExpanded 
            ? 'border-t-4 border-t-primary' 
            : 'border-gray-200'
        }`}
        style={{ 
          height: isExpanded ? `${panelHeight}vh` : 'auto',
          minHeight: isExpanded ? '300px' : 'auto'
        }}>
        
        {/* Resize Handle */}
        {isExpanded && (
          <div
            ref={resizeRef}
            onMouseDown={handleMouseDown}
            className={`absolute top-0 left-0 right-0 h-1 bg-primary cursor-ns-resize hover:bg-primary/80 transition-colors ${
              isResizing ? 'bg-primary/80' : ''
            }`}
            title="Drag to resize"
          />
        )}

        <div className={`max-w-6xl mx-auto p-6 h-full ${isExpanded ? 'flex flex-col' : ''}`}>
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold text-center flex-1 text-foreground">
              What topics would you like to explore?
            </h3>
            {isExpanded && onToggleExpanded && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onToggleExpanded}
                className="text-gray-500 hover:text-gray-700 hover:bg-gray-100 p-2 h-auto ml-4"
                title="Collapse chat assistant"
              >
                <ChevronDown className="h-4 w-4" />
              </Button>
            )}
          </div>
          
          {/* Response Window */}
          {messages.length > 0 && (
            <Card className={`mb-4 overflow-y-auto bg-muted/20 ${
              isExpanded ? 'flex-1 min-h-0' : 'max-h-32'
            }`}>
              <div className="p-4 space-y-3">
                {messages.slice(-5).map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`px-4 py-3 rounded-lg leading-relaxed ${
                      msg.role === 'user' 
                        ? 'max-w-xs text-sm bg-primary text-primary-foreground' 
                        : msg.hasFormatError
                          ? 'xxl text-base bg-orange-100 text-orange-900 border border-orange-200'
                          : 'max-w-4xl text-base bg-secondary text-secondary-foreground'
                    }`}>
                      {msg.hasFormatError && (
                        <div className="text-xs text-orange-600 mb-2 font-medium">
                          ⚠️ Response formatting issue detected
                        </div>
                      )}
                      {msg.content}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex justify-start">
                    <div className="bg-secondary text-secondary-foreground px-4 py-3 rounded-lg text-base italic leading-relaxed">
                      Assistant is thinking...
                    </div>
                  </div>
                )}
              </div>
            </Card>
          )}

          {/* Input Section */}
          <div className="flex gap-3 items-center">
            <Input
              placeholder="Ask about any topic you'd like to learn more about..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleSend(); }}
              disabled={loading}
              className="flex-1 h-12 text-lg px-4"
            />
            <Button 
              onClick={handleSend} 
              disabled={loading || !input.trim()}
              className="h-12 px-6"
            >
              {loading ? 'Sending...' : 'Explore Topics'}
            </Button>
          </div>
        </div>
      </div>
    </>
  );
} 