import React, { useState, useEffect } from 'react';
import { useUser, useAuth } from '@clerk/clerk-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Loader2, Clock, BookOpen, User, Calendar, Sparkles } from 'lucide-react';

interface HistoricalConnection {
  era: string;
  year: string;
  event: string;
  thinker?: string;
  connection: string;
  relevance: string;
}

interface ConnectionSummary {
  topic: string;
  overallTheme: string;
  modernContext: string;
  historicalPattern: string;
  connections: HistoricalConnection[];
  keyInsight: string;
}

interface HistoricalConnectionSummaryProps {
  topic: string;
  onGenerateLesson: (lessonTopic: string) => void;
  className?: string;
  educationLevel?: 'elementary' | 'highschool' | 'undergrad' | 'grad';
}

const HistoricalConnectionSummary: React.FC<HistoricalConnectionSummaryProps> = ({
  topic,
  onGenerateLesson,
  className = '',
  educationLevel = 'undergrad'
}) => {
  const { user } = useUser();
  const { getToken, has } = useAuth();
  const [summary, setSummary] = useState<ConnectionSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (topic.trim() && topic.trim().length > 3) {
      generateConnectionSummary();
      
      // Set a timeout to prevent infinite loading states
      const timeout = setTimeout(() => {
        if (loading) {
          console.warn('Connection summary generation timed out, hiding component');
          setLoading(false);
          setSummary(null);
          setError(null);
        }
      }, 15000); // 15 second timeout
      
      return () => clearTimeout(timeout);
    } else {
      setSummary(null);
      setError(null);
      setLoading(false);
    }
  }, [topic]); // eslint-disable-line react-hooks/exhaustive-deps

  const generateConnectionSummary = async () => {
    if (!topic.trim() || topic.trim().length < 4) {
      setSummary(null);
      setError(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    setSummary(null);

    try {
      // Handle authentication similar to the modal
      let token = null;
      if (user) {
        token = await getToken();
      }

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      // Check if user has monthly plan subscription
      const hasMonthlyPlan = has?.({ plan: 'monthly' }) ?? false;
      let response, data, responseContent;

      if (hasMonthlyPlan) {
        // Use Claude endpoint for monthly users
        response = await fetch('/claude/opus', {
          method: 'POST',
          headers,
          credentials: token ? 'include' : 'omit',
          body: JSON.stringify({
            messages: [
              {
                role: 'user',
                content: `Create a brief historical connection summary for the modern topic: "${topic}"`
              }
            ],
            instructionType: 'historicalConnectionGenerator',
            educationLevel: educationLevel
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to generate summary: ${response.status}`);
        }

        data = await response.json();
        // Claude returns structured data in data.connectionSummary format
        if (data.success && data.data?.connectionSummary) {
          responseContent = JSON.stringify({ connectionSummary: data.data.connectionSummary });
        } else {
          responseContent = '';
        }
      } else {
        // Use llama3 endpoint for non-monthly users
        response = await fetch('/llm/llama3', {
          method: 'POST',
          headers,
          credentials: token ? 'include' : 'omit',
          body: JSON.stringify({
            messages: [
              {
                role: 'user',
                content: `Create a brief historical connection summary for the modern topic: "${topic}"`
              }
            ],
            instructionType: 'historicalConnectionGenerator',
            educationLevel: educationLevel
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to generate summary: ${response.status}`);
        }

        data = await response.json();
        responseContent = data.result?.response || data.response || '';
      }
      
      if (!responseContent || typeof responseContent !== 'string') {
        console.warn('Invalid response content, not displaying component');
        setSummary(null);
        return;
      }
      
      // Parse the JSON response
      let parsedData;
      try {
        parsedData = JSON.parse(responseContent);
      } catch (parseError) {
        console.warn('Failed to parse JSON response, not displaying component:', parseError);
        setSummary(null);
        return;
      }
      if (parsedData.connectionSummary) {
        // Validate the summary data before setting it
        const connectionSummary = parsedData.connectionSummary;
        
        // Only set summary if we have all required fields and valid connections
        if (
          connectionSummary.topic &&
          connectionSummary.modernContext &&
          connectionSummary.historicalPattern &&
          connectionSummary.keyInsight &&
          connectionSummary.connections &&
          Array.isArray(connectionSummary.connections) &&
          connectionSummary.connections.length > 0 &&
          connectionSummary.connections.every((conn: any) => 
            conn.era && 
            conn.year && 
            conn.event && 
            conn.connection && 
            conn.relevance
          )
        ) {
          setSummary(connectionSummary);
        } else {
          console.warn('Incomplete connection summary data, not displaying component');
          setSummary(null);
        }
      } else {
        console.warn('No connectionSummary in response, not displaying component');
        setSummary(null);
      }

    } catch (err) {
      console.warn('Failed to generate connection summary, hiding component:', err);
      setSummary(null);
      setError(null);
    } finally {
      setLoading(false);
    }
  };

  // Generate a focused lesson topic for each connection
  const generateLessonTopic = (connection: HistoricalConnection, baseTopic: string) => {
    return `${baseTopic}: Learning from ${connection.event} (${connection.era})`;
  };

  // Don't render anything if topic is invalid or too short
  if (!topic.trim() || topic.trim().length < 4) {
    return null;
  }

  return (
    <div className={`w-full max-w-4xl mx-auto mt-6 ${className}`}>
      {loading && (
        <Card className="p-6 bg-gradient-to-r from-primary/5 to-accent/5 border-primary/20">
          <div className="flex items-center justify-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">
              Tracing <strong>"{topic}"</strong> through the corridors of time...
            </p>
          </div>
        </Card>
      )}

      {summary && 
       summary.connections && 
       Array.isArray(summary.connections) && 
       summary.connections.length > 0 && 
       summary.modernContext && 
       summary.historicalPattern && 
       summary.keyInsight && (
        <Card className="bg-gradient-to-br from-white via-primary/5 to-accent/5 border-primary/20 shadow-lg">
          <CardHeader className="pb-4">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="h-5 w-5 text-primary" />
              <CardTitle className="text-lg">Time Machine Connections</CardTitle>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">
                <strong>Modern Context:</strong> {summary.modernContext}
              </p>
              <p className="text-sm text-muted-foreground">
                <strong>Historical Pattern:</strong> {summary.historicalPattern}
              </p>
            </div>
          </CardHeader>
          
          <CardContent className="pt-0">
            {/* Timeline of Connections */}
            <div className="space-y-4 mb-6">
              {summary.connections.map((connection, index) => (
                <div key={index} className="relative">
                  {/* Timeline line */}
                  {index < summary.connections.length - 1 && (
                    <div className="absolute left-6 top-12 w-px h-8 bg-gradient-to-b from-primary/40 to-accent/40"></div>
                  )}
                  
                  <div className="flex gap-4">
                    {/* Timeline dot */}
                    <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center shadow-sm">
                      <span className="text-xs font-bold text-white">{index + 1}</span>
                    </div>
                    
                    {/* Content */}
                    <div className="flex-1 bg-white/80 rounded-lg p-4 border border-primary/10">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Calendar className="h-4 w-4 text-primary" />
                          <span className="text-sm font-semibold text-primary">{connection.era}</span>
                          <span className="text-xs text-muted-foreground">({connection.year})</span>
                        </div>
                        <Button
                          size="sm"
                          onClick={() => onGenerateLesson(generateLessonTopic(connection, topic))}
                          className="ml-2"
                        >
                          <Sparkles className="h-3 w-3 mr-1" />
                          Generate Lesson
                        </Button>
                      </div>
                      
                      <h4 className="font-medium text-foreground mb-1">{connection.event}</h4>
                      
                      {connection.thinker && (
                        <div className="flex items-center gap-1 mb-2">
                          <User className="h-3 w-3 text-accent" />
                          <span className="text-xs text-accent font-medium">{connection.thinker}</span>
                        </div>
                      )}
                      
                      <p className="text-sm text-muted-foreground mb-2">{connection.connection}</p>
                      <p className="text-xs text-primary/80 font-medium">
                        <strong>Why it matters:</strong> {connection.relevance}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Key Insight */}
            <div className="bg-gradient-to-r from-secondary/10 to-primary/10 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <BookOpen className="h-4 w-4 text-secondary mt-0.5 flex-shrink-0" />
                <div>
                  <h4 className="font-semibold text-secondary mb-1">Key Insight</h4>
                  <p className="text-sm text-foreground">{summary.keyInsight}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default HistoricalConnectionSummary; 