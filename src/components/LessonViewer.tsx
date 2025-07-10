import ReactMarkdown from 'react-markdown';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from '@/components/ui/badge';
import { useUser, useAuth } from "@clerk/clerk-react";
import { CodeSnippet } from './CodeSnippet';
import { CodePlaygroundTabs } from './CodePlaygroundTabs';
import { SaveProgressPrompt } from './SaveProgressPrompt';
import { LessonQuizModal } from './LessonQuizModal';
import { ChevronLeft, ChevronRight, FolderOpen, CheckCircle, Circle, Brain } from 'lucide-react';
import { useEffect, useState } from 'react';
import { toast } from 'sonner';


interface NavigationInfo {
  currentIndex: number;
  totalLessons: number;
  previousLesson?: { uuid: string; title: string };
  nextLesson?: { uuid: string; title: string };
}

interface LessonViewerProps {
  title: string;
  description: string;
  content: string;
  topic?: string;
  metaTopic?: string; // Optional meta topic for hierarchical organization
  navigationInfo?: NavigationInfo | null;
  onNavigate?: (lessonUuid: string) => void;
  lessonId?: string; // Add lesson ID prop for progress tracking
}

// Python to JavaScript code conversion helper
function convertPythonToJS(pythonCode: string): string {
  // Basic conversions
  let jsCode = pythonCode
    .replace(/print\((.*?)\)/g, 'console.log($1)') // print() -> console.log()
    .replace(/def /g, 'function ') // def -> function
    .replace(/:$/gm, ' {') // : -> {
    .replace(/^(\s*)(?=\S)/gm, '$1  ') // Indentation
    .replace(/\n/g, '\n  ') // Add indentation
    .replace(/\n\s*\n/g, '\n}\n\n') // Add closing braces
    .replace(/True/g, 'true') // True -> true
    .replace(/False/g, 'false') // False -> false
    .replace(/None/g, 'null') // None -> null
    .replace(/f"(.*?)"/g, '`$1`') // f-strings -> template literals
    .replace(/\{([^}]+)\}/g, '${$1}') // {var} -> ${var}
    .replace(/#/g, '//'); // Comments

  // Add closing brace for the last function
  if (jsCode.includes('function ')) {
    jsCode += '\n}';
  }

  return jsCode;
}

export function LessonViewer({ title, description, content, topic, metaTopic, navigationInfo, onNavigate, lessonId }: LessonViewerProps) {
  const { isSignedIn, user } = useUser();
  const { getToken } = useAuth();
  const [isCompleted, setIsCompleted] = useState(false);
  const [isMarkingComplete, setIsMarkingComplete] = useState(false);
  const [progressLoading, setProgressLoading] = useState(false);
  const [showQuizModal, setShowQuizModal] = useState(false);
  const [quizScore, setQuizScore] = useState<number | null>(null);

  // Check lesson completion status
  useEffect(() => {
    const checkProgress = async () => {
      if (!isSignedIn || !lessonId) return;

      setProgressLoading(true);
      try {
        const token = await getToken();
        if (!token) return;

        const response = await fetch(`/api/d1/user/progress?lessonId=${encodeURIComponent(lessonId)}`, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          credentials: 'include',
        });

        if (response.ok) {
          const data = await response.json();
          const progress = data.progress;
          setIsCompleted(progress?.completed === 1);
          setQuizScore(progress?.score || null);
        }
      } catch (error) {
        console.error('Error checking lesson progress:', error);
      } finally {
        setProgressLoading(false);
      }
    };

    checkProgress();
  }, [isSignedIn, lessonId, getToken]);

  // Mark lesson as complete
  const markAsComplete = async () => {
    if (!isSignedIn || !lessonId) {
      toast.error('Please sign in to track your progress');
      return;
    }

    setIsMarkingComplete(true);
    try {
      const token = await getToken();
      if (!token) {
        throw new Error('Failed to get authentication token');
      }

      const response = await fetch('/api/d1/user/progress', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          lessonId: lessonId,
          completed: true,
          additionalData: {
            completedAt: new Date().toISOString(),
            title: title
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to mark lesson as complete: ${response.status}`);
      }

      setIsCompleted(true);
      toast.success('Lesson marked as complete!', {
        description: 'Great job! Your progress has been saved.',
      });
    } catch (error) {
      console.error('Error marking lesson as complete:', error);
      toast.error('Failed to mark lesson as complete. Please try again.');
    } finally {
      setIsMarkingComplete(false);
    }
  };

  const handleQuizScoreSaved = (score: number) => {
    setQuizScore(score);
    if (score >= 70) {
      setIsCompleted(true);
    }
  };

  return (
    <div className="content-container mx-auto py-8 px-4">
      <Card className="mx-auto">
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <CardTitle className="text-3xl font-bold">{title}</CardTitle>
                {isSignedIn && lessonId && !progressLoading && (
                  <div className="flex items-center gap-2">
                    {isCompleted ? (
                      <Badge variant="success" className="flex items-center gap-1">
                        <CheckCircle className="h-3 w-3" />
                        Completed
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="flex items-center gap-1">
                        <Circle className="h-3 w-3" />
                        In Progress
                      </Badge>
                    )}
                    {quizScore !== null && (
                      <Badge variant={quizScore >= 70 ? "success" : "secondary"} className="ml-2">
                        Quiz: {quizScore}%
                      </Badge>
                    )}
                  </div>
                )}
              </div>
              <p className="text-muted-foreground mt-2">{description}</p>
            </div>
            {metaTopic && metaTopic.trim() && (
              <div className="flex flex-col gap-2 flex-shrink-0">
                <Badge variant="meta-topic" className="flex items-center gap-1">
                  <FolderOpen className="h-3 w-3" />
                  {metaTopic}
                </Badge>
              </div>
            )}
          </div>
          {!isSignedIn && (
            <SaveProgressPrompt
              title="Start Your Learning Journey"
              description="Sign in to track your progress, create custom schedules, and unlock unlimited lessons."
              className="mt-4"
            />
          )}
        </CardHeader>
        <CardContent>
          <div className="prose prose-slate dark:prose-invert max-w-none">
            <ReactMarkdown
              components={{
                a({ href, children, ...props }) {
                  const isExternal = href && /^https?:\/\//.test(href);
                  return (
                    <Button
                      asChild
                      variant="link"
                      className="p-0 h-auto align-baseline"
                    >
                      <a
                        href={href}
                        target={isExternal ? '_blank' : undefined}
                        rel={isExternal ? 'noopener noreferrer' : undefined}
                        {...props}
                      >
                        {children}
                      </a>
                    </Button>
                  );
                },
                code({ node, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  const language = match ? match[1] : '';
                  const isInteractive = className?.includes('interactive');
                  
                  if (language) {
                    if (isInteractive) {
                      const pythonCode = String(children).replace(/\n$/, '');
                      const javascriptCode = convertPythonToJS(pythonCode);
                      return (
                        <>
                          <CodePlaygroundTabs
                            pythonCode={pythonCode}
                            javascriptCode={javascriptCode}
                            title={`Interactive ${language.toUpperCase()} Playground`}
                          />
                          {!isSignedIn && (
                            <SaveProgressPrompt
                              title="Save Your Code Solutions"
                              description="Sign in to save your code solutions and get progress tracking with rewards."
                              className="mt-4"
                            />
                          )}
                        </>
                      );
                    }
                    return (
                      <CodeSnippet
                        code={String(children).replace(/\n$/, '')}
                        language={language}
                      />
                    );
                  }
                  
                  return (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                },
                h2({ node, children, ...props }) {
                  return (
                    <>
                      <h2 {...props}>{children}</h2>
                      {!isSignedIn && String(children).includes("Try It Yourself") && (
                        <SaveProgressPrompt
                          title="Track Your Learning"
                          description="Sign in to track your progress and get personalized learning paths."
                          className="mt-4 mb-8"
                        />
                      )}
                    </>
                  );
                }
              }}
            >
              {content}
            </ReactMarkdown>
            
            {/* Quiz and Completion Section */}
            {isSignedIn && lessonId && (
              <div className="mt-12 space-y-4">
                {/* Quiz Section */}
                <div className="p-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-semibold text-blue-800 dark:text-blue-200 mb-1">
                        Test Your Knowledge
                      </h3>
                      <p className="text-blue-700 dark:text-blue-300 text-sm">
                        Take a quiz to test your understanding of this lesson and earn a score.
                      </p>
                    </div>
                    <Button
                      onClick={() => setShowQuizModal(true)}
                      className="bg-blue-600 hover:bg-blue-700 text-white flex items-center gap-2"
                    >
                      <Brain className="h-4 w-4" />
                      {quizScore !== null ? 'Retake Quiz' : 'Take Quiz'}
                    </Button>
                  </div>
                </div>

                {/* Completion Section - Only show if not completed yet */}
                {!isCompleted && (
                  <div className="p-6 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg border border-green-200 dark:border-green-800">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-lg font-semibold text-green-800 dark:text-green-200 mb-1">
                          Ready to Complete This Lesson?
                        </h3>
                        <p className="text-green-700 dark:text-green-300 text-sm">
                          Mark this lesson as complete to track your progress and continue your learning journey.
                        </p>
                      </div>
                      <Button
                        onClick={markAsComplete}
                        disabled={isMarkingComplete}
                        className="bg-green-600 hover:bg-green-700 text-white"
                      >
                        {isMarkingComplete ? (
                          <>
                            <Circle className="h-4 w-4 mr-2 animate-spin" />
                            Completing...
                          </>
                        ) : (
                          <>
                            <CheckCircle className="h-4 w-4 mr-2" />
                            Mark as Complete
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Navigation buttons for lesson plans */}
            {navigationInfo && onNavigate && (
              <div className="flex items-center justify-between mt-12 pt-8 border-t border-gray-200">
                <div className="flex-1">
                  {navigationInfo.previousLesson && (
                    <Button
                      onClick={() => onNavigate(navigationInfo.previousLesson!.uuid)}
                      variant="outline"
                      className="flex items-center gap-2"
                    >
                      <ChevronLeft className="h-4 w-4" />
                      <div className="text-left">
                        <div className="text-xs text-muted-foreground">Previous</div>
                        <div className="font-medium">{navigationInfo.previousLesson.title}</div>
                      </div>
                    </Button>
                  )}
                </div>
                
                <div className="flex-shrink-0 mx-4">
                  <span className="text-sm text-muted-foreground">
                    {navigationInfo.currentIndex + 1} of {navigationInfo.totalLessons}
                  </span>
                </div>
                
                <div className="flex-1 flex justify-end">
                  {navigationInfo.nextLesson && (
                    <Button
                      onClick={() => onNavigate(navigationInfo.nextLesson!.uuid)}
                      variant="outline"
                      className="flex items-center gap-2"
                    >
                      <div className="text-right">
                        <div className="text-xs text-muted-foreground">Next</div>
                        <div className="font-medium">{navigationInfo.nextLesson.title}</div>
                      </div>
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            )}
            
            {!isSignedIn && (
              <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-3 gap-6 mt-8">
                <SaveProgressPrompt
                  title="Access Unlimited Lessons"
                  description="Sign in to unlock unlimited lessons and personalized learning paths."
                />
                
                <SaveProgressPrompt
                  title="Get Custom Schedules"
                  description="Sign in to create custom learning schedules that fit your lifestyle."
                />
                
                <SaveProgressPrompt
                  title="Track Your Progress"
                  description="Sign in to track your progress and earn rewards for your achievements."
                />
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Quiz Modal */}
      <LessonQuizModal
        isOpen={showQuizModal}
        onClose={() => setShowQuizModal(false)}
        lessonTitle={title}
        lessonId={lessonId}
        onScoreSaved={handleQuizScoreSaved}
      />
    </div>
  );
} 