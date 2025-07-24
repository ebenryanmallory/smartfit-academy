import ReactMarkdown from 'react-markdown';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from '@/components/ui/badge';
import { useUser, useAuth } from "@clerk/clerk-react";
import { CodeSnippet } from './CodeSnippet';
import { CodePlaygroundTabs } from './CodePlaygroundTabs';
import { SaveProgressPrompt } from './SaveProgressPrompt';
import { LessonQuizModal } from './LessonQuizModal';
import { TooltipWithContent } from './ui/tooltip';
import { ChevronLeft, ChevronRight, FolderOpen, CheckCircle, Circle, Brain } from 'lucide-react';
import React, { useEffect, useState } from 'react';
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
  onTopicExploration?: (topicText: string, context?: { lessonTitle?: string; listTitle?: string }) => void; // Callback for topic exploration
}

// Helper function to extract text content from React children
function extractTextContent(children: React.ReactNode): string {
  if (typeof children === 'string') {
    return children;
  }
  if (typeof children === 'number') {
    return String(children);
  }
  if (React.isValidElement(children)) {
    return extractTextContent(children.props.children);
  }
  if (Array.isArray(children)) {
    return children.map(child => extractTextContent(child)).join('');
  }
  return '';
}


export function LessonViewer({ title, description, content, metaTopic, navigationInfo, onNavigate, lessonId, onTopicExploration }: LessonViewerProps) {
  const { isSignedIn } = useUser();
  const { getToken } = useAuth();
  const [isCompleted, setIsCompleted] = useState(false);
  const [isMarkingComplete, setIsMarkingComplete] = useState(false);
  const [progressLoading, setProgressLoading] = useState(false);
  const [showQuizModal, setShowQuizModal] = useState(false);
  const [quizScore, setQuizScore] = useState<number | null>(null);
  
  // Track the last paragraph text for list title extraction
  const lastParagraphRef = React.useRef<string | null>(null);
  // Track current list title for all li elements in the current list
  const currentListTitleRef = React.useRef<string | null>(null);

  // Handle topic exploration click
  const handleTopicExploration = (topicText: string, listTitle?: string) => {
    const context = {
      lessonTitle: title,
      listTitle: listTitle
    };
    onTopicExploration?.(topicText, context);
  };

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
                h1({ node, children, ...props }) {
                  return (
                    <h1 className="text-2xl font-bold mb-4 mt-8 text-gray-900 dark:text-white" {...props}>
                      {children}
                    </h1>
                  );
                },
                h2({ node, children, ...props }) {
                  return (
                    <>
                      <h2 className="text-xl font-semibold mb-3 mt-6 text-gray-900 dark:text-white" {...props}>
                        {children}
                      </h2>
                      {!isSignedIn && String(children).includes("Try It Yourself") && (
                        <SaveProgressPrompt
                          title="Track Your Learning"
                          description="Sign in to track your progress and get personalized learning paths."
                          className="mt-4 mb-8"
                        />
                      )}
                    </>
                  );
                },
                h3({ node, children, ...props }) {
                  return (
                    <h3 className="text-lg font-semibold mb-3 mt-5 text-gray-900 dark:text-white" {...props}>
                      {children}
                    </h3>
                  );
                },
                h4({ node, children, ...props }) {
                  return (
                    <h4 className="text-base font-semibold mb-2 mt-4 text-gray-900 dark:text-white" {...props}>
                      {children}
                    </h4>
                  );
                },
                h5({ node, children, ...props }) {
                  return (
                    <h5 className="text-sm font-semibold mb-2 mt-3 text-gray-900 dark:text-white" {...props}>
                      {children}
                    </h5>
                  );
                },
                h6({ node, children, ...props }) {
                  return (
                    <h6 className="text-sm font-medium mb-2 mt-3 text-gray-700 dark:text-gray-300" {...props}>
                      {children}
                    </h6>
                  );
                },
                ul({ node, children, ...props }) {
                  // Store the current last paragraph as this list's title
                  const currentListTitle = lastParagraphRef.current;
                  console.log('List starting with potential title:', currentListTitle);
                  
                  // Set the current list title for all li elements in this list
                  currentListTitleRef.current = currentListTitle;
                  
                  // Clear the last paragraph ref so it doesn't get reused for subsequent lists
                  lastParagraphRef.current = null;
                  
                  return (
                    <ul 
                      className="list-disc list-inside mb-4 space-y-2 text-gray-700 dark:text-gray-300" 
                      {...props}
                    >
                      {children}
                    </ul>
                  );
                },
                ol({ node, children, ...props }) {
                  // Store the current last paragraph as this list's title
                  const currentListTitle = lastParagraphRef.current;
                  console.log('Ordered list starting with potential title:', currentListTitle);
                  
                  // Set the current list title for all li elements in this list
                  currentListTitleRef.current = currentListTitle;
                  
                  // Clear the last paragraph ref so it doesn't get reused for subsequent lists
                  lastParagraphRef.current = null;
                  
                  return (
                    <ol 
                      className="list-decimal list-inside mb-4 space-y-2 text-gray-700 dark:text-gray-300" 
                      {...props}
                    >
                      {children}
                    </ol>
                  );
                },
                li({ node, children, ...props }) {
                  // Get the text content of the li to check for separators
                  const liTextContent = extractTextContent(children);
                  
                  // Get the list title from the current list title ref
                  const listTitle = currentListTitleRef.current || undefined;
                  
                  console.log('Processing li item:', {
                    itemText: liTextContent,
                    listTitle: listTitle
                  });
                  
                  // Check if this li contains a strong element that could be a title
                  let hasStrongTitle = false;
                  
                  // Process children to handle complex list items with bold titles
                  const processedChildren = React.Children.map(children, (child) => {
                    if (React.isValidElement(child) && child.type === 'strong') {
                      const strongChild = child as React.ReactElement<{ children: React.ReactNode }>;
                      const textContent = extractTextContent(strongChild.props.children);
                      // Check if this strong element is likely a list item title
                      const isListItemTitle = textContent.length > 3 && textContent.includes(' ');
                      
                      if (isListItemTitle) {
                        hasStrongTitle = true;
                        return (
                          <TooltipWithContent
                            content="Click to explore this topic in depth"
                            side="top"
                            delayDuration={300}
                          >
                            <button
                              onClick={() => handleTopicExploration(textContent, listTitle)}
                              className="font-semibold text-foreground hover:text-primary cursor-pointer transition-colors duration-200 underline-offset-2 hover:underline"
                              type="button"
                            >
                              {strongChild.props.children}
                            </button>
                          </TooltipWithContent>
                        );
                      }
                    }
                    return child;
                  });

                  // Check if li contains a colon separator and no strong title
                  const colonIndex = liTextContent.indexOf(':');
                  if (!hasStrongTitle && colonIndex > 0 && colonIndex < liTextContent.length - 1) {
                    const titlePart = liTextContent.substring(0, colonIndex).trim();
                    const descriptionPart = liTextContent.substring(colonIndex + 1).trim();
                    
                    if (titlePart.length > 0 && descriptionPart.length > 0) {
                      return (
                        <li className="mb-1 text-gray-700 dark:text-gray-300" {...props}>
                          <TooltipWithContent
                            content="Click to explore this topic in depth"
                            side="top"
                            delayDuration={300}
                          >
                            <button
                              onClick={() => handleTopicExploration(titlePart, listTitle)}
                              className="font-semibold text-foreground hover:text-primary cursor-pointer transition-colors duration-200 underline-offset-2 hover:underline"
                              type="button"
                            >
                              {titlePart}
                            </button>
                          </TooltipWithContent>
                          : {descriptionPart}
                        </li>
                      );
                    }
                  }

                  // If no strong title or colon separator found, make the entire li clickable
                  if (!hasStrongTitle && colonIndex === -1) {
                    return (
                      <TooltipWithContent
                        content="Click to explore this topic in depth"
                        side="top"
                        align="start"
                        delayDuration={300}
                      >
                        <li 
                          className="mb-1 text-foreground hover:text-primary cursor-pointer transition-colors duration-200 underline-offset-2 hover:underline"
                          onClick={() => handleTopicExploration(liTextContent, listTitle)}
                          {...props}
                        >
                          {children}
                        </li>
                      </TooltipWithContent>
                    );
                  }

                  return (
                    <li className="mb-1 text-gray-700 dark:text-gray-300" {...props}>
                      {processedChildren}
                    </li>
                  );
                },
                p({ node, children, ...props }) {
                  // Extract text content and store it for potential list title use
                  const paragraphText = extractTextContent(children);
                  
                  // Store this paragraph text for potential use by the next list
                  // Only store if it's not empty and looks like it could be a list title
                  if (paragraphText && paragraphText.trim().length > 0) {
                    lastParagraphRef.current = paragraphText.trim();
                    console.log('Captured paragraph text for potential list title:', paragraphText.trim());
                  }
                  
                  return (
                    <p className="mb-4 text-gray-700 dark:text-gray-300 leading-relaxed" {...props}>
                      {children}
                    </p>
                  );
                },
                blockquote({ node, children, ...props }) {
                  return (
                    <blockquote className="border-l-4 border-blue-500 pl-4 mb-4 italic text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-800 py-2 pr-4 rounded-r" {...props}>
                      {children}
                    </blockquote>
                  );
                },
                strong({ node, children, ...props }) {
                  return (
                    <strong className="font-semibold text-gray-900 dark:text-white" {...props}>
                      {children}
                    </strong>
                  );
                },
                em({ node, children, ...props }) {
                  return (
                    <em className="italic text-gray-700 dark:text-gray-300" {...props}>
                      {children}
                    </em>
                  );
                },
                a({ href, children, ...props }) {
                  const isExternal = href && /^https?:\/\//.test(href);
                  return (
                    <Button
                      asChild
                      variant="link"
                      className="p-0 h-auto align-baseline text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
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
                      return (
                        <>
                          <CodePlaygroundTabs
                            pythonCode={pythonCode}
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