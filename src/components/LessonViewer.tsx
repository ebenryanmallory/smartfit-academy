import ReactMarkdown from 'react-markdown';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useUser } from "@clerk/clerk-react";
import { CodeSnippet } from './CodeSnippet';
import { CodePlaygroundTabs } from './CodePlaygroundTabs';
import { SaveProgressPrompt } from './SaveProgressPrompt';
import { ChevronLeft, ChevronRight } from 'lucide-react';

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
  navigationInfo?: NavigationInfo | null;
  onNavigate?: (lessonUuid: string) => void;
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

export function LessonViewer({ title, description, content, navigationInfo, onNavigate }: LessonViewerProps) {
  const { isSignedIn } = useUser();

  return (
    <div className="content-container mx-auto py-8 px-4">
      <Card className="mx-auto">
        <CardHeader>
          <CardTitle className="text-3xl font-bold">{title}</CardTitle>
          <p className="text-muted-foreground mt-2">{description}</p>
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
    </div>
  );
} 