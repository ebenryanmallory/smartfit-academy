import ReactMarkdown from 'react-markdown';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CodeSnippet } from './CodeSnippet';
import { CodePlaygroundTabs } from './CodePlaygroundTabs';
import { SaveProgressPrompt } from './SaveProgressPrompt';

interface LessonViewerProps {
  title: string;
  description: string;
  content: string;
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

export function LessonViewer({ title, description, content }: LessonViewerProps) {
  return (
    <div className="container mx-auto py-8 px-4">
      <Card className="max-w-4xl mx-auto">
        <CardHeader>
          <CardTitle className="text-3xl font-bold">{title}</CardTitle>
          <p className="text-muted-foreground mt-2">{description}</p>
          <SaveProgressPrompt
            title="Start Your Learning Journey"
            description="Sign in to track your progress and unlock more lessons."
            className="mt-4"
            buttonText="Sign in to start learning"
          />
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
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  const language = match ? match[1] : '';
                  const isInteractive = className?.includes('interactive');
                  
                  if (!inline && language) {
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
                          <SaveProgressPrompt
                            title="Save Your Code Solutions"
                            description="Sign in to save your code solutions and track your progress."
                            className="mt-4"
                            buttonText="Sign in to save your code"
                          />
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
                      {String(children).includes("Try It Yourself") && (
                        <SaveProgressPrompt
                          title="Track Your Learning"
                          description="Sign in to track your progress and get personalized recommendations."
                          className="mt-4 mb-8"
                          buttonText="Sign in to track progress"
                        />
                      )}
                    </>
                  );
                }
              }}
            >
              {content}
            </ReactMarkdown>
            
            <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-3 gap-6 mt-8">
              <SaveProgressPrompt
                title="Access More Lessons"
                description="Sign in to unlock our full library of lessons and learning paths."
                buttonText="Sign in to access more lessons"
              />
              
              <SaveProgressPrompt
                title="Get Personalized Recommendations"
                description="Sign in to receive AI-powered lesson recommendations based on your progress."
                buttonText="Sign in for recommendations"
              />
              
              <SaveProgressPrompt
                title="Join Our Learning Community"
                description="Sign in to connect with other learners and share your progress."
                buttonText="Sign in to join community"
              />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 