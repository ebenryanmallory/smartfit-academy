import React from 'react';
import { Card, CardContent } from './card';
import { Loader2 } from 'lucide-react';

interface LessonContentLoaderProps {
  lessonTitle?: string;
  topic?: string;
  variant?: 'inline' | 'card' | 'fullscreen';
  className?: string;
}

export const LessonContentLoader: React.FC<LessonContentLoaderProps> = ({ 
  lessonTitle, 
  topic, 
  variant = 'card',
  className = '' 
}) => {
  const content = (
    <div className="text-center py-8">
      <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
      <h3 className="text-lg font-medium mb-2">Generating Lesson Content</h3>
      <p className="text-muted-foreground">
        {lessonTitle 
          ? `Our AI is crafting detailed content for "${lessonTitle}"...`
          : 'Our AI is creating personalized educational content...'
        }
      </p>
      <p className="text-sm text-muted-foreground mt-2">
        This may take a moment to ensure quality educational content.
      </p>
    </div>
  );

  if (variant === 'fullscreen') {
    return (
      <div className={`content-container mx-auto py-12 px-4 ${className}`}>
        <div className="mx-auto text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
          <h1 className="text-4xl font-bold mb-4">Generating Lesson Content...</h1>
          <p className="text-muted-foreground">
            {lessonTitle 
              ? `Creating personalized content for "${lessonTitle}"`
              : 'This may take a moment as we create personalized content for you.'
            }
          </p>
        </div>
      </div>
    );
  }

  if (variant === 'inline') {
    return (
      <div className={`flex items-center justify-center py-8 ${className}`}>
        <Loader2 className="h-6 w-6 animate-spin mr-2" />
        <span className="text-sm text-muted-foreground">
          {lessonTitle 
            ? `Generating content for "${lessonTitle}"...`
            : 'Generating lesson content...'
          }
        </span>
      </div>
    );
  }

  // Default 'card' variant
  return (
    <Card className={`flex items-center justify-center min-h-[200px] ${className}`}>
      <CardContent>
        {content}
      </CardContent>
    </Card>
  );
}; 