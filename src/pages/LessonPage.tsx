import { useParams, useNavigate } from 'react-router-dom';
import { LessonViewer } from "@/components/LessonViewer";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import ChatAssistant from "@/components/ChatAssistant";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { ChevronDown, GraduationCap, ArrowLeft } from "lucide-react";
import type { LessonData, LessonSection } from "@/data/lessons/types";

// Define audience levels
export type AudienceLevel = 'elementary' | 'high-school' | 'undergraduate' | 'graduate';

const audienceLevels: { value: AudienceLevel; label: string; icon?: string }[] = [
  { value: 'elementary', label: 'Elementary School', icon: '🎓' },
  { value: 'high-school', label: 'High School', icon: '📚' },
  { value: 'undergraduate', label: 'Undergraduate', icon: '🎯' },
  { value: 'graduate', label: 'Graduate', icon: '🔬' }
];

export default function LessonPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [lesson, setLesson] = useState<LessonData | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [audienceLevel, setAudienceLevel] = useState<AudienceLevel>('undergraduate');

  useEffect(() => {
    if (!id) {
      setLesson(undefined);
      return;
    }
    setLoading(true);
    
    // Load lesson based on audience level
    import(`@/data/lessons/lesson-${id}/${audienceLevel}.ts`)
      .then(mod => {
        // Try named export lesson{id}, fallback to default
        const lessonData: LessonData = mod[`lesson${id}`] || mod.default;
        setLesson(lessonData);
      })
      .catch(() => {
        // If audience-specific version doesn't exist, try to load undergraduate as fallback
        if (audienceLevel !== 'undergraduate') {
          import(`@/data/lessons/lesson-${id}/undergraduate.ts`)
            .then(mod => {
              const lessonData: LessonData = mod[`lesson${id}`] || mod.default;
              setLesson(lessonData);
            })
            .catch(() => setLesson(undefined));
        } else {
          setLesson(undefined);
        }
      })
      .finally(() => setLoading(false));
  }, [id, audienceLevel]);

  if (loading) {
    return (
      <div className="container mx-auto py-12 px-4">
        <div className="mx-auto text-center">
          <h1 className="text-4xl font-bold mb-4">Loading Lesson...</h1>
        </div>
      </div>
    );
  }

  if (!lesson) {
    return (
      <div className="container mx-auto py-12 px-4">
        <div className="mx-auto text-center">
          <h1 className="text-4xl font-bold mb-4">Lesson Not Found</h1>
          <p className="text-muted-foreground mb-8">
            The lesson you're looking for doesn't exist or isn't available yet for the selected audience level.
          </p>
          <Button 
            onClick={() => navigate('/dashboard/lessons')}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Lessons
          </Button>
        </div>
      </div>
    );
  }

  // Combine all section content into a single markdown string with section headings
  const combinedContent = lesson.sections.map(
    (section: LessonSection) => `## ${section.title}\n\n${section.content}`
  ).join('\n\n');

  const currentAudienceData = audienceLevels.find(level => level.value === audienceLevel);

  return (
    <>
      <ChatAssistant />
      <div className="content-container mx-auto py-6 px-4">
        {/* Header with Audience Level Selector */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <Button 
              onClick={() => navigate('/dashboard/lessons')} 
              variant="outline" 
              size="sm"
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Lessons
            </Button>
          </div>
          <div className="flex items-center gap-3">
            <GraduationCap className="h-5 w-5 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Audience Level:</span>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="flex items-center gap-2">
                  <span>{currentAudienceData?.icon}</span>
                  {currentAudienceData?.label}
                  <ChevronDown className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                {audienceLevels.map((level) => (
                  <DropdownMenuItem
                    key={level.value}
                    onClick={() => setAudienceLevel(level.value)}
                    className={`flex items-center gap-2 ${
                      audienceLevel === level.value ? 'bg-accent' : ''
                    }`}
                  >
                    <span>{level.icon}</span>
                    {level.label}
                    {audienceLevel === level.value && (
                      <span className="ml-auto text-primary">✓</span>
                    )}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
      <LessonViewer title={lesson.title} description={lesson.description} content={combinedContent} />
    </>
  );
}