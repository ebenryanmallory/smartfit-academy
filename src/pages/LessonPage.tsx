import { useParams, useNavigate } from 'react-router-dom';
import { LessonViewer } from "@/components/LessonViewer";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import ChatAssistant from "@/components/ChatAssistant";
import { ArrowLeft, AlertTriangle, RefreshCw } from "lucide-react";
import { LessonContentLoader } from "@/components/ui/LessonContentLoader";
import type { LessonData, LessonSection } from "@/data/lessons/types";
import EducationLevelSelector, { type AudienceLevel } from "@/components/EducationLevelSelector";
import { useUser, useAuth } from '@clerk/clerk-react';
import { toast } from 'sonner';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { 
  getLessonType, 
  supportsEducationLevelSwitching
} from "@/utils/lessonIdUtils";

interface SavedLesson {
  id: number;
  lesson_plan_id: number;
  lesson_order: number;
  title: string;
  description: string;
  content: string | null;
  uuid: string | null;
  topic: string;
  meta_topic?: string;
  plan_title: string;
  total_estimated_time: string | null;
}

interface LessonPlanData {
  id: number;
  title: string;
  topic: string;
  lessons: Array<{
    id: number;
    uuid: string | null;
    title: string;
    lesson_order: number;
  }>;
}

interface NavigationInfo {
  currentIndex: number;
  totalLessons: number;
  previousLesson?: { uuid: string; title: string };
  nextLesson?: { uuid: string; title: string };
}

// Function to get display name for education level
const getEducationLevelDisplayName = (level: string): string => {
  switch (level) {
    case 'elementary':
      return 'Elementary School';
    case 'highschool':
      return 'High School';
    case 'undergrad':
      return 'Undergraduate';
    case 'grad':
      return 'Graduate';
    default:
      return 'Undergraduate';
  }
};

export default function LessonPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { isSignedIn } = useUser();
  const { getToken } = useAuth();
  const [lesson, setLesson] = useState<LessonData | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [generatingContent, setGeneratingContent] = useState(false);
  const [audienceLevel, setAudienceLevel] = useState<AudienceLevel>('undergraduate');
  const [lessonType, setLessonType] = useState<'custom' | 'generated'>('custom');
  const [, setLessonPlan] = useState<LessonPlanData | null>(null);
  const [navigationInfo, setNavigationInfo] = useState<NavigationInfo | null>(null);
  const [showEducationWarning, setShowEducationWarning] = useState(false);
  const [pendingEducationLevel, setPendingEducationLevel] = useState<AudienceLevel | null>(null);
  const [originalEducationLevel, setOriginalEducationLevel] = useState<AudienceLevel>('undergraduate');
  const [userEducationLevel, setUserEducationLevel] = useState<string>('undergrad');

  // Generate content for a lesson that doesn't have it yet
  const generateLessonContent = async (savedLesson: SavedLesson, lessonPlan: any) => {
    if (!isSignedIn) return null;

    setGeneratingContent(true);

    try {
      const token = await getToken();
      if (!token) throw new Error('Failed to get authentication token');

      // Generate content using the same API as the SavedLessonPlans component
      const response = await fetch('/llm/llama3', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          messages: [
            {
              role: 'user',
              content: `Create detailed lesson content for:

**Topic:** ${lessonPlan.topic}
**Lesson Title:** ${savedLesson.title}
**Lesson Description:** ${savedLesson.description || 'No specific description provided'}

Please generate comprehensive, educational content in markdown format for this specific lesson. The content should be engaging, informative, and appropriate for ${getEducationLevelDisplayName(userEducationLevel)} students.

Include practical examples, clear explanations, and interactive elements like questions or exercises where appropriate.`
            }
          ],
          instructionType: 'lessonContentGenerator',
          educationLevel: userEducationLevel
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to generate content: ${response.status}`);
      }

      const data = await response.json();
      const responseContent = data.result?.response || data.response || '';

      // Save the generated content to the database
      const saveResponse = await fetch(`/api/d1/user/lesson-plans/${savedLesson.lesson_plan_id}/lessons/${savedLesson.id}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          content: responseContent
        }),
      });

      if (!saveResponse.ok) {
        throw new Error(`Failed to save content: ${saveResponse.status}`);
      }

      return responseContent;

    } catch (error) {
      console.error('Error generating lesson content:', error);
      toast.error(`Failed to generate content for "${savedLesson.title}". Please try again.`);
      return null;
    } finally {
      setGeneratingContent(false);
    }
  };

  useEffect(() => {
    if (!id) {
      setLesson(undefined);
      return;
    }
    setLoading(true);
    
    // Determine lesson type using the new utility
    const currentLessonType = getLessonType(id);
    setLessonType(currentLessonType);
    
    if (currentLessonType === 'generated') {
      
      // Load saved lesson from database
      const loadSavedLesson = async () => {
        if (!isSignedIn) {
          setLesson(undefined);
          setLoading(false);
          return;
        }
        
        try {
          const token = await getToken();
          if (!token) {
            throw new Error('Failed to get authentication token');
          }

          // First, fetch user's current education level to use as the original level
          const userResponse = await fetch('/api/d1/user', {
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
            credentials: 'include',
          });

          if (userResponse.ok) {
            const userData = await userResponse.json();
            if (userData.user?.education_level) {
              setUserEducationLevel(userData.user.education_level);
              
              // Convert backend education level to audience level
              const educationToAudience = (education: string): AudienceLevel => {
                switch (education) {
                  case 'elementary': return 'elementary';
                  case 'highschool': return 'high-school';
                  case 'undergrad': return 'undergraduate';
                  case 'grad': return 'graduate';
                  default: return 'undergraduate';
                }
              };
              
              const userAudienceLevel = educationToAudience(userData.user.education_level);
              setAudienceLevel(userAudienceLevel);
              setOriginalEducationLevel(userAudienceLevel);
            }
          }
          
          const response = await fetch(`/api/d1/user/lessons/${id}`, {
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
            credentials: 'include',
          });
          
          if (!response.ok) {
            throw new Error(`Failed to fetch lesson: ${response.status}`);
          }
          
          const data = await response.json();
          const savedLesson: SavedLesson = data.lesson;
          
          // Fetch lesson plan data for content generation and navigation
          const lessonPlanResponse = await fetch(`/api/d1/user/lesson-plans/${savedLesson.lesson_plan_id}`, {
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
            credentials: 'include',
          });

          let lessonPlanData = null;
          if (lessonPlanResponse.ok) {
            const planData = await lessonPlanResponse.json();
            lessonPlanData = planData.lessonPlan;
            const plan: LessonPlanData = {
              id: lessonPlanData.id,
              title: lessonPlanData.title,
              topic: lessonPlanData.topic,
              lessons: lessonPlanData.lessons.map((l: any) => ({
                id: l.id,
                uuid: l.uuid,
                title: l.title,
                lesson_order: l.lesson_order
              }))
            };
            setLessonPlan(plan);

            // Calculate navigation info
            const currentIndex = plan.lessons.findIndex(l => l.uuid === id || l.id.toString() === id);
            if (currentIndex !== -1) {
              const navInfo: NavigationInfo = {
                currentIndex,
                totalLessons: plan.lessons.length,
                previousLesson: currentIndex > 0 ? {
                  uuid: plan.lessons[currentIndex - 1].uuid || plan.lessons[currentIndex - 1].id.toString(),
                  title: plan.lessons[currentIndex - 1].title
                } : undefined,
                nextLesson: currentIndex < plan.lessons.length - 1 ? {
                  uuid: plan.lessons[currentIndex + 1].uuid || plan.lessons[currentIndex + 1].id.toString(),
                  title: plan.lessons[currentIndex + 1].title
                } : undefined
              };
              setNavigationInfo(navInfo);
            }
          }

          // If lesson has no content, generate it automatically
          let lessonContent = savedLesson.content;
          if (!lessonContent && lessonPlanData) {
            lessonContent = await generateLessonContent(savedLesson, lessonPlanData);
          }
          
          // Convert saved lesson to LessonData format
          const lessonData: LessonData = {
            id: savedLesson.uuid || savedLesson.id.toString(),
            title: savedLesson.title,
            description: savedLesson.description || `Part of "${savedLesson.plan_title}" lesson plan`,
            topic: savedLesson.topic, // Include topic from the lesson plan
            // Only include metaTopic if it exists and is not empty
            ...(savedLesson.meta_topic && savedLesson.meta_topic.trim() && {
              metaTopic: savedLesson.meta_topic.trim()
            }),
            sections: lessonContent ? [
              {
                title: savedLesson.title,
                content: lessonContent
              }
            ] : []
          };
          
          setLesson(lessonData);
        } catch (error) {
          console.error('Error loading saved lesson:', error);
          setLesson(undefined);
        } finally {
          setLoading(false);
        }
      };
      
      loadSavedLesson();
    } else {
      // Load hardcoded/custom lesson based on audience level
      
              import(`@/data/lessons/${id}/${audienceLevel}.ts`)
        .then(mod => {
          // Use the standard 'lesson' export for custom lessons
          const lessonData: LessonData = mod.lesson || mod.default;
          setLesson(lessonData);
        })
        .catch(() => {
          // If audience-specific version doesn't exist, try to load undergraduate as fallback
          if (audienceLevel !== 'undergraduate') {
            import(`@/data/lessons/${id}/undergraduate.ts`)
              .then(mod => {
                const lessonData: LessonData = mod.lesson || mod.default;
                setLesson(lessonData);
              })
              .catch(() => setLesson(undefined));
          } else {
            setLesson(undefined);
          }
        })
        .finally(() => setLoading(false));
    }
  }, [id, audienceLevel, isSignedIn, getToken]);

  if (loading) {
    return (
      <div className="content-container mx-auto py-12 px-4">
        <div className="mx-auto text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <RefreshCw className="h-6 w-6 animate-spin" />
          </div>
          <h1 className="text-4xl font-bold mb-4">Loading Lesson...</h1>
        </div>
      </div>
    );
  }

  if (generatingContent) {
    return <LessonContentLoader variant="fullscreen" />;
  }

  if (!lesson) {
    return (
      <div className="content-container mx-auto py-12 px-4">
        <div className="mx-auto text-center">
          <h1 className="text-4xl font-bold mb-4">Lesson Not Found</h1>
          <p className="text-muted-foreground mb-8">
            {lessonType === 'generated' && !isSignedIn 
              ? "Please sign in to view your saved lessons."
              : lessonType === 'generated'
                ? "This saved lesson doesn't exist or you don't have access to it."
                : "The lesson you're looking for doesn't exist or isn't available yet for the selected audience level."
            }
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

  const handleEducationLevelChange = (newLevel: AudienceLevel) => {
    if (lessonType === 'generated' && newLevel !== originalEducationLevel) {
      // Show warning modal for generated lessons
      setPendingEducationLevel(newLevel);
      setShowEducationWarning(true);
    } else {
      // For custom/legacy lessons, navigate to the same lesson at the new education level
      if (supportsEducationLevelSwitching(id || '') && id) {
        // Navigate to the same lesson ID but with the new audience level
        // The useEffect will handle loading the lesson at the new level
        setAudienceLevel(newLevel);
        // The lesson will reload automatically due to the audienceLevel dependency in useEffect
      } else {
        // Fallback for any other cases
        setAudienceLevel(newLevel);
      }
    }
  };

  const handleEducationWarningConfirm = () => {
    if (pendingEducationLevel) {
      setAudienceLevel(pendingEducationLevel);
    }
    setShowEducationWarning(false);
    setPendingEducationLevel(null);
  };

  const handleEducationWarningCancel = () => {
    setShowEducationWarning(false);
    setPendingEducationLevel(null);
  };

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
            {lessonType === 'generated' && (
              <span className="text-sm text-muted-foreground bg-blue-50 px-2 py-1 rounded">
                Generated Lesson
              </span>
            )}
            {lessonType === 'custom' && (
              <span className="text-sm text-muted-foreground bg-green-50 px-2 py-1 rounded">
                Custom Lesson
              </span>
            )}
          </div>
          <EducationLevelSelector 
            value={audienceLevel}
            onChange={handleEducationLevelChange}
            variant="dropdown"
            showLabel={true}
          />
        </div>
      </div>
      <LessonViewer 
        title={lesson.title} 
        description={lesson.description} 
        content={combinedContent}
        topic={lesson.topic}
        metaTopic={lesson.metaTopic}
        navigationInfo={navigationInfo}
        onNavigate={(lessonUuid) => navigate(`/lessons/${lessonUuid}`)}
      />

      {/* Education Level Change Warning Modal */}
      <Dialog open={showEducationWarning} onOpenChange={setShowEducationWarning}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-500" />
              Change Education Level?
            </DialogTitle>
            <DialogDescription className="text-left space-y-3">
              <p>
                This is a <strong>saved lesson</strong> that was generated specifically for <strong>{originalEducationLevel}</strong> level students.
              </p>
              <p>
                Changing your education level will only affect how <strong>new lessons</strong> are generated. 
                This current lesson will remain at the {originalEducationLevel} level.
              </p>
              <p className="text-sm text-muted-foreground">
                To get this lesson content at the {pendingEducationLevel} level, you would need to regenerate 
                the entire lesson plan from your dashboard.
              </p>
              <p className="text-xs text-muted-foreground italic">
                Note: For standard lessons (not saved), you can switch between education levels instantly.
              </p>
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex-col sm:flex-row gap-2">
            <Button
              variant="outline"
              onClick={handleEducationWarningCancel}
              className="w-full sm:w-auto"
            >
              Keep Current Level
            </Button>
            <Button
              onClick={handleEducationWarningConfirm}
              className="w-full sm:w-auto"
            >
              Change Level Anyway
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}