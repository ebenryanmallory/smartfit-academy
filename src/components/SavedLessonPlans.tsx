import { useState, useEffect, useImperativeHandle, forwardRef } from 'react';
import { useUser, useAuth } from '@clerk/clerk-react';
import { toast } from 'sonner';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { LessonContentLoader } from './ui/LessonContentLoader';
import { BookOpen, RefreshCw, Clock, GraduationCap, ChevronDown, ChevronRight, Trash2, Play, Tag, FolderOpen } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Link } from 'react-router-dom';

type EducationLevel = 'elementary' | 'highschool' | 'undergrad' | 'grad';

// Function to get display name for education level
const getEducationLevelDisplayName = (level: EducationLevel): string => {
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

interface SavedLesson {
  id: number;
  lesson_plan_id: number;
  lesson_order: number;
  title: string;
  description: string;
  content: string | null;
  uuid: string | null; // UUID for user-facing URLs
  created_at: string;
  updated_at: string;
}

interface SavedLessonPlan {
  id: number;
  user_id: string;
  topic: string;
  meta_topic?: string; // Optional meta topic field
  title: string;
  total_estimated_time: string | null;
  uuid: string | null; // UUID for user-facing URLs
  created_at: string;
  updated_at: string;
  lessons: SavedLesson[];
}

interface SavedLessonPlansProps {
  className?: string;
  onLessonPlansChange?: (lessonPlans: SavedLessonPlan[]) => void;
}

export interface SavedLessonPlansRef {
  refreshLessonPlans: () => Promise<void>;
  getLessonPlans: () => SavedLessonPlan[];
}

const SavedLessonPlans = forwardRef<SavedLessonPlansRef, SavedLessonPlansProps>(({ className = "", onLessonPlansChange }, ref) => {
  const { isSignedIn } = useUser();
  const { getToken, has } = useAuth();
  const [lessonPlans, setLessonPlans] = useState<SavedLessonPlan[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedPlans, setExpandedPlans] = useState<Set<number>>(new Set());
  const [expandedLessons, setExpandedLessons] = useState<Set<number>>(new Set());
  const [loadingContent, setLoadingContent] = useState<Set<number>>(new Set());
  const [userEducationLevel, setUserEducationLevel] = useState<EducationLevel>('undergrad');

  const fetchUserEducationLevel = async () => {
    if (!isSignedIn) return;

    try {
      const token = await getToken();
      if (!token) return;

      const response = await fetch('/api/d1/user', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });

      if (response.ok) {
        const data = await response.json();
        if (data.user?.education_level) {
          setUserEducationLevel(data.user.education_level as EducationLevel);
        }
      }
    } catch (error) {
      console.error('Error fetching user education level:', error);
      // Keep default value on error
    }
  };

  const fetchLessonPlans = async () => {
    if (!isSignedIn) {
      setError('Please sign in to view your lesson plans');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const token = await getToken();
      if (!token) {
        throw new Error('Failed to get authentication token');
      }
      
      const response = await fetch('/api/d1/user/lesson-plans', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Failed to fetch lesson plans:', response.status, errorText);
        throw new Error(`Failed to fetch lesson plans: ${response.status}`);
      }
      
      const data = await response.json();
      const fetchedLessonPlans = data.lessonPlans || [];
      setLessonPlans(fetchedLessonPlans);
      onLessonPlansChange?.(fetchedLessonPlans);
    } catch (err) {
      console.error('Error fetching lesson plans:', err);
      setError('Failed to load your lesson plans');
    } finally {
      setLoading(false);
    }
  };

  const deleteLessonPlan = async (lessonPlanId: number) => {
    if (!isSignedIn) {
      toast.error('Please sign in to delete lesson plans');
      return;
    }
    
    try {
      const token = await getToken();
      if (!token) {
        throw new Error('Failed to get authentication token');
      }
      
      const response = await fetch(`/api/d1/user/lesson-plans/${lessonPlanId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Failed to delete lesson plan:', response.status, errorText);
        throw new Error(`Failed to delete lesson plan: ${response.status}`);
      }
      
      // Refresh from server
      await fetchLessonPlans();
      toast.success('Lesson plan deleted successfully');
    } catch (err) {
      console.error('Error deleting lesson plan:', err);
      toast.error('Failed to delete lesson plan');
    }
  };

  const togglePlanExpansion = (planId: number) => {
    setExpandedPlans(prev => {
      const newSet = new Set(prev);
      if (newSet.has(planId)) {
        newSet.delete(planId);
      } else {
        newSet.add(planId);
      }
      return newSet;
    });
  };

  const toggleLessonExpansion = async (lessonId: number) => {
    // Find the lesson and check if it needs content generation
    const lesson = lessonPlans.flatMap(plan => plan.lessons).find(l => l.id === lessonId);
    if (!lesson) return;

    const isCurrentlyExpanded = expandedLessons.has(lessonId);
    
    // If expanding and no content exists, generate it first
    if (!isCurrentlyExpanded && !lesson.content) {
      await generateLessonContent(lessonId);
    }

    // Toggle expansion state
    setExpandedLessons(prev => {
      const newSet = new Set(prev);
      if (newSet.has(lessonId)) {
        newSet.delete(lessonId);
      } else {
        newSet.add(lessonId);
      }
      return newSet;
    });
  };

  // Generate content for a lesson that doesn't have it yet
  const generateLessonContent = async (lessonId: number) => {
    if (!isSignedIn) return;

    const lesson = lessonPlans.flatMap(plan => plan.lessons).find(l => l.id === lessonId);
    const lessonPlan = lessonPlans.find(plan => plan.lessons.some(l => l.id === lessonId));
    if (!lesson || !lessonPlan) return;

    // Set loading state
    setLoadingContent(prev => new Set([...prev, lessonId]));

    try {
      const token = await getToken();
      if (!token) throw new Error('Failed to get authentication token');

      // Check if user has monthly plan subscription
      const hasMonthlyPlan = has?.({ plan: 'monthly' }) ?? false;
      let response, data, responseContent;

      if (hasMonthlyPlan) {
        // Use Claude endpoint for monthly users
        response = await fetch('/claude/opus', {
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
**Lesson Title:** ${lesson.title}
**Lesson Description:** ${lesson.description || 'No specific description provided'}

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

        data = await response.json();
        responseContent = data.success && data.data?.content ? JSON.stringify(data.data.content) : '';
      } else {
        // Use llama3 endpoint for non-monthly users
        response = await fetch('/llm/llama3', {
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
**Lesson Title:** ${lesson.title}
**Lesson Description:** ${lesson.description || 'No specific description provided'}

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

        data = await response.json();
        responseContent = data.result?.response || data.response || '';
      }

      // Save the generated content to the database
      const saveResponse = await fetch(`/api/d1/user/lesson-plans/${lessonPlan.id}/lessons/${lessonId}`, {
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

      // Update the local state with the new content
      setLessonPlans(prev => prev.map(plan => ({
        ...plan,
        lessons: plan.lessons.map(l => 
          l.id === lessonId ? { ...l, content: responseContent } : l
        )
      })));

    } catch (error) {
      console.error('Error generating lesson content:', error);
      toast.error(`Failed to generate content for "${lesson.title}". Please try again.`);
    } finally {
      // Remove loading state
      setLoadingContent(prev => {
        const newSet = new Set(prev);
        newSet.delete(lessonId);
        return newSet;
      });
    }
  };

  // Expose methods to parent components
  useImperativeHandle(ref, () => ({
    refreshLessonPlans: fetchLessonPlans,
    getLessonPlans: () => lessonPlans,
  }));

  useEffect(() => {
    if (isSignedIn) {
      fetchLessonPlans();
      fetchUserEducationLevel();
    }
  }, [isSignedIn]);

  if (!isSignedIn) {
    return null;
  }

  if (loading) {
    return (
      <Card className={`p-4 ${className}`}>
        <div className="flex items-center gap-2 text-muted-foreground">
          <RefreshCw className="h-4 w-4 animate-spin" />
          <span>Loading your lesson plans...</span>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={`p-4 border-red-200 bg-red-50 ${className}`}>
        <div className="flex items-center justify-between mb-4">
          <span className="text-red-700">{error}</span>
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchLessonPlans}
            className="text-red-700 hover:text-red-800"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </Card>
    );
  }

  if (lessonPlans.length === 0) {
    return (
      <Card className={`p-6 text-center ${className}`}>
        <BookOpen className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
        <h3 className="text-lg font-medium mb-2">No Saved Lesson Plans</h3>
        <p className="text-muted-foreground mb-4">
          Generate and save lesson plans from the dashboard to see them here.
        </p>
        <Button asChild variant="outline">
          <Link to="/dashboard">
            Go to Dashboard
          </Link>
        </Button>
      </Card>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <BookOpen className="h-6 w-6" />
          Your Saved Lesson Plans ({lessonPlans.length})
        </h2>
        <div className="flex items-center gap-2">
          {lessonPlans.length > 0 && lessonPlans[0].lessons.length > 0 && (
            <Button
              asChild
              variant="default"
              size="sm"
            >
              <Link to={`/lessons/${lessonPlans[0].lessons[0].uuid || lessonPlans[0].lessons[0].id}`}>
                <Play className="h-4 w-4 mr-1" />
                Continue Learning
              </Link>
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchLessonPlans}
            className="text-muted-foreground hover:text-foreground"
            title="Refresh lesson plans"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {lessonPlans.map((plan) => (
        <Card key={plan.id} className="overflow-hidden">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 flex-1">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => togglePlanExpansion(plan.id)}
                  className="p-1 h-auto"
                >
                  {expandedPlans.has(plan.id) ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                </Button>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <CardTitle className="text-lg">{plan.title}</CardTitle>
                    <div className="flex gap-1">
                      {plan.meta_topic && plan.meta_topic.trim() && (
                        <Badge variant="meta-topic" className="flex-shrink-0 flex items-center gap-1">
                          <FolderOpen className="h-3 w-3" />
                          {plan.meta_topic}
                        </Badge>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <BookOpen className="h-3 w-3" />
                      {plan.lessons.length} lessons
                    </span>
                    {plan.total_estimated_time && (
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {plan.total_estimated_time}
                      </span>
                    )}
                    <span className="flex items-center gap-1">
                      <GraduationCap className="h-3 w-3" />
                      {getEducationLevelDisplayName(userEducationLevel)}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Created: {new Date(plan.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => deleteLessonPlan(plan.id)}
                className="text-red-500 hover:text-red-700 p-1 h-auto"
                title="Delete lesson plan"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>

          {expandedPlans.has(plan.id) && (
            <CardContent className="pt-0">
              <div className="space-y-3">
                {plan.lessons.map((lesson) => (
                  <Card key={lesson.id} className="border-l-4 border-l-primary/20">
                    <CardHeader className="pb-2">
                                              <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 flex-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleLessonExpansion(lesson.id)}
                            className="p-1 h-auto"
                          >
                            {expandedLessons.has(lesson.id) ? (
                              <ChevronDown className="h-3 w-3" />
                            ) : (
                              <ChevronRight className="h-3 w-3" />
                            )}
                          </Button>
                          <div className="flex-1">
                            <h4 className="font-medium text-sm">{lesson.title}</h4>
                            {lesson.description && (
                              <p className="text-xs text-muted-foreground mt-1">
                                {lesson.description}
                              </p>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            asChild
                            variant="outline"
                            size="sm"
                            className="text-xs h-7"
                          >
                            <Link to={`/lessons/${lesson.uuid || lesson.id}`}>
                              <Play className="h-3 w-3 mr-1" />
                              Start Lesson
                            </Link>
                          </Button>
                          <span className="text-xs text-muted-foreground bg-gray-100 px-2 py-1 rounded">
                            Lesson {lesson.lesson_order} of {plan.lessons.length}
                          </span>
                        </div>
                      </div>
                    </CardHeader>

                    {expandedLessons.has(lesson.id) && (
                      <CardContent className="pt-0">
                        {loadingContent.has(lesson.id) ? (
                          <LessonContentLoader lessonTitle={lesson.title} variant="inline" />
                        ) : lesson.content ? (
                          <div className="border rounded-lg p-3 bg-muted/20">
                            <div className="prose prose-sm max-w-none">
                              <ReactMarkdown>{lesson.content}</ReactMarkdown>
                            </div>
                          </div>
                        ) : (
                          <div className="text-center py-4">
                            <p className="text-sm text-muted-foreground">
                              Content will be generated automatically when you expand this lesson.
                            </p>
                          </div>
                        )}
                      </CardContent>
                    )}
                  </Card>
                ))}
              </div>
            </CardContent>
          )}
        </Card>
      ))}
    </div>
  );
});

SavedLessonPlans.displayName = 'SavedLessonPlans';

export default SavedLessonPlans; 