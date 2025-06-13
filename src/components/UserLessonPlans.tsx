import { useState, useEffect, useImperativeHandle, forwardRef } from 'react';
import { useUser, useAuth } from '@clerk/clerk-react';
import { toast } from 'sonner';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { BookOpen, X, RefreshCw, Clock, GraduationCap, ExternalLink } from 'lucide-react';
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
  created_at: string;
  updated_at: string;
}

interface SavedLessonPlan {
  id: number;
  user_id: string;
  topic: string;
  title: string;
  total_estimated_time: string | null;

  created_at: string;
  updated_at: string;
  lessons: SavedLesson[];
}

interface UserLessonPlansProps {
  className?: string;
  onLessonPlansChange?: (lessonPlans: SavedLessonPlan[]) => void;
}

export interface UserLessonPlansRef {
  refreshLessonPlans: () => Promise<void>;
  getLessonPlans: () => SavedLessonPlan[];
}

const UserLessonPlans = forwardRef<UserLessonPlansRef, UserLessonPlansProps>(({ className = "", onLessonPlansChange }, ref) => {
  const { isSignedIn } = useUser();
  const { getToken } = useAuth();
  const [lessonPlans, setLessonPlans] = useState<SavedLessonPlan[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
      <section className="container-section bg-gradient-to-r from-blue-50 to-indigo-50">
        <div className="content-container">
          <Card className={`p-4 ${className}`}>
            <div className="flex items-center gap-2 text-muted-foreground">
              <RefreshCw className="h-4 w-4 animate-spin" />
              <span>Loading your lesson plans...</span>
            </div>
          </Card>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="container-section bg-gradient-to-r from-red-50 to-pink-50">
        <div className="content-container">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-6 text-foreground">
            Your Lesson Plans
          </h2>
          <div className="max-w-4xl mx-auto">
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
          </div>
        </div>
      </section>
    );
  }

  if (lessonPlans.length === 0) {
    return null; // Don't show anything if no lesson plans
  }

  return (
    <section className="container-section bg-gradient-to-r from-green-50 to-emerald-50">
      <div className="content-container">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-6 text-foreground">
          Continue Your Learning Journey
        </h2>
        <p className="text-lg text-muted-foreground text-center mb-8 max-w-2xl mx-auto">
          Pick up where you left off with your saved lesson plans
        </p>
        <div className="max-w-4xl mx-auto">
          <Card className={`p-4 bg-white shadow-sm ${className}`}>
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-foreground flex items-center gap-2">
                <BookOpen className="h-4 w-4" />
                Your Lesson Plans ({lessonPlans.length})
              </h3>
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
            
            <div className="space-y-3">
              {lessonPlans.map((plan) => (
                <div key={plan.id} className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 transition-colors">
                  <div className="flex-1">
                    <h4 className="font-medium text-sm text-foreground">{plan.title}</h4>
                    <div className="flex items-center gap-4 mt-1 text-xs text-muted-foreground">
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
                  <div className="flex items-center gap-2 ml-4">
                    <Button
                      asChild
                      variant="outline"
                      size="sm"
                      className="text-xs h-7"
                    >
                      <Link to="/dashboard/lessons">
                        <ExternalLink className="h-3 w-3 mr-1" />
                        View Plan
                      </Link>
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => deleteLessonPlan(plan.id)}
                      className="text-red-500 hover:text-red-700 p-1 h-auto"
                      title="Delete lesson plan"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
});

UserLessonPlans.displayName = 'UserLessonPlans';

export default UserLessonPlans; 