import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import SavedLessonPlans from "@/components/SavedLessonPlans";
import { useUser, useAuth } from '@clerk/clerk-react';
import { useEffect, useState } from 'react';
import { CheckCircle } from 'lucide-react';

import { lessonMeta } from "@/data/lessonMeta";
// lessonMeta is now the source of truth for lesson metadata

interface LessonProgress {
  lesson_id: string;
  completed: number;
  score?: number;
  updated_at: string;
}

export default function Lessons() {
  const { isSignedIn } = useUser();
  const { getToken } = useAuth();
  const [lessonProgress, setLessonProgress] = useState<Record<string, LessonProgress>>({});
  const [progressLoading, setProgressLoading] = useState(false);

  // Fetch progress for all lessons
  useEffect(() => {
    const fetchProgress = async () => {
      if (!isSignedIn) return;

      setProgressLoading(true);
      try {
        const token = await getToken();
        if (!token) return;

        const response = await fetch('/api/d1/user/progress', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          credentials: 'include',
        });

        if (response.ok) {
          const data = await response.json();
          const progressArray = data.progress || [];
          
          // Convert array to object for easy lookup
          const progressMap: Record<string, LessonProgress> = {};
          progressArray.forEach((progress: LessonProgress) => {
            progressMap[progress.lesson_id] = progress;
          });
          
          setLessonProgress(progressMap);
        }
      } catch (error) {
        console.error('Error fetching lesson progress:', error);
      } finally {
        setProgressLoading(false);
      }
    };

    fetchProgress();
  }, [isSignedIn, getToken]);

  return (
    <div className="content-container mx-auto py-12 px-4">
      {/* Saved Lesson Plans Section */}
      {isSignedIn && (
        <div className="mb-12">
          <SavedLessonPlans />
        </div>
      )}

      <h1 className="text-4xl font-bold mb-8">Available Lessons</h1>
      <div className="grid gap-6">
        {lessonMeta.map((lesson: import("@/data/lessonMeta").LessonMeta) => {
          const progress = lessonProgress[lesson.id];
          const isCompleted = progress?.completed === 1;
          
          return (
            <Card key={lesson.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <CardTitle className="text-2xl">{lesson.title}</CardTitle>
                      {isSignedIn && !progressLoading && (
                        <div className="flex items-center gap-2">
                          {isCompleted && (
                            <Badge variant="success" className="flex items-center gap-1">
                              <CheckCircle className="h-3 w-3" />
                              Completed
                            </Badge>
                          )}
                          {progress?.score !== undefined && progress?.score !== null && (
                            <Badge variant={progress.score >= 70 ? "success" : "secondary"}>
                              Quiz: {progress.score}%
                            </Badge>
                          )}
                        </div>
                      )}
                    </div>
                    <CardDescription>{lesson.description}</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {lesson.status === "available" ? (
                  <Button asChild variant="primary">
                    <Link to={`/lessons/${lesson.id}`}>
                      {isCompleted ? 'Review Lesson' : 'Start Lesson'}
                    </Link>
                  </Button>
                ) : (
                  <Button variant="outline" disabled>
                    Coming Soon
                  </Button>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
