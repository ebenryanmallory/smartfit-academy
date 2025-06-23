import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import SavedLessonPlans from "@/components/SavedLessonPlans";
import { useUser } from '@clerk/clerk-react';

import { lessonMeta } from "@/data/lessonMeta";
// lessonMeta is now the source of truth for lesson metadata

export default function Lessons() {
  const { isSignedIn } = useUser();

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
        {lessonMeta.map((lesson: import("@/data/lessonMeta").LessonMeta) => (
          <Card key={lesson.id} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <CardTitle className="text-2xl">{lesson.title}</CardTitle>
              <CardDescription>{lesson.description}</CardDescription>
            </CardHeader>
            <CardContent>
              {lesson.status === "available" ? (
                <Button asChild variant="primary">
                  <Link to={`/lessons/${lesson.id}`}>Start Lesson</Link>
                </Button>
              ) : (
                <Button variant="outline" disabled>
                  Coming Soon
                </Button>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
