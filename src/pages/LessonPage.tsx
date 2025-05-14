import { useParams, useNavigate } from 'react-router-dom';
import { LessonViewer } from "@/components/LessonViewer";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import type { LessonData, LessonSection } from "@/data/lessons/lesson-1";

export default function LessonPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [lesson, setLesson] = useState<LessonData | undefined>(undefined);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!id) {
      setLesson(undefined);
      return;
    }
    setLoading(true);
    import(`@/data/lessons/lesson-${id}.ts`)
      .then(mod => {
        // Try named export lesson{id}, fallback to default
        const lessonData: LessonData = mod[`lesson${id}`] || mod.default;
        setLesson(lessonData);
      })
      .catch(() => setLesson(undefined))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="container mx-auto py-12 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl font-bold mb-4">Loading Lesson...</h1>
        </div>
      </div>
    );
  }

  if (!lesson) {
    return (
      <div className="container mx-auto py-12 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl font-bold mb-4">Lesson Not Found</h1>
          <p className="text-muted-foreground mb-8">
            The lesson you're looking for doesn't exist or isn't available yet.
          </p>
          <Button onClick={() => navigate('/lessons')}>
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

  return <LessonViewer title={lesson.title} description={lesson.description} content={combinedContent} />;
}