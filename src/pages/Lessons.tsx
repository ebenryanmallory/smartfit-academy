import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const lessons = [
  {
    id: 1,
    title: "Introduction to AI",
    description: "Learn the fundamentals of Artificial Intelligence and its impact on our world.",
    status: "available",
  },
  {
    id: 2,
    title: "Getting Started with Programming",
    description: "Learn the basics of programming with Python, from variables to functions.",
    status: "available",
  },
  {
    id: 3,
    title: "Machine Learning Basics",
    description: "Explore the core concepts of machine learning and how it powers modern AI systems.",
    status: "coming-soon",
  },
  {
    id: 4,
    title: "Neural Networks Deep Dive",
    description: "Master the architecture and implementation of neural networks for AI applications.",
    status: "coming-soon",
  },
];

export default function Lessons() {
  return (
    <div className="container mx-auto py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-8">Available Lessons</h1>
        <div className="grid gap-6">
          {lessons.map((lesson) => (
            <Card key={lesson.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="text-2xl">{lesson.title}</CardTitle>
                <CardDescription>{lesson.description}</CardDescription>
              </CardHeader>
              <CardContent>
                {lesson.status === "available" ? (
                  <Button asChild>
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
    </div>
  );
}
