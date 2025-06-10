// Centralized lesson metadata for the Lessons page
export interface LessonMeta {
  id: number;
  title: string;
  description: string;
  status: 'available' | 'coming-soon';
  audienceLevels?: ('elementary' | 'high-school' | 'undergraduate' | 'graduate')[];
}

export const lessonMeta: LessonMeta[] = [
  {
    id: 1,
    title: "Introduction to AI",
    description: "Learn the fundamentals of Artificial Intelligence and its impact on our world.",
    status: "available",
    audienceLevels: ['elementary', 'high-school', 'undergraduate', 'graduate'],
  },
  {
    id: 2,
    title: "Getting Started with Programming",
    description: "Learn the basics of programming with Python, from variables to functions.",
    status: "available",
    audienceLevels: ['undergraduate'],
  },
  {
    id: 3,
    title: "Machine Learning Basics",
    description: "Explore the core concepts of machine learning and how it powers modern AI systems.",
    status: "available",
    audienceLevels: ['undergraduate'],
  },
  {
    id: 4,
    title: "Introduction to Data Science with Python",
    description: "Explore the basics of data science, including data manipulation, analysis, and visualization using Python.",
    status: "available",
    audienceLevels: ['undergraduate'],
  },
];
