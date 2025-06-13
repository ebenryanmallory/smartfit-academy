// Centralized lesson metadata for the Lessons page
export interface LessonMeta {
  id: string;
  title: string;
  description: string;
  status: 'available' | 'coming-soon';
  audienceLevels?: ('elementary' | 'high-school' | 'undergraduate' | 'graduate')[];
}

export const lessonMeta: LessonMeta[] = [
  {
    id: "c-intro-ai",
    title: "Introduction to AI",
    description: "Learn the fundamentals of Artificial Intelligence and its impact on our world.",
    status: "available",
    audienceLevels: ['elementary', 'high-school', 'undergraduate', 'graduate'],
  },
  {
    id: "c-python-fundamentals",
    title: "Programming Fundamentals and Problem Solving",
    description: "Master essential programming concepts, data structures, and algorithmic thinking using Python for computer science applications.",
    status: "available",
    audienceLevels: ['elementary', 'high-school', 'undergraduate', 'graduate'],
  },
  {
    id: "c-machine-learning-fundamentals",
    title: "Machine Learning Fundamentals",
    description: "Comprehensive introduction to machine learning concepts, algorithms, and practical implementation for computer science students.",
    status: "available",
    audienceLevels: ['elementary', 'high-school', 'undergraduate', 'graduate'],
  },
  {
    id: "c-data-science-fundamentals",
    title: "Data Science Fundamentals with Python",
    description: "Comprehensive introduction to data science concepts, Python libraries, statistical analysis, and machine learning.",
    status: "available",
    audienceLevels: ['elementary', 'high-school', 'undergraduate', 'graduate'],
  },
];
