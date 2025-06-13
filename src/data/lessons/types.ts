export interface LessonSection {
  title: string;
  content: string;
}

export interface LessonData {
  id: string;
  title: string;
  description: string;
  sections: LessonSection[];
} 