export interface LessonSection {
  title: string;
  content: string;
}

export interface LessonData {
  id: number;
  title: string;
  description: string;
  sections: LessonSection[];
} 