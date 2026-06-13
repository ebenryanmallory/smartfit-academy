export type FeedPostType = 'did_you_know' | 'quiz' | 'concept' | 'code_snippet';

export type FeedDifficulty = 'intro' | 'core' | 'stretch';

export interface FeedQuiz {
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}

export interface FeedCode {
  language: 'python';
  snippet: string;
  takeaway: string;
}

export interface FeedPost {
  id: string; // client-assigned on receipt; posts are ephemeral and have no server id
  type: FeedPostType;
  topic: string;
  title: string;
  body: string; // markdown
  tags: string[];
  difficulty: FeedDifficulty;
  quiz?: FeedQuiz;
  code?: FeedCode;
}

export type FeedAction = 'viewed' | 'liked' | 'more_like_this' | 'quiz_correct' | 'quiz_incorrect';

// Mirrors the worker's FEED_TOPICS catalog (worker/instructions/feed-post-generator.ts);
// shared by the post card badges and the preferences dialog.
export const FEED_TOPIC_OPTIONS: { id: string; label: string }[] = [
  { id: 'c-python-fundamentals', label: 'Python' },
  { id: 'c-intro-ai', label: 'Intro to AI' },
  { id: 'c-machine-learning-fundamentals', label: 'Machine Learning' },
  { id: 'c-data-science-fundamentals', label: 'Data Science' },
];

export const FEED_POST_TYPE_OPTIONS: { id: FeedPostType; label: string }[] = [
  { id: 'did_you_know', label: 'Did you know' },
  { id: 'quiz', label: 'Quiz' },
  { id: 'concept', label: 'Concept' },
  { id: 'code_snippet', label: 'Code' },
];

export interface FeedPreferences {
  topics: string[];
  customTopics: string[];
  postTypes: string[];
}
