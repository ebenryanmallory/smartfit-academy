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
