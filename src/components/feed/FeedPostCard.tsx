import React from 'react';
import ReactMarkdown from 'react-markdown';
import { motion } from 'motion/react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { CodeSnippet } from '@/components/CodeSnippet';
import { toast } from 'sonner';
import {
  Lightbulb, Brain, BookOpen, Code2,
  CheckCircle, XCircle, Heart, Sparkles, ArrowRight
} from 'lucide-react';
import type { FeedPost, FeedAction, FeedPostType } from '@/types/feed';

const TYPE_CONFIG: Record<FeedPostType, { label: string; icon: React.ElementType; color: string }> = {
  did_you_know: { label: 'Did you know', icon: Lightbulb, color: '#c7522a' },
  quiz: { label: 'Quiz', icon: Brain, color: '#008585' },
  concept: { label: 'Concept', icon: BookOpen, color: '#74a892' },
  code_snippet: { label: 'Code', icon: Code2, color: '#c7522a' },
};

const TOPIC_LABELS: Record<string, string> = {
  'c-python-fundamentals': 'Python',
  'c-intro-ai': 'Intro to AI',
  'c-machine-learning-fundamentals': 'Machine Learning',
  'c-data-science-fundamentals': 'Data Science',
};

const DIFFICULTY_LABELS: Record<string, string> = {
  intro: 'Intro',
  core: 'Core',
  stretch: 'Stretch',
};

interface FeedPostCardProps {
  post: FeedPost;
  onInteraction: (post: FeedPost, action: FeedAction) => void;
}

export function FeedPostCard({ post, onInteraction }: FeedPostCardProps) {
  const [liked, setLiked] = React.useState(false);
  const [selected, setSelected] = React.useState<number | null>(null);
  const likeRecordedRef = React.useRef(false);
  const viewRecordedRef = React.useRef(false);
  const cardRef = React.useRef<HTMLDivElement>(null);

  const config = TYPE_CONFIG[post.type];
  const TypeIcon = config.icon;
  const answered = selected !== null;

  // Record 'viewed' once, when the card is actually half on screen — not at
  // generation time — so the avoid-repeats signal only covers posts the user saw.
  React.useEffect(() => {
    const el = cardRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !viewRecordedRef.current) {
          viewRecordedRef.current = true;
          onInteraction(post, 'viewed');
          observer.disconnect();
        }
      },
      { threshold: 0.5 }
    );
    observer.observe(el);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleAnswer = (index: number) => {
    if (answered || !post.quiz) return;
    setSelected(index);
    onInteraction(post, index === post.quiz.correctIndex ? 'quiz_correct' : 'quiz_incorrect');
  };

  const handleLike = () => {
    if (!liked && !likeRecordedRef.current) {
      likeRecordedRef.current = true;
      onInteraction(post, 'liked');
    }
    setLiked(!liked);
  };

  const handleMoreLikeThis = () => {
    onInteraction(post, 'more_like_this');
    toast.success("Got it — we'll show you more like this");
  };

  return (
    <motion.div
      ref={cardRef}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className="p-5 border-l-4" style={{ borderLeftColor: config.color }}>
        <div className="flex items-center gap-2 flex-wrap mb-3">
          <Badge className="gap-1 text-white" style={{ backgroundColor: config.color }}>
            <TypeIcon className="h-3 w-3" />
            {config.label}
          </Badge>
          <Badge variant="secondary">{TOPIC_LABELS[post.topic] ?? post.topic}</Badge>
          <Badge variant="outline">{DIFFICULTY_LABELS[post.difficulty] ?? post.difficulty}</Badge>
        </div>

        <h3 className="font-semibold text-lg mb-2">{post.title}</h3>

        <div className="prose prose-sm max-w-none text-gray-700">
          <ReactMarkdown>{post.body}</ReactMarkdown>
        </div>

        {post.type === 'code_snippet' && post.code && (
          <div>
            <CodeSnippet code={post.code.snippet} language={post.code.language} title={post.title} />
            <p className="text-sm text-gray-600 flex items-start gap-2 mt-2">
              <ArrowRight className="h-4 w-4 mt-0.5 flex-shrink-0" style={{ color: config.color }} />
              <span>{post.code.takeaway}</span>
            </p>
          </div>
        )}

        {post.type === 'quiz' && post.quiz && (
          <div className="mt-4 space-y-2">
            <p className="font-medium text-sm">{post.quiz.question}</p>
            {post.quiz.options.map((option, index) => {
              const isCorrect = index === post.quiz!.correctIndex;
              const isSelected = index === selected;
              let resultClass = '';
              if (answered) {
                if (isCorrect) resultClass = 'border-green-500 bg-green-50 text-green-800';
                else if (isSelected) resultClass = 'border-red-500 bg-red-50 text-red-800';
                else resultClass = 'opacity-60';
              }
              return (
                <Button
                  key={index}
                  variant="outline"
                  disabled={answered}
                  onClick={() => handleAnswer(index)}
                  className={`w-full justify-start text-left h-auto py-2 whitespace-normal disabled:opacity-100 ${resultClass}`}
                >
                  <span className="flex items-start gap-2 w-full">
                    <span className="font-semibold flex-shrink-0">{String.fromCharCode(65 + index)}.</span>
                    <span className="flex-1">{option}</span>
                    {answered && isCorrect && <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />}
                    {answered && isSelected && !isCorrect && <XCircle className="h-4 w-4 text-red-600 flex-shrink-0 mt-0.5" />}
                  </span>
                </Button>
              );
            })}
            {answered && (
              <div role="status" aria-live="polite" className={`text-sm p-3 rounded border-l-4 ${selected === post.quiz.correctIndex ? 'bg-green-50 border-l-green-500 text-green-900' : 'bg-red-50 border-l-red-500 text-red-900'}`}>
                <p className="font-medium mb-1">
                  {selected === post.quiz.correctIndex ? 'Correct!' : 'Not quite.'}
                </p>
                <p>{post.quiz.explanation}</p>
              </div>
            )}
          </div>
        )}

        <div className="flex items-center justify-between mt-4 pt-3 border-t">
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleLike}
              className={liked ? 'text-[#c7522a]' : 'text-gray-500'}
            >
              <Heart className={`h-4 w-4 mr-1 ${liked ? 'fill-current' : ''}`} />
              {liked ? 'Liked' : 'Like'}
            </Button>
            <Button variant="ghost" size="sm" onClick={handleMoreLikeThis} className="text-gray-500">
              <Sparkles className="h-4 w-4 mr-1" />
              More like this
            </Button>
          </div>
          <div className="flex gap-1 flex-wrap justify-end">
            {post.tags.slice(0, 3).map(tag => (
              <Badge key={tag} variant="secondary" className="text-xs font-normal lowercase">
                {tag}
              </Badge>
            ))}
          </div>
        </div>
      </Card>
    </motion.div>
  );
}
