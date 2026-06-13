import React, { useCallback, useEffect, useState } from 'react';
import { useAuth } from '@clerk/react';
import { toast } from 'sonner';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Plus } from 'lucide-react';
import EducationLevelSelector from '@/components/EducationLevelSelector';
import { FEED_TOPIC_OPTIONS, FEED_POST_TYPE_OPTIONS, type FeedPreferences } from '@/types/feed';
import { fetchTopicSuggestions } from '@/utils/topicSuggestions';

const MAX_CUSTOM_TOPICS = 5;
const MAX_CUSTOM_TOPIC_LENGTH = 60;
const ALL_TOPIC_IDS = FEED_TOPIC_OPTIONS.map(t => t.id);
const ALL_POST_TYPE_IDS = FEED_POST_TYPE_OPTIONS.map(t => t.id);
const CATALOG_TOPIC_NAMES = new Set(
  FEED_TOPIC_OPTIONS.flatMap(t => [t.id.toLowerCase(), t.label.toLowerCase()])
);

interface FeedPreferencesDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSaved: () => void;
}

function TogglePill({ selected, onClick, children }: { selected: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      className={`px-3 py-1.5 rounded-full text-sm border transition-colors ${
        selected
          ? 'bg-teal-700 border-teal-700 text-white'
          : 'bg-white border-gray-300 text-gray-500 hover:border-gray-400'
      }`}
    >
      {children}
    </button>
  );
}

// Mute model: everything is shown by default. All catalog topics and all post
// types start active; deselecting mutes, and at least one of each must stay on.
export function FeedPreferencesDialog({ open, onOpenChange, onSaved }: FeedPreferencesDialogProps) {
  const { getToken } = useAuth();
  const [topics, setTopics] = useState<string[]>(ALL_TOPIC_IDS);
  const [customTopics, setCustomTopics] = useState<string[]>([]);
  const [postTypes, setPostTypes] = useState<string[]>(ALL_POST_TYPE_IDS);
  const [customInput, setCustomInput] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [llmSuggestions, setLlmSuggestions] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Seed state from saved preferences (empty stored arrays mean "all"), and
  // offer dashboard saved topics as one-tap suggestions.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    (async () => {
      setCustomInput('');
      setLlmSuggestions([]);
      setIsLoading(true);
      try {
        const token = await getToken();
        const headers = { 'Authorization': `Bearer ${token}` };
        const [prefsRes, topicsRes] = await Promise.all([
          fetch('/api/d1/user/feed/preferences', { headers }),
          fetch('/api/d1/user/topics', { headers }),
        ]);
        if (cancelled) return;
        if (prefsRes.ok) {
          const data: { preferences?: FeedPreferences } = await prefsRes.json();
          const saved = data.preferences;
          setTopics(saved?.topics?.length ? saved.topics : ALL_TOPIC_IDS);
          setCustomTopics(saved?.customTopics ?? []);
          setPostTypes(saved?.postTypes?.length ? saved.postTypes : ALL_POST_TYPE_IDS);
        }
        if (topicsRes.ok) {
          const data: { topics?: { topic: string }[] } = await topicsRes.json();
          setSuggestions(
            (data.topics ?? [])
              .map(t => t.topic)
              .filter(t =>
                t.length >= 2 &&
                t.length <= MAX_CUSTOM_TOPIC_LENGTH &&
                !CATALOG_TOPIC_NAMES.has(t.toLowerCase())
              )
          );
        }
      } catch (err) {
        console.error('Failed to load feed preferences:', err);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [open, getToken]);

  const activeTopicCount = topics.length + customTopics.length;

  const toggleTopic = (id: string) => {
    if (topics.includes(id)) {
      if (activeTopicCount <= 1) {
        toast.error('Keep at least one topic active');
        return;
      }
      setTopics(topics.filter(t => t !== id));
    } else {
      setTopics([...topics, id]);
    }
  };

  const removeCustomTopic = (topic: string) => {
    if (activeTopicCount <= 1) {
      toast.error('Keep at least one topic active');
      return;
    }
    setCustomTopics(prev => prev.filter(t => t !== topic));
  };

  const togglePostType = (id: string) => {
    if (postTypes.includes(id)) {
      if (postTypes.length <= 1) {
        toast.error('Keep at least one post type active');
        return;
      }
      setPostTypes(postTypes.filter(t => t !== id));
    } else {
      setPostTypes([...postTypes, id]);
    }
  };

  const addCustomTopic = useCallback((raw: string) => {
    const cleaned = raw.replace(/\s+/g, ' ').trim();
    if (cleaned.length < 2 || cleaned.length > MAX_CUSTOM_TOPIC_LENGTH) return;
    setCustomTopics(prev => {
      if (prev.length >= MAX_CUSTOM_TOPICS) {
        toast.error(`You can add up to ${MAX_CUSTOM_TOPICS} custom topics`);
        return prev;
      }
      if (prev.some(t => t.toLowerCase() === cleaned.toLowerCase())) return prev;
      return [...prev, cleaned];
    });
  }, []);

  // Free text never becomes a topic directly — it's sent to the LLM (same
  // flow as the dashboard assistant), and the user picks from the vetted
  // educational topics it returns.
  const handleSuggest = async () => {
    const query = customInput.trim();
    if (query.length < 2 || isSuggesting) return;
    setIsSuggesting(true);
    try {
      const { topics: suggested } = await fetchTopicSuggestions(
        [{ role: 'user', content: query }]
      );
      const usable = suggested.filter(t =>
        t.length >= 2 &&
        t.length <= MAX_CUSTOM_TOPIC_LENGTH &&
        !CATALOG_TOPIC_NAMES.has(t.toLowerCase())
      );
      setLlmSuggestions(usable);
      if (usable.length === 0) {
        toast.error('No topic suggestions found — try rephrasing');
      }
    } catch (err) {
      console.error('Failed to fetch topic suggestions:', err);
      toast.error(err instanceof Error && err.message ? err.message : 'Could not get topic suggestions. Please try again.');
    } finally {
      setIsSuggesting(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const token = await getToken();
      const response = await fetch('/api/d1/user/feed/preferences', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ topics, customTopics, postTypes }),
      });
      if (!response.ok) throw new Error(`Save failed (${response.status})`);
      toast.success('Feed preferences saved');
      onOpenChange(false);
      onSaved();
    } catch (err) {
      console.error('Failed to save feed preferences:', err);
      toast.error('Could not save preferences. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  const notAlreadyAdded = (s: string) => !customTopics.some(t => t.toLowerCase() === s.toLowerCase());
  const visibleSuggestions = suggestions.filter(notAlreadyAdded);
  const visibleLlmSuggestions = llmSuggestions.filter(notAlreadyAdded);

  function SuggestionPill({ topic }: { topic: string }) {
    return (
      <button
        type="button"
        onClick={() => addCustomTopic(topic)}
        className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-sm bg-white border border-dashed border-gray-400 text-gray-600 hover:border-teal-700 hover:text-teal-700 transition-colors"
      >
        <Plus className="h-3 w-3" />
        {topic}
      </button>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md max-h-[85vh] overflow-y-auto bg-white text-gray-900">
        <DialogHeader>
          <DialogTitle>Feed preferences</DialogTitle>
          <DialogDescription className="text-gray-500">
            Everything is on by default. Deselect anything you'd rather not see — at least one topic and one post type stay active.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5">
          <div>
            <Label className="mb-2 block">Topics</Label>
            <div className="flex flex-wrap gap-2">
              {FEED_TOPIC_OPTIONS.map(t => (
                <TogglePill
                  key={t.id}
                  selected={topics.includes(t.id)}
                  onClick={() => toggleTopic(t.id)}
                >
                  {t.label}
                </TogglePill>
              ))}
              {customTopics.map(t => (
                <TogglePill key={t} selected onClick={() => removeCustomTopic(t)}>
                  {t}
                </TogglePill>
              ))}
            </div>

            {visibleSuggestions.length > 0 && customTopics.length < MAX_CUSTOM_TOPICS && (
              <div className="mt-3">
                <p className="text-xs text-gray-500 mb-1">From your dashboard topics:</p>
                <div className="flex flex-wrap gap-2">
                  {visibleSuggestions.map(s => <SuggestionPill key={s} topic={s} />)}
                </div>
              </div>
            )}

            <div className="flex gap-2 mt-3">
              <Input
                value={customInput}
                maxLength={MAX_CUSTOM_TOPIC_LENGTH}
                placeholder="What would you like to learn about?"
                onChange={e => setCustomInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleSuggest();
                  }
                }}
              />
              <Button
                type="button"
                variant="outline"
                onClick={handleSuggest}
                disabled={customInput.trim().length < 2 || isSuggesting || customTopics.length >= MAX_CUSTOM_TOPICS}
              >
                {isSuggesting ? 'Finding…' : 'Suggest'}
              </Button>
            </div>
            {visibleLlmSuggestions.length > 0 && customTopics.length < MAX_CUSTOM_TOPICS && (
              <div className="mt-2">
                <p className="text-xs text-gray-500 mb-1">Pick a suggested topic:</p>
                <div className="flex flex-wrap gap-2">
                  {visibleLlmSuggestions.map(s => <SuggestionPill key={s} topic={s} />)}
                </div>
              </div>
            )}
          </div>

          <div>
            <Label className="mb-2 block">Post types</Label>
            <div className="flex flex-wrap gap-2">
              {FEED_POST_TYPE_OPTIONS.map(t => (
                <TogglePill
                  key={t.id}
                  selected={postTypes.includes(t.id)}
                  onClick={() => togglePostType(t.id)}
                >
                  {t.label}
                </TogglePill>
              ))}
            </div>
          </div>

          <EducationLevelSelector variant="select" className="mb-0" />
        </div>

        <DialogFooter>
          <Button onClick={handleSave} disabled={isSaving || isLoading}>
            {isSaving ? 'Saving…' : 'Save preferences'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
