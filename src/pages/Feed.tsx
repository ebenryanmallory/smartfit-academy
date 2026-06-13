import React, { useCallback, useEffect, useRef, useState } from 'react';
import { RedirectToSignIn, useAuth, useUser, useSession } from '@clerk/react';
import { toast } from 'sonner';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { RefreshCw, SlidersHorizontal } from 'lucide-react';
import { FeedPostCard } from '@/components/feed/FeedPostCard';
import { FeedPreferencesDialog } from '@/components/feed/FeedPreferencesDialog';
import type { FeedPost, FeedAction } from '@/types/feed';
import { loadFeedCache, saveFeedCache, clearFeedCache, type FeedPostState } from '@/utils/feedCache';

function PostSkeleton() {
  return (
    <Card className="p-5 animate-pulse">
      <div className="flex gap-2 mb-3">
        <div className="h-5 w-24 bg-gray-200 rounded-full" />
        <div className="h-5 w-20 bg-gray-200 rounded-full" />
      </div>
      <div className="h-5 w-3/4 bg-gray-200 rounded mb-3" />
      <div className="space-y-2">
        <div className="h-3 w-full bg-gray-200 rounded" />
        <div className="h-3 w-full bg-gray-200 rounded" />
        <div className="h-3 w-2/3 bg-gray-200 rounded" />
      </div>
    </Card>
  );
}

function FeedContent() {
  const { getToken, isLoaded: isAuthLoaded } = useAuth();
  const { session } = useSession();
  const { user } = useUser();
  const userId = user?.id ?? '';
  // Read once on mount; a cache hit renders the previous session's posts
  // immediately and skips the initial generation call.
  const [initialCache] = useState(() => loadFeedCache(userId));
  const hydrated = (initialCache?.posts.length ?? 0) > 0;
  const [posts, setPosts] = useState<FeedPost[]>(initialCache?.posts ?? []);
  const [postStates, setPostStates] = useState<Record<string, FeedPostState>>(initialCache?.postStates ?? {});
  const [isLoading, setIsLoading] = useState(!hydrated);
  const [isFetchingMore, setIsFetchingMore] = useState(false);
  const [isPrepending, setIsPrepending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rateLimited, setRateLimited] = useState(false);
  const [prefsOpen, setPrefsOpen] = useState(false);
  // With a short cached feed the bottom sentinel can intersect on first paint
  // and trigger a paid append; hold auto-append until the user signals scroll
  // intent. State (not a ref) so flipping it recreates the observer and an
  // already-intersecting sentinel fires immediately on enable.
  const [autoAppendEnabled, setAutoAppendEnabled] = useState(!hydrated);
  const seenTagsRef = useRef<Set<string>>(new Set(initialCache?.seenTags ?? []));
  const inFlightRef = useRef(false);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const recordInteractions = useCallback(async (events: { postType: string; topic: string; tags: string[]; action: FeedAction; difficulty: string }[]) => {
    try {
      const token = await getToken();
      await fetch('/api/d1/user/feed/interactions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        credentials: 'include',
        body: JSON.stringify({ events }),
      });
    } catch (err) {
      console.error('Failed to record feed interactions:', err);
    }
  }, [getToken]);

  const handleInteraction = useCallback((post: FeedPost, action: FeedAction) => {
    recordInteractions([{
      postType: post.type,
      topic: post.topic,
      tags: post.tags,
      action,
      difficulty: post.difficulty,
    }]);
  }, [recordInteractions]);

  const fetchBatch = useCallback(async (placement: 'initial' | 'append' | 'prepend') => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    if (placement === 'initial') setIsLoading(true);
    else if (placement === 'append') setIsFetchingMore(true);
    else setIsPrepending(true);
    setError(null);

    try {
      const token = await getToken();
      const response = await fetch('/claude/feed', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        credentials: 'include',
        body: JSON.stringify({ excludeTags: [...seenTagsRef.current].slice(-60) }),
      });

      if (response.status === 429) {
        setRateLimited(true);
        return;
      }
      if (!response.ok || !response.body) {
        throw new Error(`Feed generation failed (${response.status})`);
      }

      // NDJSON stream: one {"post": ...} line per post, appended as it arrives
      // so the first post renders while the rest are still generating.
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let received = 0;
      let streamError: string | null = null;

      const handleLine = (line: string) => {
        if (!line.trim()) return;
        let msg: { post?: Omit<FeedPost, 'id'>; error?: string };
        try {
          msg = JSON.parse(line);
        } catch {
          return;
        }
        if (msg.post) {
          const post: FeedPost = { ...msg.post, id: crypto.randomUUID() };
          post.tags.forEach(t => seenTagsRef.current.add(t));
          if (placement === 'prepend') {
            // Insert at the batch's running index (received) so the new batch
            // reads top-to-bottom in generation order, above existing posts.
            const index = received;
            setPosts(prev => [...prev.slice(0, index), post, ...prev.slice(index)]);
          } else {
            setPosts(prev => [...prev, post]);
          }
          received++;
          if (received === 1 && placement !== 'prepend') {
            // First post is on screen; swap the full-page skeletons for the
            // bottom "generating more" skeleton while the batch finishes.
            setIsLoading(false);
            setIsFetchingMore(true);
          }
        } else if (msg.error) {
          streamError = msg.error;
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        lines.forEach(handleLine);
      }
      handleLine(buffer);

      if (received === 0) {
        throw new Error(streamError || 'No posts were generated');
      }
      // 'viewed' events are recorded per-card when a post actually scrolls into view
    } catch (err) {
      console.error('Feed fetch error:', err);
      if (placement === 'prepend') {
        // Don't set `error`: it gates the sentinel observer and renders the
        // bottom Retry (append) button — neither is a sensible consequence of
        // a failed top refresh. The refresh button is the retry affordance.
        toast.error('Could not generate more posts — try again in a moment');
      } else {
        setError("We couldn't generate posts right now.");
        if (placement === 'append') {
          toast.error('Could not generate more posts — try again in a moment');
        }
      }
    } finally {
      inFlightRef.current = false;
      setIsLoading(false);
      setIsFetchingMore(false);
      setIsPrepending(false);
    }
  }, [getToken]);

  useEffect(() => {
    document.title = 'Feed View';
  }, []);

  useEffect(() => {
    if (isAuthLoaded && session && !hydrated) fetchBatch('initial');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthLoaded, session]);

  // One-time scroll-intent listener that enables auto-append after hydration.
  // Not `scroll`: it never fires if the cached feed is too short to scroll.
  useEffect(() => {
    if (autoAppendEnabled) return;
    const enable = () => setAutoAppendEnabled(true);
    window.addEventListener('wheel', enable, { passive: true });
    window.addEventListener('touchmove', enable, { passive: true });
    window.addEventListener('keydown', enable);
    return () => {
      window.removeEventListener('wheel', enable);
      window.removeEventListener('touchmove', enable);
      window.removeEventListener('keydown', enable);
    };
  }, [autoAppendEnabled]);

  // Write-through: returning to /feed in this tab session restores the feed
  // (and per-post state) without a new generation call.
  useEffect(() => {
    if (posts.length === 0) return;
    saveFeedCache(userId, { posts, postStates, seenTags: [...seenTagsRef.current] });
  }, [posts, postStates, userId]);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel || !autoAppendEnabled) return;
    const observer = new IntersectionObserver(
      (entries) => {
        // Generous rootMargin starts the next LLM call before the user reaches the bottom
        if (entries[0].isIntersecting && !inFlightRef.current && posts.length > 0 && !error && !rateLimited) {
          fetchBatch('append');
        }
      },
      { rootMargin: '600px' }
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [fetchBatch, posts.length, error, rateLimited, autoAppendEnabled]);

  // New preferences should show immediately: reset the session and regenerate.
  const handlePrefsSaved = useCallback(() => {
    clearFeedCache(userId);
    setPosts([]);
    setPostStates({});
    seenTagsRef.current.clear();
    setRateLimited(false);
    setError(null);
    fetchBatch('initial');
  }, [fetchBatch, userId]);

  const handlePostStateChange = useCallback((postId: string, change: Partial<FeedPostState>) => {
    setPostStates(prev => ({
      ...prev,
      [postId]: {
        liked: false,
        likeRecorded: false,
        quizSelected: null,
        viewed: false,
        ...prev[postId],
        ...change,
      },
    }));
  }, []);

  const handleRefresh = useCallback(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    fetchBatch('prepend');
  }, [fetchBatch]);

  return (
    <div className="max-w-xl mx-auto px-4 py-8">
      <div className="mb-6 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold">Feed</h1>
          <p className="text-gray-500 text-sm">Bite-sized lessons, generated for you</p>
        </div>
        <div className="flex items-center">
          <Button
            variant="ghost"
            size="sm"
            aria-label="Refresh feed"
            disabled={isLoading || isFetchingMore || isPrepending}
            onClick={handleRefresh}
            className="text-gray-500"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            aria-label="Feed preferences"
            onClick={() => setPrefsOpen(true)}
            className="text-gray-500"
          >
            <SlidersHorizontal className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <FeedPreferencesDialog open={prefsOpen} onOpenChange={setPrefsOpen} onSaved={handlePrefsSaved} />

      {isLoading && (
        <div className="space-y-4">
          <PostSkeleton />
          <PostSkeleton />
          <PostSkeleton />
        </div>
      )}

      {!isLoading && error && posts.length === 0 && (
        <Card className="p-8 text-center">
          <p className="text-gray-600 mb-4">We couldn't generate your feed right now. This is usually temporary.</p>
          <Button onClick={() => fetchBatch('initial')}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Try again
          </Button>
        </Card>
      )}

      {!isLoading && rateLimited && posts.length === 0 && (
        <Card className="p-8 text-center">
          <p className="text-gray-600">You've reached your hourly feed limit on the free plan. Check back in a bit, or upgrade for unlimited posts.</p>
        </Card>
      )}

      <div className="space-y-4">
        {isPrepending && <PostSkeleton />}
        {posts.map(post => (
          <FeedPostCard
            key={post.id}
            post={post}
            onInteraction={handleInteraction}
            cachedState={postStates[post.id]}
            onStateChange={handlePostStateChange}
          />
        ))}
      </div>

      <div ref={sentinelRef} className="py-6">
        {isFetchingMore && (
          <div className="space-y-4">
            <PostSkeleton />
            <p className="text-center text-sm text-gray-500">Generating more posts…</p>
          </div>
        )}
        {!isFetchingMore && error && posts.length > 0 && (
          <div className="text-center">
            <Button variant="outline" onClick={() => fetchBatch('append')}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        )}
        {!isFetchingMore && rateLimited && posts.length > 0 && (
          <p className="text-center text-sm text-gray-500">
            That's it for this hour on the free plan — check back soon, or upgrade for unlimited posts.
          </p>
        )}
      </div>
    </div>
  );
}

function Feed() {
  const { isLoaded, isSignedIn } = useUser();

  if (!isLoaded) return null;
  if (!isSignedIn) return <RedirectToSignIn />;
  return <FeedContent />;
}

export default Feed;
