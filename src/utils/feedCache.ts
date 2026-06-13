import type { FeedPost } from '@/types/feed';

export interface FeedPostState {
  liked: boolean;
  likeRecorded: boolean; // analytics dedup survives like → unlike → restore
  quizSelected: number | null;
  viewed: boolean;
}

export interface FeedCache {
  posts: FeedPost[];
  postStates: Record<string, FeedPostState>; // keyed by client post id
  seenTags: string[];
}

const MAX_POSTS = 60;

// Keyed by Clerk user id: sessionStorage survives sign-out/sign-in within a
// tab, so an unscoped key would hydrate one user's feed for another.
// Versioning is via the v1 suffix — a shape change bumps the key and old
// entries simply miss.
const cacheKey = (userId: string) => `bf:feed:v1:${userId}`;

export function loadFeedCache(userId: string): FeedCache | null {
  try {
    const raw = sessionStorage.getItem(cacheKey(userId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as FeedCache;
    if (
      !Array.isArray(parsed?.posts) ||
      typeof parsed?.postStates !== 'object' || parsed.postStates === null ||
      !Array.isArray(parsed?.seenTags)
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null; // any failure is a cache miss
  }
}

export function saveFeedCache(userId: string, cache: FeedCache): void {
  try {
    // The array is not chronological once refresh prepends (newest posts sit
    // at the top), so keep the first MAX_POSTS — what the user lands on when
    // they return — and evict the deepest-scrolled bottom posts.
    const posts = cache.posts.slice(0, MAX_POSTS);
    const kept = new Set(posts.map(p => p.id));
    const postStates = Object.fromEntries(
      Object.entries(cache.postStates).filter(([id]) => kept.has(id))
    );
    sessionStorage.setItem(
      cacheKey(userId),
      JSON.stringify({ posts, postStates, seenTags: cache.seenTags })
    );
  } catch {
    // Quota or serialization failure — the feed still works, just uncached.
  }
}

export function clearFeedCache(userId: string): void {
  try {
    sessionStorage.removeItem(cacheKey(userId));
  } catch {
    // ignore
  }
}
