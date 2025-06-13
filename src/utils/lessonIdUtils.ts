import { customAlphabet } from 'nanoid';

// Create a custom nanoid function for shorter, URL-friendly IDs
// Using only lowercase letters and numbers for consistency
const nanoid = customAlphabet('0123456789abcdefghijklmnopqrstuvwxyz', 8);

// Lesson ID prefixes for different content types
export const LESSON_ID_PREFIXES = {
  CUSTOM: 'c-',      // Custom/hardcoded lessons (c-intro-ai, c-python-basics)
  GENERATED: 'g-',   // AI-generated lessons (g-abcd-efgh)
} as const;

export type LessonIdPrefix = typeof LESSON_ID_PREFIXES[keyof typeof LESSON_ID_PREFIXES];

// Generate ID for user-generated content
export function generateUserLessonId(): string {
  const id = nanoid();
  return `${LESSON_ID_PREFIXES.GENERATED}${id.slice(0, 4)}-${id.slice(4)}`;
}

// Generate ID for custom/hardcoded content
export function generateCustomLessonId(slug: string): string {
  // Ensure slug is URL-safe and follows our conventions
  const cleanSlug = slug
    .toLowerCase()
    .replace(/[^a-z0-9-]/g, '-')  // Replace non-alphanumeric with dashes
    .replace(/-+/g, '-')          // Replace multiple dashes with single dash
    .replace(/^-|-$/g, '');       // Remove leading/trailing dashes
  
  return `${LESSON_ID_PREFIXES.CUSTOM}${cleanSlug}`;
}

// Check if a lesson ID is custom/hardcoded content
export function isCustomLesson(id: string): boolean {
  return id.startsWith(LESSON_ID_PREFIXES.CUSTOM);
}

// Check if a lesson ID is user-generated content
export function isGeneratedLesson(id: string): boolean {
  return id.startsWith(LESSON_ID_PREFIXES.GENERATED);
}

// Get lesson type from ID
export function getLessonType(id: string): 'custom' | 'generated' {
  if (isCustomLesson(id)) return 'custom';
  if (isGeneratedLesson(id)) return 'generated';
  
  throw new Error(`Invalid lesson ID format: ${id}. Expected format: c-[slug] or g-[xxxx-xxxx]`);
}

// Check if lesson supports education level switching (custom lessons do, generated don't)
export function supportsEducationLevelSwitching(id: string): boolean {
  const type = getLessonType(id);
  return type === 'custom';
}

// Extract the slug from a custom lesson ID
export function getCustomLessonSlug(id: string): string | null {
  if (!isCustomLesson(id)) return null;
  return id.substring(LESSON_ID_PREFIXES.CUSTOM.length);
}

// Validate lesson ID format
export function isValidLessonId(id: string): boolean {
  // Custom format: c-[slug]
  if (/^c-[a-z0-9-]+$/.test(id)) return true;
  
  // Generated format: g-[4chars]-[4chars]
  if (/^g-[a-z0-9]{4}-[a-z0-9]{4}$/.test(id)) return true;
  
  return false;
}

// Examples and documentation
export const LESSON_ID_EXAMPLES = {
  custom: ['c-intro-ai', 'c-python-basics', 'c-machine-learning', 'c-data-science'],
  generated: ['g-abcd-efgh', 'g-1234-5678', 'g-xyz9-abc3']
} as const;

// URL patterns for routing
export const LESSON_URL_PATTERNS = {
  custom: /^\/lessons\/(c-[a-z0-9-]+)$/,
  generated: /^\/lessons\/(g-[a-z0-9]{4}-[a-z0-9]{4})$/
} as const; 