import type { D1Database, Fetcher } from '@cloudflare/workers-types'

// Define the user type
export interface User {
  id: string;
  emailAddresses: Array<{ emailAddress: string }>;
  firstName: string | null;
  lastName: string | null;
}

// Extend the Hono context type
export type Variables = {
  user: User;
}

// Define the bindings type
export type Bindings = {
  DB: D1Database;
  ASSETS: Fetcher;
  WORKERS_AI_TOKEN: string;
  WORKERS_AI_ACCOUNT_ID: string;
  CLAUDE_API_KEY: string;
}

// Combined context type for Hono
export type AppContext = {
  Variables: Variables;
  Bindings: Bindings;
} 