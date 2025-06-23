import type { D1Database } from '@cloudflare/workers-types'

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
  WORKERS_AI_TOKEN: string;
  WORKERS_AI_ACCOUNT_ID: string;
}

// Combined context type for Hono
export type AppContext = {
  Variables: Variables;
  Bindings: Bindings;
} 