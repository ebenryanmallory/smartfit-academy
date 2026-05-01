# Clerk Upgrade Plan — 2026-04-30

## Goal
Upgrade all Clerk packages to latest major versions:
- `@clerk/clerk-react`: `5.31.2` → `5.61.6` (minor bump, same major)
- `@clerk/backend`: `1.32.0` → `3.4.3` (two major versions)
- `@hono/clerk-auth`: `2.0.0` → `3.1.1` (one major version)

**Important nuance:** `@hono/clerk-auth@3.x` depends on `@clerk/backend@^2.x`, NOT `^3.x`.
So the actual installed `@clerk/backend` will be `3.4.3` (listed in package.json) but
`@hono/clerk-auth` will pull in `@clerk/backend@2.x` as its own dependency. These will coexist.
The project's direct `@clerk/backend` dep is not directly used in code — the worker only imports
from `@hono/clerk-auth`. So upgrading `@clerk/backend` in package.json to `^3` is fine but
won't affect the worker code paths (those go through hono's bundled clerk/backend@2).

---

## Runtime Context

- **Backend:** Cloudflare Workers (Hono framework) — NOT Node.js
- **Frontend:** React + Vite (SPA)
- **Node.js requirement for @clerk/backend@3:** Node 20.9.0+ — irrelevant for CF Workers
- **Current Node.js (local tooling):** v25.2.1 ✓

---

## Breaking Changes Summary

### @clerk/clerk-react 5.31.2 → 5.61.6
**Low risk.** Same major version, no breaking API changes that affect this codebase.
- `useOrganizations` → `useOrganization` (singular) — not used here
- `useSyncExternalStore` internal refactor — transparent to consumers
- `setSession()` → `setActive()` — not used here
- **No changes needed to existing code.**

### @clerk/backend 1.x → 3.x (two major hops)
The project does **not** call `@clerk/backend` directly in any worker route.
All backend auth goes through `@hono/clerk-auth` which bundles its own `@clerk/backend@2`.
Direct `@clerk/backend` in package.json is a top-level dep but unused in code.
**No code changes needed.** Just bump the version in package.json.

Key API changes (for awareness, not needed here):
- `Clerk` constructor → `createClerkClient` (not used)
- List methods now return `{ data, totalCount }` instead of arrays (not used)
- `profileImageUrl` / `logoUrl` → `imageUrl` (not used)
- `verifySecret()` / `verifyToken()` → `.verify()` (not used)

### @hono/clerk-auth 2.0.0 → 3.1.1
**Low risk.** The API surface used in this project is stable:
- `clerkMiddleware()` — unchanged
- `getAuth(c)` — unchanged  
- `getAuth(c, { acceptsToken: 'api_key' })` — new option added in 3.1.0 (not needed here)
- The package now bundles `@clerk/backend@2` as a direct dep (not peer)
- `CLERK_SECRET_KEY` env var — still the same

**No code changes needed.** The middleware call pattern `clerkMiddleware()(c, next)` is unchanged.

---

## Files to Touch

### Step 1 — Bump versions in package.json
File: `package.json`

```diff
- "@clerk/backend": "^1.32.0",
- "@clerk/clerk-react": "^5.31.2",
- "@hono/clerk-auth": "^2.0.0",
+ "@clerk/backend": "^3.4.3",
+ "@clerk/clerk-react": "^5.61.6",
+ "@hono/clerk-auth": "^3.1.1",
```

### Step 2 — Install
```bash
npm install
```

### Step 3 — Verify TypeScript still compiles
```bash
npx tsc --noEmit
```

### Step 4 — Build and smoke test
```bash
npm run build:dev
```

### Step 5 — Run local worker
```bash
npx wrangler dev
```
Then manually hit:
- `GET /api/protected/user` — should return 401 (no auth)
- A protected route with a valid token — should return 200

---

## Validation

```bash
# 1. No TypeScript errors
npx tsc --noEmit

# 2. Build succeeds
npm run build:dev

# 3. Wrangler dev starts without errors
npx wrangler dev --local
```

---

## PR Stack

| PR | Branch | Steps | Description |
|----|--------|-------|-------------|
| 1  | feat/clerk-upgrade | 1–5 | Bump all Clerk packages to latest major versions |

---

## Risk Assessment

**Overall: LOW RISK**

This project uses a very narrow slice of each package's API:
- Frontend: `ClerkProvider`, `useAuth`, `useUser`, `SignedIn`, `SignedOut`, `SignInButton`, `UserButton`, `PricingTable` — all stable across 5.x
- Backend: `clerkMiddleware()` and `getAuth(c)` from `@hono/clerk-auth` — API unchanged in 3.x
- No direct `@clerk/backend` usage in application code

The most likely failure mode is a TypeScript type incompatibility, caught by `tsc --noEmit`.
