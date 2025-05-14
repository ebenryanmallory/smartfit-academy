# Progressive AI Academy: Action Plan

**Date**: 2025-04-26

## 1. Overview
Progressive AI Academy is a next-generation learning platform that leverages AI to deliver adaptive, personalized education from elementary through graduate levels. The platform uses modern web technologies and Cloudflare's edge infrastructure for a scalable, fast, and secure experience.

## 2. Architecture & Tech Stack
- **Frontend**: Vite + React + shadcn/ui (Tailwind CSS), built to `dist/`
- **Backend/API**: Hono server running on Cloudflare Workers (TypeScript)
- **AI SDK**: Anthropic Claude (JavaScript/TypeScript SDK)
- **Database**: Cloudflare D1 (SQL), KV (caching/session)
- **Hosting/Infra**: Cloudflare Workers (serving API and static assets from `dist/`), Wrangler for CI/CD
- **Auth**: Clerk.js with GitHub and Google providers

## 3. Cloudflare Usage
- **Static Assets**: Built frontend (`dist/`) is served directly by the Hono Worker using Cloudflare's edge network for low-latency global access.
- **API**: All backend endpoints (lesson data, user progress, chat, etc.) are handled by Hono routes on the Worker.
- **Data**: Cloudflare D1 is used for structured data (users, lessons, progress), and KV for fast key-value caching and session storage.
- **AI**: Anthropic Claude SDK is called from the Worker for adaptive recommendations and tutoring.
- **CI/CD**: Wrangler scripts automate build, local dev, and deployment to Cloudflare.

### Week 2
- [ ] Integrate Anthropic SDK for AI-powered chat tutor (simple Q&A)
- [ ] Add progress tracking endpoints to Worker
- [ ] Write initial unit/integration tests for API
- [ ] Prepare demo content and internal review
- [ ] Implement hybrid lesson system: hard-coded lesson content in `/src/data`, but track user progress, recommendations, and access control in D1 using lesson IDs. All APIs and personalization reference lesson IDs, so lesson content can later move to DB seamlessly.

#### Hybrid Lesson Implementation Plan

**Lesson Content**
- Hard-coded in `/src/data` with unique `lesson_id` for each lesson.

**Database Tables**
- User Progress: Tracks per-user lesson status by `lesson_id`.
- User Recommendations: Stores personalized lesson suggestions per user.
- User Access: Controls access level for each lesson per user.

**Backend Logic**
- APIs join user data with hard-coded lesson metadata by `lesson_id`.
- All personalization, recommendations, and access control are DB-driven.

**Frontend Logic**
- Renders hard-coded lesson content by `lesson_id`.
- Displays user progress, access, and recommendations using backend data.

**Future-Proofing**
- All APIs and tables reference `lesson_id`, so moving content to DB later is seamless.
| M3        | Content authoring/admin tools    | Admin UI, content workflow          | 2           |
| M4        | Polish, test, and launch         | QA, docs, public launch             | 1           |

## 6. Development Workflow
- **Monorepo**: Single repo for frontend and Worker code
- **Cloudflare Infra**: All-in-one deployment via Wrangler
- **Testing**: Local dev via `wrangler dev`, deploy with `wrangler deploy`
- **Docs**: Update this plan as architecture evolves

## 7. Recent Updates
- **React & Dependencies**: Updated to React 19.1.0, react-dom, react-router-dom, @types/react, @types/react-dom, tailwindcss, typescript, vite, wrangler, @tailwindcss/postcss, hono, eslint, and typescript-eslint to their latest versions for compatibility with shadcn/ui and Tailwind v4.
- **Landing Page**: Implemented the landing page with hero section, sample lesson preview, learning path overview, and sign-up CTA using shadcn/ui components.
- **Authentication**: Implemented Clerk.js authentication with GitHub and Google providers, custom styling, and proper auth state management.

## 8. Next Steps
1. Create auth middleware in Hono for protected API routes
2. Implement basic user and lesson models in D1
3. Build onboarding flow and sample lesson page
4. Add progress persistence for non-authenticated users

## 9. Learning Path & Content Model
- **Progression Levels**:
  - Elementary
  - High School
  - Undergraduate
  - Graduate
- **Module Types**:
  - Concept lessons (text + video)
  - Interactive coding exercises
  - Quizzes & assessments
  - Final projects
- **Data Models**:
  - User, Profile, Progress
  - Lesson, Module, Quiz, Project

## 10. Core Features
1. **Onboarding & Assessment**: Skill quiz to place learners at the right level
2. **Adaptive Recommendations**: Claude-powered lesson suggestions
3. **Lesson Viewer**: Interactive UI with code sandbox snippets
4. **Chat Tutor**: Anthropic-powered Q&A assistant
5. **Progress Dashboard**: Track completed modules, scores, badges
6. **Admin CMS**: Authoring interface for lessons, quizzes, projects

## 5. Milestones & Timeline
| Milestone | Description                                    | Deliverables                               | ETA (Weeks) |
|-----------|------------------------------------------------|--------------------------------------------|-------------|
| M1        | Scaffold frontend + UI theme                   | Vite repo, shadcn setup, layout components | 1           |
| M2        | Auth & DB integration                          | NextAuth, Prisma schema, DB migrations     | 2           |
| M3        | Lesson model & static content pages            | Lesson page template, sample content       | 2           |
| M4        | Chat Tutor prototype                           | Anthropic SDK integration, chat UI         | 1           |
| M5        | Adaptive recommendation engine                 | Prompt templates, recommendation API       | 2           |
| M6        | Quizzes & progress tracking                    | Quiz components, progress endpoints        | 1           |
| M7        | Admin CMS & final polish                       | Admin UI, permissions, deployment ready    | 2           |

## 6. Development Workflow
- **Repository**: Single repo (monorepo optional)
- **Infrastructure**: Cloudflare (Pages for frontend hosting, Workers for serverless/API, D1 for database, Access for auth, KV for caching)
- **CI/CD**: GitHub Actions → Cloudflare Pages/Workers deployments
- **Testing**: Jest + React Testing Library
- **Linting/Formatting**: ESLint, Prettier

