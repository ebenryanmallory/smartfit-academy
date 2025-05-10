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
- **Auth**: Clerk.js with GitHub provider

## 3. Cloudflare Usage
- **Static Assets**: Built frontend (`dist/`) is served directly by the Hono Worker using Cloudflare's edge network for low-latency global access.
- **API**: All backend endpoints (lesson data, user progress, chat, etc.) are handled by Hono routes on the Worker.
- **Data**: Cloudflare D1 is used for structured data (users, lessons, progress), and KV for fast key-value caching and session storage.
- **AI**: Anthropic Claude SDK is called from the Worker for adaptive recommendations and tutoring.
- **CI/CD**: Wrangler scripts automate build, local dev, and deployment to Cloudflare.

## 4. Sprint Items (Next 2 Weeks)
### Week 1: Public Experience & Auth Foundation
#### Public-Facing Features
- [x] Build landing page with:
  - [x] Hero section explaining the platform
  - [x] Sample lesson preview
  - [x] Learning path overview
  - [x] Call-to-action for sign up
- [ ] Create public lesson viewer component:
  - [x] Lesson content display
    - [x] Create basic lesson content layout with title, description, and markdown content rendering
  - [x] Code snippet viewer
    - [x] Add syntax highlighting for code blocks
  - [ ] Basic interactive elements
    - [x] Add copy button for code snippets
    - [x] Add interactive code playground
      - [x] Add Python code display
      - [x] Add JavaScript code display
      - [x] Add local JavaScript execution
      - [ ] Add Python backend service
  - [x] "Sign in to save progress" prompts
    - [x] Create reusable SaveProgressPrompt component
    - [x] Add prompts at key interaction points
    - [x] Add prompts for code playground
    - [x] Add prompts for lesson completion
- [x] Implement sample public lessons:
  - [x] "Introduction to AI" lesson
  - [x] "Getting Started with Programming" lesson
- [ ] Add progress persistence in localStorage for non-authenticated users
- [x] Create conversion points for sign-up:
  - [x] "Save your progress" prompts
  - [x] "Access more lessons" CTAs
  - [x] "Track your learning" features
  - [x] "Join community" CTAs
  - [x] "Get recommendations" CTAs

#### Authentication Setup
- [ ] Set up Clerk.js in the project:
  - [ ] Install and configure Clerk
  - [ ] Set up GitHub OAuth provider
  - [ ] Configure environment variables
- [ ] Implement auth components:
  - [ ] Sign-in button
  - [ ] Sign-up flow
  - [ ] User profile menu
- [ ] Add auth state management:
  - [ ] Auth context provider
  - [ ] Protected route wrapper
  - [ ] Auth status hooks
- [ ] Create auth middleware in Hono:
  - [ ] Session validation
  - [ ] Protected API routes
  - [ ] User context injection

### Week 2
- [ ] Implement basic user and lesson models in D1
- [ ] Build onboarding flow and sample lesson page
- [ ] Integrate Anthropic SDK for AI-powered chat tutor (simple Q&A)
- [ ] Add progress tracking endpoints to Worker
- [ ] Write initial unit/integration tests for API
- [ ] Prepare demo content and internal review

## 5. Milestones
| Milestone | Description                      | Deliverables                        | ETA (Weeks) |
|-----------|----------------------------------|-------------------------------------|-------------|
| M1        | Scaffold & deploy MVP infra      | Worker + static site + D1/KV setup  | 1           |
| M2        | Core learning & AI integration   | Lessons, chat tutor, progress model | 2           |
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

## 8. Next Steps
1. Create the public lesson viewer component
2. Implement sample public lessons
3. Set up Clerk.js authentication
4. Add progress persistence for non-authenticated users

## 3. Learning Path & Content Model
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

## 4. Core Features
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

## 7. Next Sprint
<!-- Add new sprint items here -->

