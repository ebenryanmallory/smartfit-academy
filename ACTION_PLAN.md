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
- **Auth**: (Pluggable; future: Cloudflare Access or third-party)

## 3. Cloudflare Usage
- **Static Assets**: Built frontend (`dist/`) is served directly by the Hono Worker using Cloudflare's edge network for low-latency global access.
- **API**: All backend endpoints (lesson data, user progress, chat, etc.) are handled by Hono routes on the Worker.
- **Data**: Cloudflare D1 is used for structured data (users, lessons, progress), and KV for fast key-value caching and session storage.
- **AI**: Anthropic Claude SDK is called from the Worker for adaptive recommendations and tutoring.
- **CI/CD**: Wrangler scripts automate build, local dev, and deployment to Cloudflare.

## 4. Sprint Items (Next 2 Weeks)
### Week 1
- [ ] Finalize project structure and naming
- [ ] Set up Vite + React + shadcn/ui frontend
- [ ] Scaffold Hono Worker backend and static asset serving
- [ ] Configure Wrangler and wrangler.toml for local/dev/deploy
- [ ] Deploy "Hello World" Worker serving static frontend
- [ ] Set up Cloudflare D1 and KV (schema, bindings)

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

