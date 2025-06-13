# SmartFit Academy

A next-generation learning platform that leverages AI to deliver adaptive, personalized education from elementary through graduate levels. Built with modern web technologies and Cloudflare's edge infrastructure for a scalable, fast, and secure experience.

## 🌟 Features

- **Onboarding & Assessment**: Skill quiz to place learners at the right level
- **Adaptive Recommendations**: Claude-powered lesson suggestions
- **Lesson Viewer**: Interactive UI with code sandbox snippets
- **Chat Tutor**: Anthropic-powered Q&A assistant
- **Progress Dashboard**: Track completed modules, scores, badges
- **Admin CMS**: Authoring interface for lessons, quizzes, projects

## 🚀 Tech Stack

- **Frontend**: Vite + React + shadcn/ui (Tailwind CSS)
- **Backend/API**: Hono.js running on Cloudflare Workers (TypeScript)
  - Uses Cloudflare Workers runtime (V8 JavaScript engine)
  - Edge-first architecture for global low-latency
- **AI Integration**: Anthropic Claude (JavaScript/TypeScript SDK)
- **Database**: 
  - Cloudflare D1 (SQL) for structured data
  - Cloudflare KV for caching and session storage
- **Authentication**: Clerk.js with GitHub and Google providers

## 🛠️ Getting Started

### Prerequisites

- Node.js (Latest LTS version)
- npm or yarn
- Cloudflare account (for deployment)
- Anthropic API key (for AI features)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ebenryanmallory/progressive-ai-academy.git
cd progressive-ai-academy
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Set up environment variables:
```bash
cp .env.example .env
```
Fill in the required environment variables in `.env`

4. Start the development server:
```bash
npm run dev
# or
yarn dev
```

### Development

- Run tests: `npm run test`
- Build for production: `npm run build`
- Deploy to Cloudflare: `npm run deploy`

## 📊 Architecture Overview

### Cloudflare Infrastructure
- **Static Assets**: Built frontend (`dist/`) served directly by Hono Worker using Cloudflare's edge network
- **API**: All backend endpoints (lesson data, user progress, chat) handled by Hono routes on Worker
- **Data**: Cloudflare D1 for structured data (users, lessons, progress), KV for caching and session storage
- **AI**: Anthropic Claude SDK integration for adaptive recommendations and tutoring
- **CI/CD**: Automated build, local dev, and deployment via Wrangler

### Core Components
- **Lesson System**: Dynamic content delivery with progress tracking and recommendations
- **User System**: Authentication, profiles, progress tracking
- **AI Integration**: Claude-powered chat tutor and adaptive recommendations
- **Analytics**: Learning progress and engagement tracking

## Learning Path & Content Model

### Progression Levels
- **Elementary**: Foundational concepts and basic programming principles
- **High School**: Intermediate programming and computer science fundamentals
- **Undergraduate**: Advanced topics and practical applications
- **Graduate**: Specialized subjects and research-oriented content

### Module Types
- **Concept Lessons**: Text and video content explaining core concepts
- **Interactive Coding Exercises**: Hands-on programming challenges with instant feedback
- **Quizzes & Assessments**: Knowledge checks and skill evaluations
- **Final Projects**: Comprehensive applications demonstrating learned concepts

### Data Models
- **User & Profile**: User accounts, preferences, and learning history
- **Progress Tracking**: Completion status, scores, and achievements
- **Content Management**: Lessons, modules, quizzes, and projects
- **AI Recommendations**: Personalized learning paths and suggestions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anthropic](https://www.anthropic.com/) for Claude AI
- [Cloudflare](https://www.cloudflare.com/) for infrastructure
- [Clerk](https://clerk.com/) for authentication
- [shadcn/ui](https://ui.shadcn.com/) for UI components

## 📞 Support

For support, please open an issue in the GitHub repository or contact us at [support@progressiveai.academy](mailto:support@progressiveai.academy).

---

Built with ❤️ by SmartFit Academy
