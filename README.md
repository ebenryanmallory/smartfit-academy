# Progressive AI Academy

An open-source, next-generation learning platform that leverages AI to deliver adaptive, personalized education from elementary through graduate levels. Built with modern web technologies and Cloudflare's edge infrastructure for a scalable, fast, and secure experience.

## 🌟 Features

- **Adaptive Learning**: AI-powered personalized education paths
- **Interactive Lessons**: Code playgrounds, quizzes, and assessments
- **AI Chat Tutor**: Real-time assistance powered by Anthropic Claude
- **Progress Tracking**: Monitor your learning journey with detailed analytics
- **Multi-level Content**: Content ranging from elementary to graduate level
- **Modern Tech Stack**: Built with React, TypeScript, and Cloudflare's edge infrastructure

## 🚀 Tech Stack

- **Frontend**: Vite + React + shadcn/ui (Tailwind CSS)
- **Backend/API**: Hono server running on Cloudflare Workers (TypeScript)
- **AI Integration**: Anthropic Claude (JavaScript/TypeScript SDK)
- **Database**: Cloudflare D1 (SQL), KV (caching/session)
- **Hosting**: Cloudflare Workers (serving API and static assets)
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

## 📚 Project Structure

```
progressive-ai-academy/
├── src/                    # Source files
│   ├── components/        # React components
│   ├── pages/            # Page components
│   ├── api/              # API routes
│   └── utils/            # Utility functions
├── public/               # Static assets
├── dist/                # Build output
└── wrangler.toml        # Cloudflare configuration
```

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

Built with ❤️ by the Progressive AI Academy team
