# Claude Agent Context: Physical AI and Humanoid Robotics Book Project

## Project Overview
AI-powered interactive book about Physical AI and Humanoid Robotics with integrated RAG chatbot, personalization features, and Urdu translation capabilities.

## Technology Stack
- **Frontend**: Docusaurus v3+, React, JavaScript/TypeScript
- **Backend**: FastAPI, Python 3.11
- **Database**: Neon Serverless PostgreSQL
- **Vector Store**: Qdrant Cloud
- **Authentication**: Better Auth
- **AI Services**: OpenAI Agents SDK frameworks with Google Gemini 2.5 Pro Flash for RAG and translation
- **Hosting**: GitHub Pages (frontend), cloud hosting (backend)

## Key Components
- **RAG Chatbot**: OpenAI Assistants API with Qdrant vector search
- **Personalization Engine**: Rule-based system with ML enhancement
- **Translation Service**: OpenAI GPT-4 with prompt engineering for Urdu
- **Content Management**: Markdown-based with frontmatter metadata

## API Endpoints
- `/api/chat/query` - RAG chatbot functionality
- `/api/personalize/chapter` - Chapter personalization
- `/api/translate/urdu` - Urdu translation service
- `/api/chapters` - Book chapter management

## Data Models
- User, UserProfile, BookChapter, ChatInteraction, PersonalizedChapter, TranslationCache, VectorEmbedding

## Architecture Patterns
- Monolithic backend with modular services
- Static site generation with Docusaurus
- Multi-layer caching strategy
- Async processing for AI operations

## Security Considerations
- JWT token authentication
- Input sanitization and output encoding
- Rate limiting for AI services
- Secure API key management

## Performance Goals
- 95% of RAG queries answered within 3 seconds
- 99.9% uptime
- Support 100 concurrent users
- <200ms p95 for personalization
- <10 seconds for Urdu translation
- <5 seconds page load time

## Special Instructions
- All content generation must follow spec-driven, deterministic rules
- RAG responses must be grounded in source material with proper citations
- Personalization must maintain technical accuracy while adapting to user profiles
- Translation must preserve technical terminology and code blocks
- Follow WCAG 2.1 AA accessibility standards
- Implement graceful degradation for service outages