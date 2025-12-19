# Quickstart Guide: AI-Powered Book - Physical AI and Humanoid Robotics

## Overview
This guide provides instructions for setting up and running the AI-powered book project locally.

## Prerequisites

### System Requirements
- Python 3.11+
- Node.js 18+ with npm
- Git
- Docker (optional, for local PostgreSQL)

### External Services
- OpenAI API key (for RAG chatbot and translation)
- Qdrant Cloud account (free tier)
- Neon Serverless PostgreSQL account
- Better Auth account

## Environment Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

## Configuration

### 1. Environment Variables

Create `.env` files in both backend and frontend directories:

**Backend (.env):**
```env
OPENAI_API_KEY=your_openai_api_key
NEON_DATABASE_URL=your_neon_database_url
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
JWT_SECRET=your_jwt_secret
BETTER_AUTH_SECRET=your_better_auth_secret
BETTER_AUTH_URL=http://localhost:3000
```

**Frontend (.env):**
```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_BETTER_AUTH_URL=http://localhost:3000
```

### 2. Database Setup
```bash
# From backend directory
python -m src.core.database init
python -m src.core.database migrate
```

### 3. Vector Store Setup
```bash
# Initialize Qdrant collections
python -m src.core.vector_store init
```

## Running the Application

### 1. Start Backend Server
```bash
# From backend directory
python -m uvicorn src.main:app --reload --port 8000
```

### 2. Start Frontend Server
```bash
# From frontend directory
npm start
```

## Key Endpoints

### Backend API
- `GET /api/health` - Health check
- `POST /api/chat/query` - RAG chatbot endpoint
- `POST /api/personalize/chapter` - Chapter personalization
- `POST /api/translate/urdu` - Urdu translation
- `GET /api/chapters` - List all book chapters
- `GET /api/chapters/{id}` - Get specific chapter

### Frontend
- `http://localhost:3000` - Main book interface
- `http://localhost:3000/chapter/{slug}` - Individual chapter view
- `http://localhost:3000/chat` - Standalone chat interface

## Development Workflow

### 1. Adding New Chapters
```bash
# Add chapter content to static/chapters/ directory
# Format: chapter-slug.md with frontmatter
```

### 2. Testing RAG Functionality
```bash
# Re-index content after adding/changing chapters
python -m src.services.rag_service reindex
```

### 3. Running Tests
```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

## Deployment

### 1. Build Frontend
```bash
cd frontend
npm run build
```

### 2. Deploy to GitHub Pages
```bash
# Frontend deployment
npm run deploy

# Backend deployment (to your preferred hosting)
# Configuration varies by provider
```

## Troubleshooting

### Common Issues

1. **Qdrant Connection Errors**
   - Verify QDRANT_URL and QDRANT_API_KEY in environment
   - Check network connectivity to Qdrant cluster

2. **Database Connection Errors**
   - Verify NEON_DATABASE_URL in environment
   - Ensure database migration has run

3. **OpenAI API Errors**
   - Verify OPENAI_API_KEY in environment
   - Check API quota and billing

### Useful Commands

```bash
# Check backend health
curl http://localhost:8000/api/health

# View API documentation
http://localhost:8000/docs

# Run specific test
python -m pytest tests/test_chat.py
```