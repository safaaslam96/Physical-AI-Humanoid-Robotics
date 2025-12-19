# Physical AI & Humanoid Robotics Book - Backend

This is the backend API for the interactive Physical AI and Humanoid Robotics book, built with FastAPI.

## Features

- User authentication and profile management
- RAG (Retrieval-Augmented Generation) chatbot API
- Content personalization based on user profiles
- Urdu translation service
- Integration with Google Gemini 2.5 Pro Flash
- Qdrant vector store for content retrieval
- Neon PostgreSQL for user data

## Tech Stack

- FastAPI
- Python 3.11+
- Google Generative AI SDK
- Qdrant vector database
- SQLAlchemy for database operations
- Better Auth for authentication

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the application:
```bash
uvicorn main:app --reload
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `/api/v1/auth/` - Authentication endpoints
- `/api/v1/chat/` - RAG chatbot endpoints
- `/api/v1/personalization/` - Content personalization endpoints
- `/api/v1/translation/` - Translation endpoints