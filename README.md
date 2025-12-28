# Physical AI & Humanoid Robotics - RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot for the Physical AI & Humanoid Robotics book, allowing users to ask questions about the book content and receive contextually relevant answers.

## 🚀 Features

- **AI-Powered Q&A**: Ask questions about the Physical AI & Humanoid Robotics book content
- **Semantic Search**: Uses vector embeddings to find relevant book sections
- **Google Gemini Integration**: Leverages Gemini models for embeddings and generation
- **Qdrant Vector Store**: Efficiently stores and retrieves book content
- **Docusaurus Integration**: Embedded directly into the documentation site
- **Text Selection Support**: Ask questions about specific text selected on the page

## 🛠️ Tech Stack

- **Backend**: FastAPI with Google Gemini integration
- **Vector Store**: Qdrant Cloud
- **Frontend**: Docusaurus with React
- **Embeddings**: Google's text-embedding-004 model
- **Generation**: Google Gemini 2.5 Pro Flash model

## 📋 Prerequisites

- Python 3.9+
- Node.js 18+
- Google AI Studio API key (for Gemini models)
- Qdrant Cloud account (optional, with fallback to local storage)

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Physical-AI-Humanoid-Robotics
```

### 2. Backend Setup

#### Navigate to the backend directory:
```bash
cd backend
```

#### Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install dependencies:
```bash
pip install -r requirements.txt
```

#### Create environment file:
```bash
cp .env.example .env
```

#### Configure environment variables in `.env`:
- `GEMINI_API_KEY`: Your Google AI Studio API key
- `QDRANT_URL`: Your Qdrant Cloud URL (optional)
- `QDRANT_API_KEY`: Your Qdrant Cloud API key (optional)
- `QDRANT_COLLECTION_NAME`: Collection name (default: book_content)
- `GEMINI_MODEL`: Model to use for generation (default: gemini-2.5-pro-flash)
- `GEMINI_EMBEDDING_MODEL`: Model to use for embeddings (default: models/text-embedding-004)

### 3. Frontend Setup

#### Navigate to the docusaurus directory:
```bash
cd docusaurus  # From project root
```

#### Install dependencies:
```bash
npm install
```

#### Build the site:
```bash
npm run build
```

### 4. Content Ingestion

#### Index the book content into the vector store:

**Method 1: Using the ingestion script (recommended)**
```bash
cd backend
python ingest.py
```

**Method 2: Using the API endpoint**
```bash
curl -X POST http://localhost:8000/api/ingest
```

### 5. Running the Application

#### Start the backend server:
```bash
cd backend
python -m uvicorn rag_backend:app --host 0.0.0.0 --port 8000
```

#### Start the frontend server:
```bash
cd docusaurus
npm start
```

The Docusaurus site will be available at `http://localhost:3000`, and the backend API at `http://localhost:8000`.

## 🏗️ Architecture

### Backend Structure
```
backend/
├── rag_backend.py          # Main FastAPI application
├── src/
│   ├── core/
│   │   ├── config.py       # Configuration settings
│   │   └── vector_store.py # Qdrant integration with Gemini embeddings
│   ├── services/
│   │   ├── rag_service.py  # RAG logic
│   │   └── content_loader.py # Content ingestion
├── ingest.py               # Content ingestion script
└── requirements.txt        # Dependencies
```

### Key Endpoints
- `POST /api/chat` - Chat with the RAG system
- `POST /api/ingest` - Ingest book content into vector store
- `POST /api/select-text` - Store selected text highlights
- `POST /api/translate` - Translate content
- `GET /health` - Health check

### Frontend Integration
The chatbot is integrated into Docusaurus as a floating component that can be accessed from any page.

## 🚀 Usage

1. **Ask General Questions**: Type questions about Physical AI and Humanoid Robotics in the chat interface
2. **Ask About Selected Text**: Select text on any page and use the "Ask about this" functionality
3. **Get Contextual Answers**: The system retrieves relevant book content and generates answers based on the context

## 📊 Environment Variables

### Backend (.env)
| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google AI Studio API key | required |
| `GEMINI_MODEL` | Generation model | `gemini-2.5-pro-flash` |
| `GEMINI_EMBEDDING_MODEL` | Embedding model | `models/text-embedding-004` |
| `QDRANT_URL` | Qdrant Cloud URL | optional |
| `QDRANT_API_KEY` | Qdrant Cloud API key | optional |
| `QDRANT_COLLECTION_NAME` | Vector collection name | `book_content` |

### Frontend (build-time)
| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_CHATBOT_API_URL` | Backend API URL | `http://127.0.0.1:8000` |

## 🔒 Security

- API keys are loaded from environment variables
- Never commit `.env` files to version control
- Use proper CORS configuration for production
- Implement authentication for sensitive operations

## 🚀 Deployment

### Backend Deployment
The backend can be deployed to platforms like:
- Render
- Railway
- Vercel
- Fly.io

### Frontend Deployment
The Docusaurus frontend can be deployed to:
- GitHub Pages
- Vercel
- Netlify
- Any static hosting service

### Environment Variables for Deployment
Make sure to set all required environment variables in your deployment platform's configuration.

## 🤖 Model Configuration

The system uses Google's Gemini models:

- **Generation**: `gemini-2.5-pro-flash` - Fast, capable model for generating responses
- **Embeddings**: `models/text-embedding-004` - High-quality embedding model for semantic search

You can change these in the `.env` file if needed.

## 📚 Content Structure

The system automatically processes all markdown files in the `docusaurus/docs` directory, splitting content by headings and indexing it into the vector store for retrieval.

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure your Google AI Studio API key is valid and has sufficient quota
2. **Qdrant Connection**: If Qdrant is not available, the system falls back to in-memory storage
3. **Content Not Found**: Run the ingestion script to index book content
4. **CORS Errors**: Check backend CORS configuration matches your frontend URL

### Logging
Check backend logs for detailed error information and debugging.

## 📈 Performance Tips

- Use Qdrant Cloud for production deployments
- Monitor API usage and adjust model selection as needed
- Implement caching for frequently asked questions
- Consider using a CDN for static assets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support, please open an issue in the repository or contact the maintainers.