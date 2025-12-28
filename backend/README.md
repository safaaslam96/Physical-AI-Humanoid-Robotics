# Physical AI & Humanoid Robotics RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot for the Physical AI & Humanoid Robotics book. It uses Google's Gemini model and Qdrant vector database to provide accurate answers based on the book content.

## Features

- **Book Knowledge**: Answers questions based on the entire 20-chapter book content
- **Selected Text Queries**: Ask about highlighted text with the "Ask AI" button
- **Accurate Responses**: Uses RAG to ensure answers are grounded in book content
- **Source Attribution**: Provides links to relevant chapters/sections

## Architecture

- **Frontend**: Docusaurus-based book with floating AI chat popup
- **Backend**: FastAPI server with RAG capabilities
- **Vector Store**: Qdrant Cloud for storing document embeddings
- **LLM**: Google Gemini for question answering
- **Embeddings**: Google's embedding model for document retrieval

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- Node.js (for the Docusaurus frontend)
- Google API Key for Gemini
- Qdrant Cloud account

### 2. Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```bash
cp .env.example .env
# Edit .env with your actual keys
```

4. Run the ingestion script to index the book content:
```bash
python ingest.py
```

5. Start the backend server:
```bash
cd src/api
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup

1. Navigate to the Docusaurus directory:
```bash
cd docusaurus
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run start
```

## Usage

1. The AI chatbot appears as a floating robot icon on the bottom-right of the page
2. Click to open the chat interface
3. Ask questions about Physical AI, ROS2, Gazebo, NVIDIA Isaac, etc.
4. To ask about highlighted text:
   - Select text in the book content
   - Click the "Ask AI" button that appears above the selection
   - The chat will pre-populate with a question about the selected text

## API Endpoints

- `POST /chat` - Main chat endpoint
  - Request body: `{ "query": "your question", "selected_text": "optional selected text", "history": "chat history" }`
  - Response: `{ "response": "AI answer", "sources": ["relevant chapters"] }`

- `GET /health` - Health check endpoint

## Configuration

- **Qdrant**: Uses a collection named "physical_ai_book"
- **Embeddings**: Uses Google's embedding-001 model
- **LLM**: Uses Gemini 1.5 Flash model
- **Chunk Size**: 1000 characters with 200 character overlap

## Troubleshooting

- Make sure your API keys are correctly set in the `.env` file
- Verify that the ingestion script completed successfully
- Check that both frontend and backend servers are running
- Ensure CORS settings allow communication between frontend and backend