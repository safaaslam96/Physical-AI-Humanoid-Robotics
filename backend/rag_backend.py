from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import openai
import json
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from google.generativeai import embedding
from dotenv import load_dotenv
import asyncio
from src.api.translation_routes import router as translation_router

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Physical AI & Humanoid Robotics RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include translation routes
app.include_router(translation_router, prefix="/api/v1", tags=["translation"])

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
NEON_DB_URL = os.getenv("NEON_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize OpenAI client (kept for fallback/alternative use)
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI client: {e}")
        openai_client = None
else:
    print("Warning: OPENAI_API_KEY not found in environment variables")
    openai_client = None

# Initialize Google Generative AI client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables")

gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest")
embedding_model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")  # Gemini embedding model

# Qdrant client initialization with fallback
qdrant_client = None
QDRANT_AVAILABLE = False

# Define helper functions for Gemini embeddings
def get_gemini_embedding(text: str):
    """Get embedding from Gemini model"""
    try:
        # Check if text is empty or too short
        if not text or len(text.strip()) == 0:
            # Return a zero vector or handle empty text appropriately
            return [0.0] * 768  # Standard embedding size

        result = genai.embed_content(
            model=embedding_model_name,
            content=text,
            task_type="retrieval_document"  # or "retrieval_query" for queries
        )
        return result['embedding']
    except Exception as e:
        print(f"Gemini embedding failed: {e}")
        # Fallback to OpenAI if Gemini fails
        if openai_client is not None:
            try:
                response = openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                return response.data[0].embedding
            except Exception as oe:
                print(f"OpenAI embedding also failed: {oe}")
                # Return a default embedding as last resort
                return [0.0] * 1536  # Standard OpenAI embedding size
        else:
            print("OpenAI client not available, using default embedding")
            # Return a default embedding as last resort
            return [0.0] * 768  # Standard embedding size

# Define wrapper functions first so they're available during initialization
def create_collection_fallback(collection_name: str, vectors_config):
    """Fallback function for creating collections"""
    print(f"Created in-memory collection: {collection_name}")
    return True

def get_collection_fallback(collection_name: str):
    """Fallback function for getting collection info"""
    # Return success if collection exists conceptually
    if collection_name in ["book_content", "highlights"]:
        return True
    raise Exception(f"Collection {collection_name} not found")

def create_collection(collection_name: str, vectors_config):
    global QDRANT_AVAILABLE
    if QDRANT_AVAILABLE and qdrant_client is not None:
        try:
            return qdrant_client.create_collection(collection_name=collection_name, vectors_config=vectors_config)
        except Exception as e:
            print(f"Qdrant create_collection failed: {e}. Switching to in-memory fallback.")
            QDRANT_AVAILABLE = False
            return create_collection_fallback(collection_name, vectors_config)
    else:
        return create_collection_fallback(collection_name, vectors_config)

def get_collection(collection_name: str):
    global QDRANT_AVAILABLE
    if QDRANT_AVAILABLE and qdrant_client is not None:
        try:
            return qdrant_client.get_collection(collection_name)
        except Exception as e:
            print(f"Qdrant get_collection failed: {e}. Switching to in-memory fallback.")
            QDRANT_AVAILABLE = False
            return get_collection_fallback(collection_name)
    else:
        return get_collection_fallback(collection_name)

def upsert(collection_name: str, points):
    global QDRANT_AVAILABLE
    if QDRANT_AVAILABLE and qdrant_client is not None:
        try:
            return qdrant_client.upsert(collection_name=collection_name, points=points)
        except Exception as e:
            print(f"Qdrant upsert failed: {e}. Switching to in-memory fallback.")
            QDRANT_AVAILABLE = False
            return upsert_fallback(collection_name, points)
    else:
        return upsert_fallback(collection_name, points)

def search(collection_name: str, query_vector, query_filter=None, limit=5):
    global QDRANT_AVAILABLE
    if QDRANT_AVAILABLE and qdrant_client is not None:
        try:
            return qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit
            )
        except Exception as e:
            print(f"Qdrant search failed: {e}. Switching to in-memory fallback.")
            QDRANT_AVAILABLE = False
            # For the fallback, we'll handle search differently since we don't have the query vector
            # The search_in_memory functions will be called directly from the endpoints
            return search_fallback(collection_name, query_vector, query_filter, limit)
    else:
        # For the fallback, we'll handle search differently since we don't have the query vector
        # The search_in_memory functions will be called directly from the endpoints
        return search_fallback(collection_name, query_vector, query_filter, limit)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    QDRANT_AVAILABLE = True
    print("Qdrant connection successful")
except Exception as e:
    print(f"Qdrant connection failed: {e}. Using in-memory fallback.")
    QDRANT_AVAILABLE = False
    # Import models from qdrant_client for type hints if needed, but define minimal versions for fallback
    class MockDistance:
        COSINE = "cosine"
    class MockFieldCondition:
        def __init__(self, key, match):
            self.key = key
            self.match = match
    class MockMatchValue:
        def __init__(self, value):
            self.value = value
    class MockFilter:
        def __init__(self, must=None):
            self.must = must or []
    class MockPointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload
    class MockVectorParams:
        def __init__(self, size, distance):
            self.size = size
            self.distance = distance

    # Define mock models for fallback
    models = type('Models', (), {
        'Distance': MockDistance(),
        'FieldCondition': MockFieldCondition,
        'MatchValue': MockMatchValue,
        'Filter': MockFilter,
        'PointStruct': MockPointStruct,
        'VectorParams': MockVectorParams
    })()

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    chapter_context: Optional[str] = None
    use_highlights_only: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]
    timestamp: str

class SelectTextRequest(BaseModel):
    text: str
    user_id: str
    chapter_id: str
    highlight_color: Optional[str] = "yellow"
    context: Optional[str] = ""

class FeedbackRequest(BaseModel):
    query: str
    response: str
    user_rating: int  # 1-5 scale
    user_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize database connection and Qdrant collection or in-memory storage"""
    if QDRANT_AVAILABLE:
        # Initialize Qdrant collections when available
        try:
            get_collection("book_content")
        except:
            create_collection(
                collection_name="book_content",
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )

        # Initialize highlights collection
        try:
            get_collection("highlights")
        except:
            create_collection(
                collection_name="highlights",
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )
    else:
        # For in-memory fallback, just initialize the collections conceptually
        print("Using in-memory storage - Qdrant not available")
        # Initialize in-memory collections by calling create_collection_fallback
        create_collection("book_content", None)
        create_collection("highlights", None)

    # Load sample data for demonstration if available (do this regardless of QDRANT availability)
    try:
        print("Attempting to load sample data from ../rag_data/summaries.json")
        with open("../rag_data/summaries.json", "r") as f:
            sample_data = json.load(f)

        for item in sample_data.get("summaries", []):
            # Create embedding for the content using Gemini
            content_embedding = get_gemini_embedding(item["content"])

            # Store in storage (Qdrant or in-memory)
            highlight_id = str(uuid.uuid4())

            upsert(
                collection_name="book_content",
                points=[
                    models.PointStruct(
                        id=highlight_id,
                        vector=content_embedding,
                        payload={
                            "content": item["content"],
                            "chapter_id": item["id"],
                            "title": item["title"],
                            "key_terms": ", ".join(item["key_terms"]),
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                ]
            )
        print(f"Loaded {len(sample_data.get('summaries', []))} sample documents for demonstration")
    except Exception as e:
        print(f"Could not load sample data: {e}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint that answers questions based on book content"""
    try:
        # Generate embedding for the query using Gemini
        query_embedding = get_gemini_embedding(request.message)

        # Search based on availability of Qdrant
        if QDRANT_AVAILABLE:
            # Use Qdrant for search
            if request.use_highlights_only and request.user_id:
                # Search in user's highlights
                search_results = search(
                    collection_name="highlights",
                    query_vector=query_embedding,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=request.user_id)
                            )
                        ]
                    ),
                    limit=5
                )
            else:
                # Search in full book content
                search_results = search(
                    collection_name="book_content",
                    query_vector=query_embedding,
                    limit=5
                )
        else:
            # Use in-memory search fallback
            if request.use_highlights_only and request.user_id:
                # Search in user's highlights using TF-IDF
                search_results = search_in_memory_highlights(request.message, request.user_id, limit=5)
            else:
                # Search in full book content using TF-IDF
                search_results = search_in_memory_content(request.message, limit=5)

        # Prepare context from search results
        context_texts = [result.payload.get('content', '') for result in search_results]
        context = "\n\n".join(context_texts)

        # Generate response using Gemini
        gemini_model = genai.GenerativeModel(gemini_model_name)

        # Prepare the prompt for Gemini
        prompt = f"""You are an expert assistant for the Physical AI & Humanoid Robotics book. Answer questions based on the provided context from the book. Be helpful and accurate.

Context: {context}

Question: {request.message}"""

        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.7
                )
            )
            answer = response.text
        except Exception as e:
            print(f"Gemini generation failed: {e}")
            # When Gemini fails, provide a helpful response based on available context
            if context.strip():
                answer = f"Based on the book content: {context[:500]}... For your question '{request.message}', please refer to the relevant sections in the book. The AI service is temporarily unavailable."
            else:
                answer = f"I understand you're asking about: {request.message}. The AI service is currently unavailable, but please refer to the Physical AI & Humanoid Robotics book for comprehensive information on this topic."

        # Prepare sources
        sources = []
        for result in search_results:
            sources.append({
                "chapter": result.payload.get('chapter_id', ''),
                "content_preview": result.payload.get('content', '')[:200],
                "score": getattr(result, 'score', 0.5)  # Use getattr to handle fallback case
            })

        return ChatResponse(
            response=answer,
            sources=sources,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# In-memory storage for highlights and feedback (in production, use a proper database)
user_highlights_db: Dict[str, List[Dict]] = {}
chat_feedback_db: List[Dict] = []

# Global in-memory storage for fallback (defined regardless of QDRANT availability)
# Storage for book content and highlights
book_content_storage = []
highlights_storage = []

# TF-IDF vectorizers for similarity search fallback
book_content_vectorizer = TfidfVectorizer()
book_content_vectors = None
highlights_vectorizer = TfidfVectorizer()
highlights_vectors = None
user_highlights_vectors = {}  # For each user's highlights

def upsert_fallback(collection_name: str, points):
    """Fallback function for upserting vectors"""
    global book_content_vectors, highlights_vectors
    for point in points:
        if collection_name == "book_content":
            book_content_storage.append({
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload
            })
            # Rebuild vectorizer and vectors
            if book_content_storage:
                contents = [item["payload"].get("content", "") for item in book_content_storage]
                book_content_vectors = book_content_vectorizer.fit_transform(contents)
        elif collection_name == "highlights":
            highlights_storage.append({
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload
            })
            # Group by user and rebuild vectors for user highlights
            user_id = point.payload.get("user_id", "")
            if user_id not in user_highlights_vectors:
                user_highlights_vectors[user_id] = []

            user_highlights_vectors[user_id].append({
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload
            })

            # Rebuild vectorizer for this user's highlights
            user_highlights = [item["payload"].get("highlighted_text", "") for item in user_highlights_vectors[user_id]]
            if user_highlights:
                user_highlights_vectors[user_id + "_vectors"] = highlights_vectorizer.fit_transform(user_highlights)

def search_fallback(collection_name: str, query_vector, query_filter=None, limit=5):
    """Fallback function for searching vectors"""
    from types import SimpleNamespace

    results = []

    if collection_name == "book_content":
        if book_content_storage and book_content_vectors is not None:
            # For fallback, we'll use TF-IDF similarity instead of vector similarity
            # We'll create a simple search based on content matching
            query_text = f"Query vector converted to text"  # Placeholder
            # Instead of using the query vector directly, let's just do a simple similarity search
            # Since we don't have the original text to vectorize in the same way, we'll use a different approach
            # We'll create a simple keyword-based search for the fallback
            pass  # We'll implement this in the chat endpoint directly
    elif collection_name == "highlights":
        if query_filter and query_filter.must:
            # Filter by user_id
            user_id = query_filter.must[0].match.value
            user_highlights = [h for h in highlights_storage if h["payload"].get("user_id") == user_id]

            # For now, return all user highlights (this is a simplified fallback)
            for h in user_highlights[:limit]:
                result = SimpleNamespace()
                result.payload = h["payload"]
                result.score = 0.5  # Placeholder score
                results.append(result)
        else:
            results = [SimpleNamespace(payload=h["payload"], score=0.5) for h in highlights_storage[:limit]]

    return results

def search_in_memory_content(query_text: str, limit=5):
    """Search in book content using TF-IDF similarity"""
    from types import SimpleNamespace

    if not book_content_storage:
        return []

    # Extract content texts
    contents = [item["payload"].get("content", "") for item in book_content_storage]

    if not contents:
        return []

    # Create TF-IDF vectors for search
    vectorizer = TfidfVectorizer()
    content_vectors = vectorizer.fit_transform(contents)

    # Transform the query
    query_vector = vectorizer.transform([query_text])

    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, content_vectors).flatten()

    # Get top results
    top_indices = similarities.argsort()[-limit:][::-1]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:  # Only return results with some similarity
            result = SimpleNamespace()
            result.payload = book_content_storage[idx]["payload"]
            result.score = float(similarities[idx])
            results.append(result)

    return results

def search_in_memory_highlights(query_text: str, user_id: str, limit=5):
    """Search in user's highlights using TF-IDF similarity"""
    from types import SimpleNamespace

    # Get user's highlights
    user_highlights = [h for h in highlights_storage if h["payload"].get("user_id") == user_id]

    if not user_highlights:
        return []

    # Extract highlighted texts
    contents = [item["payload"].get("highlighted_text", "") for item in user_highlights]

    if not contents:
        return []

    # Create TF-IDF vectors for search
    vectorizer = TfidfVectorizer()
    content_vectors = vectorizer.fit_transform(contents)

    # Transform the query
    query_vector = vectorizer.transform([query_text])

    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, content_vectors).flatten()

    # Get top results
    top_indices = similarities.argsort()[-limit:][::-1]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:  # Only return results with some similarity
            result = SimpleNamespace()
            result.payload = user_highlights[idx]["payload"]
            result.score = float(similarities[idx])
            results.append(result)

    return results

# When Qdrant is available, we still have the fallback functions defined globally
if not QDRANT_AVAILABLE:
    print("Using in-memory storage - Qdrant not available")
    # Initialize in-memory collections by calling create_collection_fallback
    create_collection("book_content", None)
    create_collection("highlights", None)

    # Load sample data for demonstration if available
    try:
        print("Attempting to load sample data from ../rag_data/summaries.json")
        with open("../rag_data/summaries.json", "r") as f:
            sample_data = json.load(f)
        print(f"Successfully loaded sample data with {len(sample_data.get('summaries', []))} summaries")

        for item in sample_data.get("summaries", []):
            # Create embedding for the content using Gemini
            content_embedding = get_gemini_embedding(item["content"])

            # Store in in-memory book content storage
            highlight_id = str(uuid.uuid4())

            upsert(
                collection_name="book_content",
                points=[
                    models.PointStruct(
                        id=highlight_id,
                        vector=content_embedding,
                        payload={
                            "content": item["content"],
                            "chapter_id": item["id"],
                            "title": item["title"],
                            "key_terms": ", ".join(item["key_terms"]),
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                ]
            )
        print(f"Loaded {len(sample_data.get('summaries', []))} sample documents for demonstration")
    except Exception as e:
        print(f"Could not load sample data: {e}")
else:
    # When Qdrant is available, define the behavior for startup
    pass

@app.post("/api/select-text")
async def select_text_endpoint(request: SelectTextRequest):
    """Endpoint to store highlighted text"""
    try:
        # Generate embedding for the selected text using Gemini
        text_embedding = get_gemini_embedding(request.text)

        # Store in highlights collection (Qdrant or in-memory)
        highlight_id = str(uuid.uuid4())

        upsert(
            collection_name="highlights",
            points=[
                models.PointStruct(
                    id=highlight_id,
                    vector=text_embedding,
                    payload={
                        "user_id": request.user_id,
                        "chapter_id": request.chapter_id,
                        "highlighted_text": request.text,
                        "context": request.context,
                        "highlight_color": request.highlight_color,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            ]
        )

        # Store in in-memory database
        if request.user_id not in user_highlights_db:
            user_highlights_db[request.user_id] = []

        user_highlights_db[request.user_id].append({
            "id": highlight_id,
            "user_id": request.user_id,
            "chapter_id": request.chapter_id,
            "highlighted_text": request.text,
            "context": request.context,
            "highlight_color": request.highlight_color,
            "timestamp": datetime.now().isoformat()
        })

        return {"success": True, "highlight_id": highlight_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Highlight error: {str(e)}")

@app.post("/api/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Endpoint to collect feedback on responses"""
    try:
        # Store feedback in in-memory database
        feedback_entry = {
            "query": request.query,
            "response": request.response,
            "user_rating": request.user_rating,
            "user_id": request.user_id,
            "timestamp": datetime.now().isoformat()
        }

        chat_feedback_db.append(feedback_entry)

        return {"success": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")

@app.get("/api/user-highlights/{user_id}/{chapter_id}")
async def get_user_highlights(user_id: str, chapter_id: str):
    """Get all highlights for a user in a specific chapter"""
    try:
        # Get highlights from in-memory database
        user_highlights = user_highlights_db.get(user_id, [])
        chapter_highlights = [
            h for h in user_highlights
            if h.get('chapter_id') == chapter_id
        ]

        return {"highlights": chapter_highlights}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get highlights error: {str(e)}")


@app.post("/api/ingest")
async def ingest_content():
    """Ingest book content into the vector store"""
    try:
        from src.services.content_loader import ContentLoader

        # Create content loader instance
        loader = ContentLoader()

        # Load the book content into the vector store
        success = await loader.load_book_content(force_reload=True)

        if success:
            return {
                "status": "success",
                "message": "Book content ingested successfully into vector store"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest content")
    except Exception as e:
        print(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)