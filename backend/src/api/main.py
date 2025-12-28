from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Physical AI & Humanoid Robotics RAG Chatbot",
              description="RAG chatbot for the Physical AI & Humanoid Robotics book",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
collection_name = "physical_ai_book"

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Using the latest Gemini model
    temperature=0.1,
    max_tokens=1000
)

class ChatRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None
    history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[dict]

@app.get("/")
def read_root():
    return {"message": "Physical AI & Humanoid Robotics RAG Chatbot API"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that retrieves relevant context from the book and generates a response
    """
    try:
        # Build the query with selected text context if provided
        query = request.query
        if request.selected_text:
            query = f"Based on the selected text: '{request.selected_text}', {request.query}"

        # Retrieve relevant chunks from Qdrant
        relevant_chunks = retrieve_relevant_chunks(query)

        if not relevant_chunks:
            return ChatResponse(
                response="I couldn't find any relevant information in the book to answer your question.",
                sources=[]
            )

        # Build context from retrieved chunks
        context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])

        # Build the prompt with context and query
        if request.selected_text:
            # If selected text is provided, focus on explaining that specific content
            prompt = f"""
            You are an expert assistant for the Physical AI & Humanoid Robotics book.
            The user has selected/highlighted the following text and wants to understand it better:
            "{request.selected_text}"

            Here is additional context from the book:
            {context}

            Please explain the concept in the selected text in detail, using the context from the book to provide a comprehensive explanation.
            If the concept is not directly explained in the context, try to relate it to similar concepts in the book.

            Make sure to provide a clear and detailed explanation based on the book content, and list the relevant chapters/sections that you used to answer the question.
            """
        else:
            # Standard query without selected text
            prompt = f"""
            You are an expert assistant for the Physical AI & Humanoid Robotics book.
            Answer the user's question based only on the provided context from the book.
            If the answer cannot be found in the context, say "I couldn't find that information in the book."

            Context from the book:
            {context}

            User question: {query}

            Please provide a comprehensive answer based on the book content, and list the relevant chapters/sections that you used to answer the question.
            """

        # Generate response using Gemini
        response = llm.invoke(prompt)

        # Extract sources (unique page titles and URLs)
        sources = []
        seen_urls = set()
        for chunk in relevant_chunks:
            url = chunk.get("url", "")
            if url not in seen_urls:
                sources.append({
                    "title": chunk.get("page_title", "Unknown"),
                    "url": url
                })
                seen_urls.add(url)

        return ChatResponse(
            response=response.content,
            sources=sources
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

def retrieve_relevant_chunks(query: str, top_k: int = 5):
    """
    Retrieve relevant chunks from Qdrant based on the query
    """
    try:
        # Generate embedding for the query using Gemini
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        result = genai.embed_content(
            model=embedding_model,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = result['embedding']

        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )

        # Extract relevant chunks
        relevant_chunks = []
        for result in search_results:
            if result.payload:
                relevant_chunks.append({
                    "content": result.payload.get("content", ""),
                    "page_title": result.payload.get("page_title", ""),
                    "url": result.payload.get("url", ""),
                    "score": result.score
                })

        return relevant_chunks

    except Exception as e:
        logger.error(f"Error retrieving chunks from Qdrant: {str(e)}")
        return []

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)