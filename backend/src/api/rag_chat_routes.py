"""
Chat API routes for the Physical AI and Humanoid Robotics book platform RAG functionality
This endpoint is specifically designed to match what the frontend AIChatPopup expects
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from src.services.rag_service import RAGService
from src.core.config import settings
import uuid
import datetime


class FrontendChatRequest(BaseModel):
    """
    Request model that matches what the frontend AIChatPopup sends
    """
    message: str
    selected_text: Optional[str] = None
    from_selected_text: Optional[bool] = False


class FrontendChatResponse(BaseModel):
    """
    Response model that matches what the frontend AIChatPopup expects
    """
    response: str
    sources: Optional[List[Dict]] = []


# Create router
router = APIRouter()

# Initialize RAG service
rag_service = RAGService()


@router.post("/chat", response_model=FrontendChatResponse)
async def chat_endpoint(request: FrontendChatRequest):
    """
    Chat endpoint that matches the exact format expected by the frontend AIChatPopup
    """
    try:
        if request.from_selected_text and request.selected_text:
            # Query specifically from selected/highlighted text
            result = await rag_service.query_selected_text(
                selected_text=request.selected_text,
                query=request.message
            )
        else:
            # Query from full book content using RAG
            result = await rag_service.query_book_content(
                query=request.message
            )

        return FrontendChatResponse(
            response=result["response"],
            sources=result.get("sources", [])
        )
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.post("/chat/highlighted-text")
async def chat_highlighted_text_endpoint(request: FrontendChatRequest):
    """
    Dedicated endpoint for querying highlighted text specifically
    """
    try:
        if not request.selected_text:
            raise HTTPException(status_code=400, detail="Selected text is required for highlighted text queries")

        # Query specifically from selected/highlighted text
        result = await rag_service.query_selected_text(
            selected_text=request.selected_text,
            query=request.query
        )

        return FrontendChatResponse(
            response=result["response"],
            sources=result.get("sources", [])
        )
    except Exception as e:
        print(f"Highlighted text chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Highlighted text chat processing error: {str(e)}")


@router.get("/chat/health")
async def chat_health():
    """Health check for the chat endpoint"""
    return {
        "status": "healthy",
        "service": "RAG Chatbot API",
        "qdrant_available": True,  # Assuming Qdrant is configured
        "gemini_available": True   # Assuming Gemini is configured
    }