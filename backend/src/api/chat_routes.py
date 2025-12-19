"""
Chat API routes for the Physical AI and Humanoid Robotics book platform
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from ..services.rag_service import RAGService
from ..core.config import settings
import uuid


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    from_selected_text: Optional[bool] = False
    selected_text: Optional[str] = None
    context_window: Optional[int] = 5  # Number of previous messages to include
    temperature: Optional[float] = 0.7  # AI creativity parameter


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]
    session_id: str
    query: str
    timestamp: str
    tokens_used: Optional[Dict] = None


class ChatHistoryRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    messages: List[Dict]
    session_id: str


class ChatSessionRequest(BaseModel):
    user_id: Optional[str] = None
    session_name: Optional[str] = None


class ChatSessionResponse(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    created_at: str
    name: Optional[str] = None


router = APIRouter()

# Initialize RAG service
rag_service = RAGService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for interacting with the RAG chatbot"""
    try:
        import datetime

        session_id = request.session_id or str(uuid.uuid4())

        if request.from_selected_text and request.selected_text:
            # Query specifically from selected text
            result = await rag_service.query_selected_text(
                selected_text=request.selected_text,
                query=request.message
            )
        else:
            # Query from full book content
            result = await rag_service.query_book_content(
                query=request.message,
                user_id=request.user_id
            )

        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            session_id=session_id,
            query=request.message,
            timestamp=datetime.datetime.now().isoformat(),
            tokens_used={"input": len(request.message.split()), "output": len(result["response"].split())}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.post("/chat/selected-text")
async def chat_selected_text(request: ChatRequest):
    """Chat endpoint for interacting with only the selected text"""
    if not request.selected_text:
        raise HTTPException(status_code=400, detail="Selected text is required")

    try:
        import datetime

        session_id = request.session_id or str(uuid.uuid4())

        result = await rag_service.query_selected_text(
            selected_text=request.selected_text,
            query=request.message
        )

        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            session_id=session_id,
            query=request.message,
            timestamp=datetime.datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Selected text chat processing error: {str(e)}")


@router.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get details of a specific chat session"""
    # In a real implementation, this would fetch from database
    # For now, returning mock data
    return {
        "session_id": session_id,
        "title": f"Chat about Physical AI - {session_id[:8]}",
        "created_at": "2024-01-01T00:00:00Z",
        "message_count": 0,
        "last_activity": "2024-01-01T00:00:00Z"
    }


@router.get("/chat/sessions/user/{user_id}")
async def get_user_sessions(user_id: str):
    """Get all chat sessions for a user"""
    # In a real implementation, this would fetch from database
    # For now, returning mock data
    return {
        "user_id": user_id,
        "sessions": [
            {
                "session_id": "session123",
                "title": "Introduction to Physical AI",
                "created_at": "2024-01-01T00:00:00Z",
                "message_count": 5,
                "last_activity": "2024-01-01T01:00:00Z"
            }
        ]
    }


@router.post("/chat/session")
async def create_chat_session(request: ChatSessionRequest):
    """Create a new chat session"""
    import datetime

    session_id = str(uuid.uuid4())
    return ChatSessionResponse(
        session_id=session_id,
        user_id=request.user_id,
        created_at=datetime.datetime.now().isoformat(),
        name=request.session_name or f"Chat Session - {session_id[:8]}"
    )


@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    # In a real implementation, this would fetch from database
    # For now, returning mock data
    return ChatHistoryResponse(
        messages=[],
        session_id=session_id
    )


@router.post("/chat/reset-session")
async def reset_chat_session(request: ChatSessionRequest):
    """Reset a chat session"""
    session_id = request.session_id or str(uuid.uuid4())
    return {"session_id": session_id, "status": "reset", "message": "Session reset successfully"}


@router.get("/chat/models")
async def get_available_models():
    """Get available AI models for chat"""
    return {
        "default_model": settings.gemini_model,
        "available_models": [
            {"name": "gemini-2.5-pro-flash", "description": "Fast and efficient model"},
            {"name": "gemini-2.5-pro", "description": "More capable model"},
            {"name": "gemini-ultra", "description": "Most capable model"}
        ]
    }