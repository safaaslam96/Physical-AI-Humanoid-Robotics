"""
Chat interaction model for tracking conversations with the RAG chatbot
"""
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime


class ChatInteraction(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: str
    query: str
    response: str
    sources: List[Dict] = []
    timestamp: datetime = datetime.now()
    is_from_selected_text: bool = False
    selected_text: Optional[str] = None