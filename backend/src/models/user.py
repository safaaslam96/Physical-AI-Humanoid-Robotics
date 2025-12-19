"""
User model for the Physical AI and Humanoid Robotics book platform
"""
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class User(BaseModel):
    id: Optional[str] = None
    email: str
    username: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True