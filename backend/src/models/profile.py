"""
Profile model for user background and learning goals
"""
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class BackgroundLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class Profile(BaseModel):
    id: Optional[str] = None
    user_id: str
    software_background: BackgroundLevel = BackgroundLevel.beginner
    hardware_background: BackgroundLevel = BackgroundLevel.beginner
    learning_goals: List[str] = []
    experience_level: BackgroundLevel = BackgroundLevel.beginner
    preferred_content_type: List[str] = ["text", "code", "diagrams"]