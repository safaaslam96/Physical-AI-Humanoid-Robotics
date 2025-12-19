"""
Authentication API routes for the Physical AI and Humanoid Robotics book platform
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional
from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


class ProfileRequest(BaseModel):
    software_background: Optional[str] = "beginner"
    hardware_background: Optional[str] = "beginner"
    learning_goals: Optional[list] = []


router = APIRouter()


@router.post("/login")
async def login(request: LoginRequest):
    """User login endpoint"""
    # Integration with Better Auth will be handled here
    return {"message": "Login successful", "user_id": "user123"}


@router.post("/profile")
async def create_profile(request: ProfileRequest):
    """Create or update user profile with background information"""
    # Store user profile information for personalization
    return {
        "message": "Profile updated successfully",
        "profile": {
            "software_background": request.software_background,
            "hardware_background": request.hardware_background,
            "learning_goals": request.learning_goals
        }
    }