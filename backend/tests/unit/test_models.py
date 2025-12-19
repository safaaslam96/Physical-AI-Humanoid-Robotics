"""
Unit tests for the data models in the Physical AI and Humanoid Robotics book platform
"""
import pytest
from datetime import datetime
from src.models.user import User
from src.models.profile import Profile, BackgroundLevel


def test_user_model_creation():
    """Test creating a user model"""
    user = User(
        id="user123",
        email="test@example.com",
        username="testuser",
        is_active=True
    )

    assert user.email == "test@example.com"
    assert user.username == "testuser"
    assert user.is_active is True


def test_profile_model_creation():
    """Test creating a profile model"""
    profile = Profile(
        user_id="user123",
        software_background=BackgroundLevel.intermediate,
        hardware_background=BackgroundLevel.beginner,
        learning_goals=["learn robotics", "understand AI"],
        experience_level=BackgroundLevel.intermediate
    )

    assert profile.user_id == "user123"
    assert profile.software_background == BackgroundLevel.intermediate
    assert profile.hardware_background == BackgroundLevel.beginner
    assert len(profile.learning_goals) == 2
    assert profile.experience_level == BackgroundLevel.intermediate


def test_profile_default_values():
    """Test profile model default values"""
    profile = Profile(user_id="user123")

    assert profile.software_background == BackgroundLevel.beginner
    assert profile.hardware_background == BackgroundLevel.beginner
    assert profile.experience_level == BackgroundLevel.beginner
    assert profile.preferred_content_type == ["text", "code", "diagrams"]