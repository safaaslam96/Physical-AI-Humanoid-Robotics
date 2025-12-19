"""
Configuration settings for the Physical AI and Humanoid Robotics book platform
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database settings
    database_url: Optional[str] = None
    neon_database_url: Optional[str] = None

    # Qdrant settings
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "book_content"

    # Google Gemini settings
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-pro-flash"

    # Content directory settings
    content_directory: str = "./docusaurus/docs"

    # Application settings
    app_name: str = "Physical AI and Humanoid Robotics Book"
    app_version: str = "1.0.0"
    debug: bool = False
    secret_key: str = "your-secret-key-here"  # In production, this should be in environment variables

    # Frontend settings
    frontend_url: str = "http://localhost:3000"
    backend_cors_origins: list = ["http://localhost:3000", "http://localhost:3001", "http://localhost:8080"]

    class Config:
        env_file = ".env"


settings = Settings()