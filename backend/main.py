"""
Main FastAPI application for the Physical AI and Humanoid Robotics book platform
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import auth_routes, chat_routes, personalization_routes, translation_routes, content_routes
from src.core.config import settings


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routes
app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(chat_routes.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(personalization_routes.router, prefix="/api/v1/personalization", tags=["personalization"])
app.include_router(translation_routes.router, prefix="/api/v1/translation", tags=["translation"])
app.include_router(content_routes.router, prefix="/api/v1/content", tags=["content"])


@app.get("/")
async def root():
    return {"message": "Welcome to the Physical AI and Humanoid Robotics Book API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.app_version}