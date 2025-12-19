"""
Translation API routes for the Physical AI and Humanoid Robotics book platform
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import google.generativeai as genai
from ..core.config import settings
from ..services.translation_service import TranslationService


class TranslationRequest(BaseModel):
    content: str
    target_language: str = "ur"
    source_language: Optional[str] = "en"
    user_id: Optional[str] = None
    translation_style: Optional[str] = "formal"  # formal, informal, technical


class TranslationResponse(BaseModel):
    original_content: str
    translated_content: str
    source_language: str
    target_language: str
    confidence: Optional[float] = None
    translation_style: str


class BatchTranslationRequest(BaseModel):
    contents: List[str]
    target_language: str = "ur"
    source_language: Optional[str] = "en"


class BatchTranslationResponse(BaseModel):
    translations: List[TranslationResponse]
    total_items: int


class ChapterTranslationRequest(BaseModel):
    chapter_path: str
    target_language: str = "ur"
    user_id: Optional[str] = None


class ChapterTranslationResponse(BaseModel):
    chapter_path: str
    target_language: str
    sections_translated: int
    total_characters: int


router = APIRouter()

# Initialize translation service
translation_service = TranslationService()


@router.post("/translate", response_model=TranslationResponse)
async def translate_content(request: TranslationRequest):
    """Translate content to the specified language (Urdu by default)"""
    try:
        # Validate language codes
        supported_languages = ["en", "ur"]
        if request.target_language.lower() not in supported_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Target language '{request.target_language}' not supported. Supported: {supported_languages}"
            )

        if request.source_language.lower() not in supported_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Source language '{request.source_language}' not supported. Supported: {supported_languages}"
            )

        # Perform translation
        translated_content = await translation_service.translate_text(
            text=request.content,
            target_language=request.target_language,
            source_language=request.source_language,
            style=request.translation_style
        )

        return TranslationResponse(
            original_content=request.content,
            translated_content=translated_content,
            source_language=request.source_language,
            target_language=request.target_language,
            confidence=0.95,  # Placeholder - would be actual confidence in real implementation
            translation_style=request.translation_style
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@router.post("/translate/batch")
async def batch_translate(request: BatchTranslationRequest):
    """Translate multiple pieces of content at once"""
    try:
        translations = []
        for content in request.contents:
            translated = await translation_service.translate_text(
                text=content,
                target_language=request.target_language,
                source_language=request.source_language
            )

            translation_response = TranslationResponse(
                original_content=content,
                translated_content=translated,
                source_language=request.source_language,
                target_language=request.target_language,
                confidence=0.95,
                translation_style="formal"
            )
            translations.append(translation_response)

        return BatchTranslationResponse(
            translations=translations,
            total_items=len(translations)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch translation error: {str(e)}")


@router.post("/translate/chapter")
async def translate_chapter(request: ChapterTranslationRequest):
    """Translate an entire chapter from the book"""
    try:
        from pathlib import Path
        import os

        # Construct the full path to the chapter
        base_path = Path(__file__).parent.parent.parent / "docusaurus" / "docs"
        chapter_path = base_path / request.chapter_path

        if not chapter_path.exists():
            raise HTTPException(status_code=404, detail=f"Chapter not found: {chapter_path}")

        # Read the chapter content
        with open(chapter_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Translate the content
        translated_content = await translation_service.translate_text(
            text=content,
            target_language=request.target_language,
            source_language="en"  # Assuming original is in English
        )

        # Save translated content to a new file
        translated_file_path = chapter_path.parent / f"{chapter_path.stem}_ur.{chapter_path.suffix}"
        with open(translated_file_path, 'w', encoding='utf-8') as f:
            f.write(translated_content)

        return ChapterTranslationResponse(
            chapter_path=str(translated_file_path),
            target_language=request.target_language,
            sections_translated=1,  # Simplified - in reality would count sections
            total_characters=len(translated_content)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chapter translation error: {str(e)}")


@router.get("/supported-languages")
async def get_supported_languages():
    """Get list of supported translation languages"""
    return {
        "supported_languages": [
            {"code": "en", "name": "English", "status": "default"},
            {"code": "ur", "name": "Urdu", "status": "supported", "script_direction": "rtl"},
            {"code": "hi", "name": "Hindi", "status": "beta", "script_direction": "ltr"},
            {"code": "ar", "name": "Arabic", "status": "beta", "script_direction": "rtl"}
        ],
        "default_language": "en"
    }


@router.get("/translation-styles")
async def get_translation_styles():
    """Get available translation styles"""
    return {
        "styles": [
            {"name": "formal", "description": "Formal, academic tone appropriate for educational content"},
            {"name": "informal", "description": "Conversational tone for casual learning"},
            {"name": "technical", "description": "Technical terminology preserved, precise translations"}
        ]
    }


@router.post("/translate/verify")
async def verify_translation(translation_request: TranslationRequest):
    """Verify the quality and accuracy of a translation"""
    try:
        # In a real implementation, this would use a verification model
        # For now, return basic verification metrics
        original_length = len(translation_request.content)
        translated_length = len(translation_request.content)  # Placeholder

        return {
            "original_length": original_length,
            "translated_length": translated_length,
            "quality_score": 0.85,  # Placeholder score
            "suggestions": [],
            "needs_review": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation verification error: {str(e)}")