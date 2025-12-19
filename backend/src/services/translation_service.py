"""
Translation service for the Physical AI and Humanoid Robotics book platform
"""
import google.generativeai as genai
from typing import Optional, Dict, List
from ..core.config import settings
import asyncio
import re


class TranslationService:
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

    async def translate_text(self, text: str, target_language: str = "ur", source_language: str = "en", style: str = "formal") -> str:
        """Translate text using Gemini AI"""

        # Always use Gemini for translation
        return await self._translate_with_gemini(text, target_language, source_language, style)

    async def _translate_with_gemini(self, text: str, target_language: str, source_language: str, style: str) -> str:
        """Translate using Gemini AI with proper context and style"""
        # Create a detailed translation prompt
        language_names = {
            "ur": "Urdu",
            "en": "English",
            "hi": "Hindi",
            "ar": "Arabic",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean"
        }

        target_lang_name = language_names.get(target_language.lower(), target_language.upper())
        source_lang_name = language_names.get(source_language.lower(), source_language.upper())

        prompt = f"""
        You are an expert translator specializing in technical and educational content.
        Translate the following text from {source_lang_name} to {target_lang_name}.
        Maintain the educational and technical accuracy of the content.
        Use a {style} tone appropriate for academic content about robotics and AI.
        Preserve technical terminology and concepts.
        For robotics/AI specific terms, use equivalent terms in the target language or keep the original term if no good translation exists.

        Text to translate:
        {text}

        Translation in {target_lang_name}:
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more consistent translations
                    max_output_tokens=2000
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"[GEMINI TRANSLATION ERROR: {str(e)}] Original text: {text}"

    async def _translate_urdu_with_gemini(self, text: str, style: str) -> str:
        """Special handling for Urdu translation using Gemini"""
        prompt = f"""
        You are an expert translator specializing in technical and educational content.
        Translate the following text to Urdu with proper Urdu script and grammar.
        This is for educational content about robotics and AI.
        Use a {style} tone appropriate for academic content.
        Maintain technical accuracy and preserve important terminology.
        Use proper Urdu sentence structure and grammar.
        Keep all code blocks, technical terms, and proper nouns in English.

        Text to translate:
        {text}

        Translation in Urdu:
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2000
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"[URDU TRANSLATION ERROR: {str(e)}] Original text: {text}"

    async def translate_to_urdu(self, content: str) -> str:
        """Translate content to Urdu while preserving technical accuracy - legacy method"""
        return await self._translate_urdu_with_gemini(content, "formal")

    async def translate_preserving_code(self, content: str) -> str:
        """Translate content while specifically preserving code blocks and technical terms - legacy method"""
        return await self._translate_urdu_with_gemini(content, "formal")

    async def translate_document(self, content: str, target_language: str = "ur", source_language: str = "en") -> Dict:
        """Translate an entire document with metadata"""
        # Split content into chunks to handle large documents
        chunks = self._split_content_into_chunks(content, max_chunk_size=1000)

        translated_chunks = []
        for chunk in chunks:
            translated_chunk = await self.translate_text(
                chunk,
                target_language,
                source_language
            )
            translated_chunks.append(translated_chunk)

        full_translation = " ".join(translated_chunks)

        return {
            "original_length": len(content),
            "translated_length": len(full_translation),
            "chunks_processed": len(chunks),
            "target_language": target_language,
            "source_language": source_language,
            "translated_content": full_translation
        }

    def _split_content_into_chunks(self, content: str, max_chunk_size: int = 1000) -> list:
        """Split content into chunks of specified size while preserving sentences"""
        sentences = content.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def get_translation_quality_score(self, original: str, translated: str) -> Dict:
        """Get quality metrics for a translation"""
        # This would implement more sophisticated quality checking in a real system
        original_length = len(original)
        translated_length = len(translated)

        # Simple length ratio check (in real system, would use more sophisticated metrics)
        length_ratio = abs(len(translated) - len(original)) / max(len(original), 1)

        # Quality score based on various factors
        quality_score = max(0.0, min(1.0, 1.0 - length_ratio * 0.5))

        return {
            "quality_score": quality_score,
            "original_length": original_length,
            "translated_length": translated_length,
            "length_ratio": length_ratio,
            "needs_review": quality_score < 0.7
        }