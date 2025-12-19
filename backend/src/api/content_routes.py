"""
Content management API routes for the Physical AI and Humanoid Robotics book platform
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from ..services.content_loader import ContentLoader
import asyncio


class LoadContentRequest(BaseModel):
    force_reload: bool = False
    content_path: Optional[str] = None


class LoadContentResponse(BaseModel):
    success: bool
    message: str
    documents_loaded: int
    details: Optional[Dict] = None


class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5


class SearchResponse(BaseModel):
    results: List[Dict]
    query: str
    total_results: int


class ContentStatsResponse(BaseModel):
    total_documents: int
    indexed_parts: List[str]
    last_update: Optional[str]
    details: Optional[Dict] = None


router = APIRouter()
content_loader = ContentLoader()


@router.post("/content/load", response_model=LoadContentResponse)
async def load_book_content(request: LoadContentRequest):
    """Load book content into the vector store"""
    try:
        # Update content directory if provided
        if request.content_path:
            from ..core.config import settings
            settings.content_directory = request.content_path

        # Load content
        success = await content_loader.load_book_content(force_reload=request.force_reload)

        if success:
            return LoadContentResponse(
                success=True,
                message="Book content loaded successfully",
                documents_loaded=0,  # We don't track exact count in this implementation
                details={"force_reload": request.force_reload}
            )
        else:
            return LoadContentResponse(
                success=False,
                message="Failed to load book content",
                documents_loaded=0
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content loading error: {str(e)}")


@router.post("/content/search", response_model=SearchResponse)
async def search_content(request: SearchRequest):
    """Search for content in the book"""
    try:
        results = await content_loader.search_content(request.query, request.limit)
        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.get("/content/stats", response_model=ContentStatsResponse)
async def get_content_stats():
    """Get statistics about loaded content"""
    try:
        stats = await content_loader.get_content_stats()
        return ContentStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval error: {str(e)}")


@router.post("/content/load-chapter")
async def load_specific_chapter(chapter_path: str, title: str = None):
    """Load a specific chapter into the vector store"""
    try:
        import os
        from pathlib import Path

        # Construct the full path
        base_path = Path(__file__).parent.parent.parent / "docusaurus" / "docs"
        full_path = base_path / chapter_path

        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Chapter file not found: {full_path}")

        # Read the chapter content
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Generate a document ID
        import hashlib
        doc_id = hashlib.md5(f"chapter_{chapter_path}".encode()).hexdigest()

        # Prepare metadata
        metadata = {
            "source_file": chapter_path,
            "title": title or chapter_path,
            "type": "chapter",
            "added_at": "datetime.now().isoformat()"  # In real implementation, use actual datetime
        }

        # Load into vector store
        await content_loader.load_specific_content(doc_id, content, metadata)

        return {
            "success": True,
            "message": f"Chapter {chapter_path} loaded successfully",
            "doc_id": doc_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chapter loading error: {str(e)}")