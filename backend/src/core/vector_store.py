"""
Vector store integration for the Physical AI and Humanoid Robotics book platform
"""
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional
from .config import settings


class VectorStore:
    def __init__(self):
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            # For development, you might want to use local Qdrant
            self.client = QdrantClient(":memory:")  # In-memory for development

        self.collection_name = settings.qdrant_collection_name
        self._initialize_collection()

        # Initialize Google Generative AI for embeddings
        genai.configure(api_key=settings.gemini_api_key)
        self.embedding_model = settings.gemini_embedding_model

    def _initialize_collection(self):
        """Initialize the Qdrant collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
        except:
            # Create collection if it doesn't exist
            # Using size 768 for text-embedding-004 model
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Google Gemini"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Fallback to a zero vector if embedding fails
            return [0.0] * 768

    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """Add a document to the vector store"""
        embedding = self._get_embedding(content)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": content,
                        "metadata": metadata or {}
                    }
                )
            ]
        )

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar documents"""
        query_embedding = self._get_embedding(query)

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "content": result.payload.get("content", ""),
                "metadata": result.payload.get("metadata", {}),
                "score": result.score
            })

        return results

    def search_with_filter(self, query: str, filters: Dict = None, limit: int = 5) -> List[Dict]:
        """Search for similar documents with optional filters"""
        query_embedding = self._get_embedding(query)

        # Build filter conditions if provided
        if filters:
            try:
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=value)
                        )
                    )

                query_filter = models.Filter(
                    must=filter_conditions
                )
            except Exception:
                # Fallback if filtering fails
                query_filter = None
        else:
            query_filter = None

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit
        )

        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "content": result.payload.get("content", ""),
                "metadata": result.payload.get("metadata", {}),
                "score": result.score
            })

        return results