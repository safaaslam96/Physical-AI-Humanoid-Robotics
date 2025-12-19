"""
Vector store integration for the Physical AI and Humanoid Robotics book platform
"""
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

    def _initialize_collection(self):
        """Initialize the Qdrant collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
        except:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )

    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """Add a document to the vector store"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(content).tolist()

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
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query).tolist()

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