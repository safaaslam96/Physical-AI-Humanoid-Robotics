"""
RAG (Retrieval Augmented Generation) service for the Physical AI and Humanoid Robotics book platform
"""
import google.generativeai as genai
from typing import List, Dict, Optional
from ..core.vector_store import VectorStore
from ..core.config import settings


class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

    async def query_book_content(self, query: str, user_id: Optional[str] = None) -> Dict:
        """Query book content using RAG approach"""
        # Search for relevant content in the vector store
        search_results = self.vector_store.search(query)

        # Prepare context from search results
        context = "\n".join([result["content"] for result in search_results])

        # Generate response using Gemini
        prompt = f"""
        Based on the following context, answer the user's question.
        If the context doesn't contain enough information to answer the question,
        say "I don't have enough information from the book to answer this question."

        Context: {context}

        Question: {query}

        Answer:
        """

        response = self.model.generate_content(prompt)

        return {
            "response": response.text,
            "sources": search_results,
            "query": query
        }

    async def query_selected_text(self, selected_text: str, query: str) -> Dict:
        """Query specifically from selected text only"""
        # Generate response based only on the selected text
        prompt = f"""
        Based on the following text, answer the user's question.
        Only use information from this text to answer.

        Selected text: {selected_text}

        Question: {query}

        Answer:
        """

        response = self.model.generate_content(prompt)

        return {
            "response": response.text,
            "sources": [{"content": selected_text, "type": "selected_text"}],
            "query": query
        }

    async def add_content_to_vector_store(self, content_id: str, content: str, metadata: Dict = None):
        """Add content to vector store for retrieval"""
        self.vector_store.add_document(content_id, content, metadata)