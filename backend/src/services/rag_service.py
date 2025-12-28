"""
RAG (Retrieval Augmented Generation) service for the Physical AI and Humanoid Robotics book platform
"""
import google.generativeai as genai
import re
from typing import List, Dict, Optional
from ..core.vector_store import VectorStore
from ..core.config import settings


class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

    def _detect_language(self, text: str) -> str:
        """Detect if text contains Urdu (Arabic script) or English"""
        # Check for Urdu/Arabic Unicode range
        urdu_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
        if re.search(urdu_pattern, text):
            return "urdu"
        return "english"

    async def query_book_content(self, query: str, user_id: Optional[str] = None) -> Dict:
        """Query book content using RAG approach"""
        # Detect question language
        query_language = self._detect_language(query)

        # Search for relevant content in the vector store with increased limit for more context
        search_results = self.vector_store.search(query, limit=10)

        # Prepare context from search results with enhanced metadata
        context_parts = []
        sources = []

        for idx, result in enumerate(search_results, 1):
            content = result["content"]
            metadata = result.get("metadata", {})

            if content.strip():  # Only add non-empty content
                # Add structured context with metadata
                page_title = metadata.get("page_title", "Unknown")
                chapter = metadata.get("chapter", "")

                # Format context with clear structure
                context_entry = f"[Source {idx}] {page_title}\n{content}"
                context_parts.append(context_entry)

                # Add source information
                source_info = {
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "source_file": metadata.get("source", "Unknown"),
                    "page_title": page_title,
                    "chapter": chapter,
                    "url": metadata.get("url", ""),
                    "score": result.get("score", 0.0)
                }
                sources.append(source_info)

        if not context_parts:
            # If no relevant content found, provide a helpful response in the appropriate language
            if query_language == "urdu":
                response_text = (
                    f"مجھے '{query}' کے بارے میں کتاب میں مخصوص معلومات نہیں مل سکیں۔ "
                    f"براہ کرم Physical AI & Humanoid Robotics کتاب کے متعلقہ ابواب دیکھیں، "
                    f"یا اپنا سوال دوسرے الفاظ میں پوچھنے کی کوشش کریں۔"
                )
            else:
                response_text = (
                    f"I couldn't find specific information about '{query}' in the book content. "
                    f"Please check the relevant chapters in the Physical AI & Humanoid Robotics book, "
                    f"or try rephrasing your question."
                )
        else:
            context = "\n\n".join(context_parts)

            # Language-specific instruction
            if query_language == "urdu":
                language_instruction = """
CRITICAL LANGUAGE REQUIREMENT: The user's question is in Urdu. You MUST respond COMPLETELY in Urdu (اردو).
Do NOT mix English words or phrases. Write your entire response in Urdu script only.
"""
            else:
                language_instruction = """
CRITICAL LANGUAGE REQUIREMENT: The user's question is in English. You MUST respond COMPLETELY in English.
Do NOT use Urdu or any other language. Write your entire response in English only.
"""

            # Enhanced prompt for detailed, comprehensive responses
            prompt = f"""You are an expert teacher and assistant for the "Physical AI & Humanoid Robotics" book.
Your role is to provide DETAILED, COMPREHENSIVE, and EDUCATIONAL answers based on the book content.

{language_instruction}

INSTRUCTIONS FOR YOUR RESPONSE:

1. ANSWER LENGTH: Give a DETAILED and COMPLETE answer. Your response should be 3-5 paragraphs long.
   - Do NOT give short, single-line answers
   - Do NOT give brief snippets
   - Explain thoroughly like a teacher explaining to a student

2. USE ALL CONTEXT: You have been provided with multiple sources from the book below.
   - Read and synthesize information from ALL the provided sources
   - Combine related information from different sections
   - Reference specific concepts, examples, and details from the context

3. STRUCTURE YOUR ANSWER:
   - Start with a clear, direct answer to the question
   - Then provide detailed explanation with examples from the book
   - Include relevant technical details, frameworks, tools mentioned
   - End with practical implications or key takeaways

4. BE EDUCATIONAL: Explain concepts as if teaching, not just stating facts.
   - Define technical terms when you first use them
   - Provide context and background
   - Explain "why" and "how", not just "what"

5. LANGUAGE CONSISTENCY:
   - If the question is in English, respond ENTIRELY in English
   - If the question is in Urdu, respond ENTIRELY in Urdu
   - Never mix languages in your response

BOOK CONTEXT FROM MULTIPLE CHAPTERS:
{context}

USER'S QUESTION: {query}

Now provide a DETAILED, COMPREHENSIVE answer using ALL the context above. Remember to respond in the SAME language as the question, and make your answer thorough and educational (3-5 paragraphs).

YOUR DETAILED ANSWER:
"""

            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=2000,  # Increased for longer, detailed responses
                        temperature=0.7
                    )
                )
                response_text = response.text
            except Exception as e:
                print(f"Gemini generation error: {e}")
                if query_language == "urdu":
                    response_text = (
                        f"کتاب کے مواد کی بنیاد پر، مجھے '{query}' سے متعلق کچھ معلومات ملیں۔ "
                        f"براہ کرم تفصیلی معلومات کے لیے Physical AI & Humanoid Robotics کتاب کے متعلقہ ابواب دیکھیں۔"
                    )
                else:
                    response_text = (
                        f"Based on the book content, I found some information related to '{query}'. "
                        f"Please refer to the relevant chapters in the Physical AI & Humanoid Robotics book for detailed information."
                    )

        return {
            "response": response_text,
            "sources": sources,
            "query": query
        }

    async def query_selected_text(self, selected_text: str, query: str) -> Dict:
        """Query specifically from selected text only"""
        if not selected_text or not selected_text.strip():
            return {
                "response": "Please select some text to ask a question about.",
                "sources": [],
                "query": query
            }

        # Detect language for consistency
        query_language = self._detect_language(query)

        # Language-specific instruction
        if query_language == "urdu":
            language_instruction = """
CRITICAL: The user's question is in Urdu. Respond COMPLETELY in Urdu (اردو) only.
"""
        else:
            language_instruction = """
CRITICAL: The user's question is in English. Respond COMPLETELY in English only.
"""

        # Generate detailed response based on selected text
        prompt = f"""You are an expert teacher for the "Physical AI & Humanoid Robotics" book.
The user has highlighted/selected specific text from the book and wants to understand it better.

{language_instruction}

INSTRUCTIONS:
1. Provide a DETAILED explanation (2-3 paragraphs)
2. Explain the concepts thoroughly like a teacher
3. Define any technical terms
4. Give examples or context where relevant
5. Respond in the SAME language as the user's question

SELECTED TEXT FROM BOOK:
{selected_text}

USER'S QUESTION: {query}

Provide a DETAILED, educational answer about the selected text:
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1500,  # Increased for detailed responses
                    temperature=0.7
                )
            )
            response_text = response.text
        except Exception as e:
            print(f"Gemini generation error for selected text: {e}")
            if query_language == "urdu":
                response_text = (
                    f"منتخب متن کے بارے میں: '{selected_text[:100]}...', "
                    f"یہ آپ کے سوال '{query}' سے متعلق ہے۔ "
                    f"براہ کرم مزید تفصیلات کے لیے Physical AI & Humanoid Robotics کتاب میں اس متن کے ارد گرد کا سیاق و سباق دیکھیں۔"
                )
            else:
                response_text = (
                    f"Regarding the selected text: '{selected_text[:100]}...', "
                    f"I can confirm this relates to your question '{query}'. "
                    f"Please refer to the context around this text in the Physical AI & Humanoid Robotics book for more details."
                )

        return {
            "response": response_text,
            "sources": [{"content": selected_text[:200] + "..." if len(selected_text) > 200 else selected_text, "type": "selected_text"}],
            "query": query
        }

    async def add_content_to_vector_store(self, content_id: str, content: str, metadata: Dict = None):
        """Add content to vector store for retrieval"""
        self.vector_store.add_document(content_id, content, metadata)