"""
Test script to verify the RAG functionality with the existing book content
"""
import asyncio
from src.services.rag_service import RAGService
from src.core.vector_store import VectorStore

async def test_rag_functionality():
    print("Testing RAG functionality...")

    # Initialize services
    rag_service = RAGService()
    vector_store = VectorStore()

    print("✓ RAG Service and Vector Store initialized")

    # Test 1: Check if we can search the vector store
    print("\nTest 1: Testing vector store search...")
    try:
        results = vector_store.search("Physical AI", limit=3)
        print(f"✓ Search completed, found {len(results)} results")
        if results:
            print(f"  First result preview: {results[0]['content'][:100]}...")
    except Exception as e:
        print(f"✗ Search failed: {e}")

    # Test 2: Test book content query
    print("\nTest 2: Testing book content query...")
    try:
        result = await rag_service.query_book_content("What is Physical AI?")
        print(f"✓ Book content query completed")
        print(f"  Response preview: {result['response'][:100]}...")
        print(f"  Sources found: {len(result['sources'])}")
    except Exception as e:
        print(f"✗ Book content query failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Test selected text query
    print("\nTest 3: Testing selected text query...")
    try:
        sample_text = "Physical AI extends beyond digital spaces into the physical world. This capstone quarter introduces Physical AI—AI systems that function in reality and comprehend physical laws."
        result = await rag_service.query_selected_text(
            selected_text=sample_text,
            query="Explain Physical AI"
        )
        print(f"✓ Selected text query completed")
        print(f"  Response preview: {result['response'][:100]}...")
    except Exception as e:
        print(f"✗ Selected text query failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nRAG functionality test completed!")

if __name__ == "__main__":
    asyncio.run(test_rag_functionality())