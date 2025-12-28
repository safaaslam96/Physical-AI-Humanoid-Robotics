import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Qdrant client
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION_NAME", "book_content")

if not qdrant_url or not qdrant_api_key:
    print("ERROR: QDRANT_URL and/or QDRANT_API_KEY not found in environment!")
    exit(1)

try:
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    # Check if collection exists
    collections = client.get_collections()
    print(f"Available collections: {[col.name for col in collections.collections]}")

    # Get collection info
    collection_info = client.get_collection(collection_name)
    print(f"Collection '{collection_name}' info:")
    print(f"  - Points count: {collection_info.points_count}")
    print(f"  - Vector size: {collection_info.config.params.vectors.size}")
    print(f"  - Distance: {collection_info.config.params.vectors.distance}")

    # Get a few sample points to verify content
    if collection_info.points_count > 0:
        print(f"\nSample points from collection '{collection_name}':")

        # Get first few points
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=5,
            with_payload=True,
            with_vectors=False
        )

        points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points
        for i, point in enumerate(points):
            payload = point.payload
            print(f"\nPoint {i+1}:")
            print(f"  ID: {point.id}")
            print(f"  Content length: {len(payload.get('content', ''))} chars")
            print(f"  Source: {payload.get('source', 'N/A')}")
            print(f"  Page title: {payload.get('page_title', 'N/A')}")
            print(f"  URL: {payload.get('url', 'N/A')}")
            print(f"  Chunk ID: {payload.get('chunk_id', 'N/A')}")

    # Get list of all unique sources (markdown files)
    if collection_info.points_count > 0:
        # Get all points and collect unique sources
        all_sources = set()
        offset = None
        while True:
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=100,  # Process in batches to avoid memory issues
                with_payload=True,
                with_vectors=False,
                offset=offset
            )

            points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points
            if not points:
                break

            for point in points:
                source = point.payload.get('source', '')
                if source:
                    all_sources.add(source)

            # Use the last point as the offset (for newer versions of qdrant_client)
            offset = scroll_result.next_page_offset if hasattr(scroll_result, 'next_page_offset') else (points[-1].id + 1 if points else None)
            if len(points) < 100:  # If we got fewer than limit, we've reached the end
                break

        print(f"\nTotal unique source files in vector DB: {len(all_sources)}")
        print("Source files:")
        for source in sorted(all_sources):
            print(f"  - {source}")

except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    import traceback
    traceback.print_exc()