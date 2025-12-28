"""
New RAG Ingestion Pipeline for Physical AI & Humanoid Robotics Book
Pipeline: Docs → Neon Postgres → Fetch → Embed → JSON → Qdrant
"""
import os
import logging
import json
import re
from typing import List, Dict, Tuple
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import glob
from natsort import natsorted
from src.core.database import NeonDatabase

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BookIngestor:
    """
    New RAG Ingestion Pipeline:
    1. Read docs from docusaurus/docs
    2. Inject into Neon Postgres
    3. Fetch from Neon
    4. Embed with Gemini
    5. Save to JSON
    6. Upload to Qdrant
    """
    def __init__(self):
        # Initialize Gemini for embeddings
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        genai.configure(api_key=gemini_api_key)
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

        # Initialize Neon Database
        logger.info("Connecting to Neon Postgres...")
        self.db = NeonDatabase()

        # Initialize Qdrant client
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            logger.error("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
            raise ValueError("Missing required environment variables for Qdrant connection")

        try:
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )

            # Test the connection
            self.qdrant_client.get_collections()
            logger.info("Successfully connected to Qdrant")

        except Exception as client_error:
            logger.error(f"Failed to initialize Qdrant client: {str(client_error)}")
            logger.error("Please check your QDRANT_URL and QDRANT_API_KEY environment variables")
            raise client_error

        # Collection name
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "physical_ai_book")
        self._create_collection()

    def _create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.warning(f"Collection {self.collection_name} does not exist, creating...")

            try:
                # Test embedding to get the correct dimensions
                test_embedding = genai.embed_content(
                    model=self.embedding_model,
                    content="test",
                    task_type="RETRIEVAL_DOCUMENT"
                )['embedding']

                embedding_size = len(test_embedding)
                logger.info(f"Using embedding model with {embedding_size} dimensions")

                # Create new collection with correct dimensions
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection {self.collection_name}")
            except Exception as create_error:
                logger.error(f"Failed to create collection {self.collection_name}: {str(create_error)}")
                raise create_error

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def extract_text_from_markdown(self, file_path: str) -> str:
        """Extract plain text from markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Convert markdown to HTML
        html = markdown.markdown(content)

        # Extract plain text from HTML
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()

        return text

    def get_all_markdown_files(self, docs_dir: str) -> List[Tuple[str, int]]:
        """Get all markdown files from docs directory with proper chapter ordering
        Returns: List of tuples (file_path, chapter_order)
        """
        # Use forward slashes for cross-platform compatibility and ensure proper pattern
        pattern = os.path.join(docs_dir, "**", "*.md*").replace(os.sep, "/")
        files = glob.glob(pattern, recursive=True)
        # Also try with .mdx files specifically
        pattern_mdx = os.path.join(docs_dir, "**", "*.mdx").replace(os.sep, "/")
        files_mdx = glob.glob(pattern_mdx, recursive=True)
        all_files = [f for f in (files + files_mdx) if os.path.isfile(f)]

        # Sort files naturally (handles Chapter 1, 2, ..., 10, 11 correctly)
        sorted_files = natsorted(all_files)

        # Assign chapter order and return tuples
        return [(file, idx) for idx, file in enumerate(sorted_files, start=1)]

    def _extract_chapter_number(self, file_path: str) -> int:
        """Extract chapter number from file path for natural ordering
        Handles patterns like: chapter1, chapter10, part1/chapter2, etc.
        """
        # Try to extract chapter number from filename or path
        match = re.search(r'chapter[_-]?(\d+)', file_path.lower())
        if match:
            return int(match.group(1))

        # Try appendix (treat as high numbers)
        match = re.search(r'appendix[_-]?([a-z])', file_path.lower())
        if match:
            # Convert appendix letter to number (a=100, b=101, etc.)
            return 100 + (ord(match.group(1)) - ord('a'))

        # Try intro (treat as chapter 0)
        if 'intro' in file_path.lower():
            return 0

        # Default: use position in sorted list
        return 999

    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into chunks and return with metadata"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_text(text)

        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "source": source,
                "chunk_id": i,
                "page_title": self._extract_title_from_path(source),
                "url": self._convert_path_to_url(source)
            })

        return documents

    def _extract_title_from_path(self, file_path: str) -> str:
        """Extract page title from file path"""
        # Extract filename without extension
        filename = Path(file_path).stem
        # Convert to more readable format (replace underscores/hyphens with spaces)
        title = filename.replace('_', ' ').replace('-', ' ').title()
        return title

    def _convert_path_to_url(self, file_path: str) -> str:
        """Convert file path to Docusaurus URL"""
        # Convert file path to relative path from docs
        docs_path = Path(file_path).parent
        relative_path = os.path.relpath(file_path, "docusaurus/docs")

        # Convert to URL format
        url_path = relative_path.replace('\\', '/').replace('.md', '').replace('.mdx', '')
        if url_path == 'intro':
            return '/'
        return f'/{url_path}'

    def process_book(self, docs_dir: str):
        """
        NEW PIPELINE:
        1. Read from docusaurus/docs
        2. Inject into Neon Postgres
        3. Fetch from Neon
        4. Embed with Gemini
        5. Save to JSON
        6. Upload to Qdrant
        """
        logger.info("=" * 80)
        logger.info("STARTING NEW RAG INGESTION PIPELINE")
        logger.info("=" * 80)

        # Step 1: Read from docs and inject into Neon
        logger.info("\n[STEP 1/6] Reading docs and injecting into Neon Postgres...")
        self._inject_docs_to_neon(docs_dir)

        # Step 2: Fetch from Neon
        logger.info("\n[STEP 2/6] Fetching content from Neon Postgres...")
        neon_content = self.db.get_all_content()
        logger.info(f"Fetched {len(neon_content)} chunks from Neon")

        # Step 3: Embed with Gemini
        logger.info("\n[STEP 3/6] Generating embeddings with Gemini...")
        embedded_data = self._embed_content(neon_content)

        # Step 4: Save to JSON
        logger.info("\n[STEP 4/6] Saving embeddings to JSON file...")
        json_file_path = self._save_to_json(embedded_data)

        # Step 5: Upload to Qdrant
        logger.info("\n[STEP 5/6] Uploading embeddings to Qdrant...")
        self._upload_to_qdrant_from_json(json_file_path)

        # Step 6: Summary
        logger.info("\n[STEP 6/6] Pipeline Complete!")
        logger.info("=" * 80)
        logger.info("PIPELINE SUMMARY:")
        logger.info(f"  - Chapters in Neon: {self.db.get_chapter_count()}")
        logger.info(f"  - Chunks in Neon: {self.db.get_content_chunk_count()}")
        logger.info(f"  - Embeddings generated: {len(embedded_data)}")
        logger.info(f"  - JSON file: {json_file_path}")
        logger.info(f"  - Qdrant collection: {self.collection_name}")
        logger.info("=" * 80)

    def _inject_docs_to_neon(self, docs_dir: str):
        """Step 1: Read docs from docusaurus/docs and inject into Neon with proper ordering"""
        markdown_files_with_order = self.get_all_markdown_files(docs_dir)
        logger.info(f"Found {len(markdown_files_with_order)} markdown files")

        total_chunks = 0

        for file_path, chapter_order in markdown_files_with_order:
            try:
                logger.info(f"Processing [{chapter_order}] {file_path}")

                # Extract text from markdown
                text = self.extract_text_from_markdown(file_path)

                # Get metadata
                page_title = self._extract_title_from_path(file_path)
                url = self._convert_path_to_url(file_path)

                # Insert chapter into Neon with chapter order
                chapter_id = self.db.insert_chapter(
                    file_path=file_path,
                    page_title=page_title,
                    url=url,
                    chapter_order=chapter_order
                )

                # Create chunks
                chunks = self.chunk_text(text, file_path)

                # Insert chunks into Neon
                for i, chunk_data in enumerate(chunks):
                    self.db.insert_content_chunk(
                        chapter_id=chapter_id,
                        chunk_id=i,
                        content=chunk_data["content"]
                    )

                total_chunks += len(chunks)
                logger.info(f"  ✓ Inserted {len(chunks)} chunks into Neon (order: {chapter_order})")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue

        logger.info(f"Total chunks injected into Neon: {total_chunks}")

    def _embed_content(self, neon_content: List[Dict]) -> List[Dict]:
        """Step 3: Generate embeddings for content from Neon"""
        embedded_data = []

        for i, item in enumerate(neon_content):
            try:
                if (i + 1) % 10 == 0:
                    logger.info(f"  Embedding progress: {i + 1}/{len(neon_content)}")

                # Generate embedding
                embedding = self._get_embedding(item['content'])

                # Create embedded data entry
                embedded_data.append({
                    'id': item['id'],
                    'content': item['content'],
                    'chunk_id': item['chunk_id'],
                    'file_path': item['file_path'],
                    'page_title': item['page_title'],
                    'url': item['url'],
                    'embedding': embedding
                })

            except Exception as e:
                logger.error(f"Error embedding content {item['id']}: {str(e)}")
                continue

        logger.info(f"Generated {len(embedded_data)} embeddings")
        return embedded_data

    def _save_to_json(self, embedded_data: List[Dict]) -> str:
        """Step 4: Save embedded data to JSON file"""
        json_file_path = "embeddings_data.json"

        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(embedded_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(embedded_data)} embeddings to {json_file_path}")
            return json_file_path

        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")
            raise

    def _upload_to_qdrant_from_json(self, json_file_path: str):
        """Step 5: Upload embeddings from JSON to Qdrant"""
        try:
            # Load JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                embedded_data = json.load(f)

            logger.info(f"Loaded {len(embedded_data)} embeddings from {json_file_path}")

            # Upload to Qdrant in batches
            points = []
            batch_size = 25

            for i, item in enumerate(embedded_data):
                try:
                    # Create Qdrant point
                    point = models.PointStruct(
                        id=item['id'],
                        vector=item['embedding'],
                        payload={
                            "content": item['content'],
                            "source": item['file_path'],
                            "page_title": item['page_title'],
                            "url": item['url'],
                            "chunk_id": item['chunk_id']
                        }
                    )
                    points.append(point)

                    # Batch upload
                    if len(points) >= batch_size:
                        self._upsert_with_retry(points)
                        logger.info(f"  Uploaded {len(points)} points to Qdrant")
                        points = []

                except Exception as e:
                    logger.error(f"Error creating point for item {item['id']}: {str(e)}")
                    continue

            # Upload remaining points
            if points:
                self._upsert_with_retry(points)
                logger.info(f"  Uploaded final {len(points)} points to Qdrant")

            logger.info(f"Successfully uploaded all embeddings to Qdrant collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error uploading to Qdrant from JSON: {str(e)}")
            raise

    def _upsert_with_retry(self, points, max_retries=3):
        """Upsert points to Qdrant with retry logic for network errors"""
        import time
        for attempt in range(max_retries):
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True  # Wait for operation to complete
                )
                return  # Success, exit the retry loop
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Failed to upsert after {max_retries} attempts")
                    raise e
                # Wait before retry with exponential backoff
                time.sleep(2 ** attempt)  # Wait 1s, 2s, 4s, etc.

def main():
    """Main function to run the new RAG ingestion pipeline"""
    import os

    # Verify environment variables are loaded
    neon_url = os.getenv("NEON_DATABASE_URL")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    print("\n" + "=" * 80)
    print("NEW RAG INGESTION PIPELINE - Environment Check")
    print("=" * 80)
    print(f"NEON_DATABASE_URL loaded: {'✓' if neon_url else '✗'}")
    print(f"QDRANT_URL loaded: {'✓' if qdrant_url else '✗'}")
    print(f"QDRANT_API_KEY loaded: {'✓' if qdrant_api_key else '✗'}")
    print(f"GEMINI_API_KEY loaded: {'✓' if gemini_api_key else '✗'}")
    print("=" * 80 + "\n")

    if not all([neon_url, qdrant_url, qdrant_api_key, gemini_api_key]):
        print("ERROR: Missing required environment variables!")
        print("Please check your .env file has:")
        print("  - NEON_DATABASE_URL")
        print("  - QDRANT_URL")
        print("  - QDRANT_API_KEY")
        print("  - GEMINI_API_KEY")
        return

    # Initialize the ingester
    try:
        ingester = BookIngestor()
    except Exception as e:
        print(f"ERROR: Failed to initialize BookIngestor: {e}")
        if "403" in str(e) or "Forbidden" in str(e):
            print("This is likely due to an invalid QDRANT_API_KEY or incorrect QDRANT_URL")
            print("Please verify:")
            print("  - Your QDRANT_API_KEY has Read & Write permissions")
            print("  - Your QDRANT_URL is correct")
        return

    # Process the book content
    # Check if we're running from project root or backend directory
    if os.path.basename(os.getcwd()) == 'backend':
        # We're in the backend directory, so go up one level to find docs
        docs_dir = "../docusaurus/docs"
    else:
        # We're in the project root
        docs_dir = "docusaurus/docs"

    ingester.process_book(docs_dir)

    # Close database connection
    ingester.db.close()

if __name__ == "__main__":
    main()