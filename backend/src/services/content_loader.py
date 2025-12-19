"""
Content loader service for the Physical AI and Humanoid Robotics book platform
This service loads book content into the vector store for RAG functionality
"""
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from ..core.vector_store import VectorStore
from ..core.config import settings
import hashlib


class ContentLoader:
    def __init__(self):
        self.vector_store = VectorStore()
        self.content_dir = Path(settings.content_directory) if hasattr(settings, 'content_directory') else \
                          Path(__file__).parent.parent.parent / "docusaurus" / "docs"

    def _generate_doc_id(self, source: str, section: str = "") -> str:
        """Generate a unique document ID based on source and section"""
        content = f"{source}_{section}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _extract_content_from_md(self, file_path: Path) -> List[Dict]:
        """Extract content from markdown file, splitting into sections"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split content into sections based on headers
        lines = content.split('\n')
        sections = []
        current_section = {"title": "", "content": "", "line_start": 0, "line_end": 0}

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if this is a header (starts with #)
            if line.startswith('#'):
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section.copy())

                # Start new section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('# ').strip()
                current_section = {
                    "title": title,
                    "content": f"{title}\n",  # Include title in content
                    "line_start": i,
                    "line_end": i
                }
            else:
                # Add content to current section
                current_section["content"] += line + '\n'
                current_section["line_end"] = i

            i += 1

        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)

        # Filter out sections that are too short
        sections = [s for s in sections if len(s["content"].strip()) > 50]

        return sections

    async def load_book_content(self, force_reload: bool = False):
        """Load all book content from docs directory into vector store"""
        print(f"Loading book content from: {self.content_dir}")

        if not self.content_dir.exists():
            print(f"Content directory does not exist: {self.content_dir}")
            return False

        # Find all markdown files in the content directory
        md_files = list(self.content_dir.rglob("*.md"))
        print(f"Found {len(md_files)} markdown files to process")

        total_sections = 0
        for md_file in md_files:
            print(f"Processing: {md_file}")

            try:
                sections = self._extract_content_from_md(md_file)

                for i, section in enumerate(sections):
                    doc_id = self._generate_doc_id(str(md_file.relative_to(self.content_dir)), f"section_{i}")

                    # Prepare metadata
                    metadata = {
                        "source_file": str(md_file.relative_to(self.content_dir)),
                        "section_title": section["title"],
                        "line_start": section["line_start"],
                        "line_end": section["line_end"],
                        "file_path": str(md_file),
                        "part": self._extract_part_from_path(md_file),
                        "chapter": self._extract_chapter_from_path(md_file)
                    }

                    # Add to vector store
                    self.vector_store.add_document(
                        doc_id=doc_id,
                        content=section["content"],
                        metadata=metadata
                    )

                    total_sections += 1

                print(f"  Processed {len(sections)} sections from {md_file.name}")

            except Exception as e:
                print(f"  Error processing {md_file}: {e}")
                continue

        print(f"Successfully loaded {total_sections} content sections into vector store")
        return True

    def _extract_part_from_path(self, file_path: Path) -> str:
        """Extract part from file path (e.g., part1, part2, etc.)"""
        for parent in file_path.parts:
            if parent.startswith('part'):
                return parent
        return 'unknown'

    def _extract_chapter_from_path(self, file_path: Path) -> str:
        """Extract chapter from file path (e.g., chapter1, chapter2, etc.)"""
        file_name = file_path.stem
        if 'chapter' in file_name:
            return file_name
        return file_name

    async def load_specific_content(self, content_id: str, content: str, metadata: Dict = None):
        """Load specific content into vector store"""
        self.vector_store.add_document(content_id, content, metadata or {})

    async def search_content(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for content in the vector store"""
        return self.vector_store.search(query, limit)

    async def get_content_stats(self) -> Dict:
        """Get statistics about loaded content"""
        # This would require implementing a method in VectorStore to count documents
        # For now, we'll return a mock implementation
        return {
            "total_documents": 0,  # This would be implemented in the vector store
            "indexed_parts": [],
            "last_update": None
        }


async def load_book_content_main():
    """Main function to load book content - can be called from command line"""
    loader = ContentLoader()
    success = await loader.load_book_content(force_reload=True)

    if success:
        print("Book content loaded successfully!")
    else:
        print("Failed to load book content.")


if __name__ == "__main__":
    # Run the content loader if this script is executed directly
    asyncio.run(load_book_content_main())