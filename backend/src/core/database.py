"""
Neon Postgres database connection and schema management for book content
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class NeonDatabase:
    def __init__(self):
        self.database_url = os.getenv("NEON_DATABASE_URL")
        if not self.database_url:
            raise ValueError("NEON_DATABASE_URL environment variable is required")

        self.connection = None
        self.connect()
        self.create_schema()

    def connect(self):
        """Establish connection to Neon Postgres"""
        try:
            self.connection = psycopg2.connect(self.database_url)
            logger.info("Successfully connected to Neon Postgres")
        except Exception as e:
            logger.error(f"Failed to connect to Neon Postgres: {e}")
            raise

    def create_schema(self):
        """Create database schema for book content if it doesn't exist"""
        try:
            with self.connection.cursor() as cursor:
                # Create book_chapters table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS book_chapters (
                        id SERIAL PRIMARY KEY,
                        file_path TEXT NOT NULL UNIQUE,
                        page_title TEXT NOT NULL,
                        url TEXT NOT NULL,
                        chapter_order INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Add chapter_order column if it doesn't exist (for existing databases)
                cursor.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='book_chapters' AND column_name='chapter_order'
                        ) THEN
                            ALTER TABLE book_chapters ADD COLUMN chapter_order INTEGER DEFAULT 0;
                        END IF;
                    END $$;
                """)

                # Create book_content table (chunks)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS book_content (
                        id SERIAL PRIMARY KEY,
                        chapter_id INTEGER REFERENCES book_chapters(id) ON DELETE CASCADE,
                        chunk_id INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(chapter_id, chunk_id)
                    )
                """)

                # Create index for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_book_content_chapter_id
                    ON book_content(chapter_id)
                """)

                self.connection.commit()
                logger.info("Database schema created successfully")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to create schema: {e}")
            raise

    def insert_chapter(self, file_path: str, page_title: str, url: str, chapter_order: int = 0) -> int:
        """Insert a chapter and return its ID"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO book_chapters (file_path, page_title, url, chapter_order)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (file_path)
                    DO UPDATE SET
                        page_title = EXCLUDED.page_title,
                        url = EXCLUDED.url,
                        chapter_order = EXCLUDED.chapter_order,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (file_path, page_title, url, chapter_order))

                chapter_id = cursor.fetchone()[0]
                self.connection.commit()
                return chapter_id
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to insert chapter: {e}")
            raise

    def insert_content_chunk(self, chapter_id: int, chunk_id: int, content: str):
        """Insert a content chunk for a chapter"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO book_content (chapter_id, chunk_id, content)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (chapter_id, chunk_id)
                    DO UPDATE SET content = EXCLUDED.content
                """, (chapter_id, chunk_id, content))

                self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to insert content chunk: {e}")
            raise

    def get_all_content(self) -> List[Dict]:
        """Fetch all book content with metadata, ordered by chapter sequence"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT
                        bc.id,
                        bc.content,
                        bc.chunk_id,
                        ch.file_path,
                        ch.page_title,
                        ch.url,
                        ch.chapter_order
                    FROM book_content bc
                    JOIN book_chapters ch ON bc.chapter_id = ch.id
                    ORDER BY ch.chapter_order, ch.id, bc.chunk_id
                """)

                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to fetch content: {e}")
            raise

    def clear_all_content(self):
        """Clear all book content (for re-ingestion)"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM book_content")
                cursor.execute("DELETE FROM book_chapters")
                self.connection.commit()
                logger.info("All book content cleared")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to clear content: {e}")
            raise

    def get_chapter_count(self) -> int:
        """Get total number of chapters"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM book_chapters")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get chapter count: {e}")
            return 0

    def get_content_chunk_count(self) -> int:
        """Get total number of content chunks"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM book_content")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get chunk count: {e}")
            return 0

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")