"""
Database connection and setup for the Physical AI and Humanoid Robotics book platform
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings


# For Neon PostgreSQL, we'll use asyncpg for async operations
# For now, using standard PostgreSQL connection
SQLALCHEMY_DATABASE_URL = settings.neon_database_url or settings.database_url or "postgresql://user:password@localhost/dbname"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()