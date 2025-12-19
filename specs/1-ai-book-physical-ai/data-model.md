# Data Model: AI-Powered Book - Physical AI and Humanoid Robotics

## Overview
This document defines the data models for the AI-powered book project, including entities, relationships, and validation rules.

## Core Entities

### User
**Description**: System users who can access the book, personalize content, and use translation features

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the user
- `email` (String, Required, Unique): User's email address for authentication
- `name` (String, Required): User's full name
- `created_at` (DateTime, Required): Account creation timestamp
- `updated_at` (DateTime, Required): Last update timestamp
- `is_active` (Boolean, Default: true): Account status

**Validation Rules**:
- Email must be valid email format
- Email must be unique across all users
- Name must be 1-100 characters

### UserProfile
**Description**: Extended user profile information used for personalization

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the profile
- `user_id` (UUID, Foreign Key): Reference to the User
- `software_background` (String, Optional): User's software development experience level
- `hardware_background` (String, Optional): User's hardware/robotics experience level
- `learning_goals` (Text, Optional): User's specific learning objectives
- `difficulty_preference` (String, Default: "balanced"): Preferred content difficulty ("beginner", "intermediate", "advanced", "balanced")
- `created_at` (DateTime, Required): Profile creation timestamp
- `updated_at` (DateTime, Required): Last update timestamp

**Validation Rules**:
- `user_id` must reference an existing User
- `software_background` must be one of predefined values if specified
- `hardware_background` must be one of predefined values if specified
- `difficulty_preference` must be one of the allowed values

### BookChapter
**Description**: Individual chapters of the book content

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the chapter
- `title` (String, Required): Chapter title
- `slug` (String, Required, Unique): URL-friendly identifier
- `content` (Text, Required): Chapter content in Markdown format
- `order_index` (Integer, Required): Chapter position in the book
- `word_count` (Integer, Required): Number of words in the chapter
- `estimated_reading_time` (Integer, Required): Estimated reading time in minutes
- `created_at` (DateTime, Required): Chapter creation timestamp
- `updated_at` (DateTime, Required): Last update timestamp

**Validation Rules**:
- `slug` must be unique across all chapters
- `order_index` must be positive
- `word_count` must be non-negative
- `estimated_reading_time` must be positive

### ChatInteraction
**Description**: Record of user interactions with the RAG chatbot

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the interaction
- `user_id` (UUID, Foreign Key, Optional): Reference to the User (null for anonymous)
- `chapter_id` (UUID, Foreign Key, Optional): Reference to the Chapter being discussed
- `query` (Text, Required): User's question/query
- `response` (Text, Required): Chatbot's response
- `sources` (JSON, Required): List of sources cited in the response
- `created_at` (DateTime, Required): Interaction timestamp
- `is_validated` (Boolean, Default: false): Whether response was validated for accuracy

**Validation Rules**:
- `user_id` must reference an existing User if provided
- `chapter_id` must reference an existing Chapter if provided
- `query` and `response` must not be empty
- `sources` must be a valid JSON array

### PersonalizedChapter
**Description**: Personalized version of a chapter for a specific user

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the personalized chapter
- `user_id` (UUID, Foreign Key): Reference to the User
- `chapter_id` (UUID, Foreign Key): Reference to the original Chapter
- `personalized_content` (Text, Required): Personalized chapter content
- `personalization_settings` (JSON, Required): Settings used for personalization
- `created_at` (DateTime, Required): Personalization timestamp
- `updated_at` (DateTime, Required): Last update timestamp

**Validation Rules**:
- `user_id` must reference an existing User
- `chapter_id` must reference an existing Chapter
- `personalized_content` must not be empty
- `personalization_settings` must be valid JSON
- Combination of `user_id` and `chapter_id` must be unique

### TranslationCache
**Description**: Cached translations of chapters to Urdu

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the cached translation
- `chapter_id` (UUID, Foreign Key): Reference to the original Chapter
- `user_id` (UUID, Foreign Key, Optional): Reference to the User (for user-specific translations)
- `urdu_content` (Text, Required): Translated content in Urdu
- `translation_hash` (String, Required): Hash of original content to detect changes
- `created_at` (DateTime, Required): Translation creation timestamp
- `updated_at` (DateTime, Required): Last update timestamp

**Validation Rules**:
- `chapter_id` must reference an existing Chapter
- `user_id` must reference an existing User if provided
- `urdu_content` must not be empty
- `translation_hash` must be unique for the same `chapter_id`

### VectorEmbedding
**Description**: Vector embeddings for RAG functionality

**Fields**:
- `id` (UUID, Primary Key): Unique identifier for the embedding
- `chapter_id` (UUID, Foreign Key): Reference to the Chapter
- `content_chunk` (Text, Required): Text chunk that was embedded
- `embedding_vector` (JSON, Required): The vector embedding (array of floats)
- `chunk_index` (Integer, Required): Position of the chunk in the original content
- `created_at` (DateTime, Required): Embedding creation timestamp

**Validation Rules**:
- `chapter_id` must reference an existing Chapter
- `content_chunk` must not be empty
- `embedding_vector` must be a valid array of floats
- `chunk_index` must be non-negative

## Relationships

### User → UserProfile (One-to-One)
- One User has one UserProfile
- UserProfile is deleted when User is deleted (CASCADE)

### User → ChatInteraction (One-to-Many)
- One User can have many ChatInteractions
- ChatInteractions remain when User is deleted (SET NULL)

### User → PersonalizedChapter (One-to-Many)
- One User can have many PersonalizedChapters
- PersonalizedChapters are deleted when User is deleted (CASCADE)

### Chapter → ChatInteraction (One-to-Many)
- One Chapter can be associated with many ChatInteractions
- ChatInteractions remain when Chapter is deleted (SET NULL)

### Chapter → PersonalizedChapter (One-to-Many)
- One Chapter can have many PersonalizedChapters (one per user)
- PersonalizedChapters are deleted when Chapter is deleted (CASCADE)

### Chapter → TranslationCache (One-to-Many)
- One Chapter can have many TranslationCaches
- TranslationCaches are deleted when Chapter is deleted (CASCADE)

### Chapter → VectorEmbedding (One-to-Many)
- One Chapter can have many VectorEmbeddings
- VectorEmbeddings are deleted when Chapter is deleted (CASCADE)

## Indexes

### User
- Index on `email` (unique) for authentication performance
- Index on `created_at` for account management queries

### UserProfile
- Index on `user_id` (unique) for profile lookups
- Index on `updated_at` for profile updates

### BookChapter
- Index on `slug` (unique) for URL routing
- Index on `order_index` for chapter ordering
- Index on `updated_at` for content updates

### ChatInteraction
- Index on `user_id` for user-specific queries
- Index on `chapter_id` for chapter-specific queries
- Index on `created_at` for chronological queries

### PersonalizedChapter
- Index on `user_id` and `chapter_id` (unique) for personalized content lookup
- Index on `updated_at` for cache management

### TranslationCache
- Index on `chapter_id` and `user_id` for translation lookups
- Index on `translation_hash` for cache validation

### VectorEmbedding
- Index on `chapter_id` and `chunk_index` for content retrieval
- Index on `created_at` for embedding management

## Constraints

### Referential Integrity
- Foreign key constraints enforce relationship validity
- CASCADE and SET NULL behaviors are explicitly defined

### Data Integrity
- NOT NULL constraints on required fields
- CHECK constraints where appropriate for value validation
- UNIQUE constraints to prevent duplicates where required

### Business Logic
- Chapter order_index must be sequential within a book
- Personalization settings must match user profile when generating personalized content
- Translation cache must be invalidated when original content changes