<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.0.0 (initial creation)
- Added sections: All principles and sections for Physical AI and Humanoid Robotics project
- Templates requiring updates: ✅ Updated all placeholders in constitution
- Follow-up TODOs: None
-->

# Physical AI and Humanoid Robotics Constitution

## Core Principles

### Spec-Driven Content Creation
All book content must be generated following Spec-Kit Plus methodology with deterministic generation rules. Every chapter, section, and subsection must have clear specifications before implementation. Content quality is measured by technical accuracy, clarity, and adherence to the project's educational objectives in Physical AI and Humanoid Robotics.

### AI-Agent Assisted Development
The book creation process must leverage Claude Code CLI and AI agents for content generation, technical review, and quality assurance. All content generation must use spec-first approaches with reusable prompts and deterministic generation rules. Human oversight is required for technical accuracy validation.

### Deterministic Generation (NON-NEGOTIABLE)
All content generation must follow spec-driven, deterministic rules that ensure reproducible results. Content must be generated from specifications rather than ad-hoc creation. All generation processes must be version-controlled and reproducible for consistency and quality assurance.

### Technical Accuracy and Domain Validation
All content related to Physical AI, Humanoid Robotics, hardware-software co-design, and embodied intelligence must undergo rigorous technical validation. Content must be reviewed by domain experts or validated through authoritative sources. No technical misinformation or oversimplification that could mislead readers is permitted.

### Modular Architecture and Reusability
The book system must follow a modular architecture with reusable components, prompts, and generation rules. Content modules must be independently testable and reusable across different sections or chapters. Components should follow DRY principles while maintaining educational coherence.

### RAG Safety and Grounding Rules
The integrated Retrieval-Augmented Generation chatbot must strictly adhere to content boundaries defined by the book's text. The chatbot must not hallucinate beyond retrieved content and must cite chapter/section sources for all responses. No generated responses should contain information not present in the source material.

## Content Quality Standards

All content must maintain professional academic standards appropriate for Physical AI and Humanoid Robotics education. Content must be accessible to readers with varying technical backgrounds while maintaining technical precision. Educational effectiveness is measured by clarity, accuracy, and progressive complexity appropriate to the reader's background.

Content must focus on:
- Physical AI principles and implementations
- Humanoid Robotics design and control
- Hardware-software co-design methodologies
- Embodied intelligence concepts and applications
- AI agents in robotics systems
- Future research directions in the field

## Personalization and Accessibility Constraints

Personalization features must respect user data privacy and provide deterministic, spec-driven content adaptation. User background information (software/hardware experience, learning goals) must be used to appropriately adjust content difficulty and examples. All personalization must maintain technical accuracy while adapting explanation styles and complexity levels.

Translation features must preserve technical accuracy, code blocks, and terminology consistency when converting content to Urdu. Translation must maintain the original meaning and technical precision while adapting to the target language's educational conventions.

## Technology Stack Requirements

The project must use:
- Docusaurus for static site generation and deployment
- GitHub Pages for hosting and distribution
- OpenAI Agents SDK / ChatKit frameworks with Google Gemini 2.5 Pro Flash models for RAG chatbot
- FastAPI backend for API services
- Neon Serverless PostgreSQL for user data and metadata
- Qdrant Cloud (Free Tier) for vector search
- Better-Auth.com for authentication and user profiles

All technology choices must support the book's educational mission while maintaining performance, reliability, and cost-effectiveness.

## Governance

This constitution governs all development activities for the Physical AI and Humanoid Robotics book project. All code changes, content additions, and feature implementations must comply with these principles. Amendments to this constitution require explicit approval and documentation of the changes and their impact on the project.

All pull requests and code reviews must verify compliance with these principles. Technical complexity must be justified by educational value or user experience improvements. Use this constitution as the primary guidance for all development decisions.

**Version**: 1.0.0 | **Ratified**: 2025-12-17 | **Last Amended**: 2025-12-17
