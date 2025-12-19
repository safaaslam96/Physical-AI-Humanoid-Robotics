---
id: 1
title: Create Constitution
stage: constitution
date: 2025-12-17
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: main
user: Administrator
command: /sp.constitution
labels: [constitution, spec-driven, ai-book, physical-ai, humanoid-robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
---

## Prompt

Create a complete `/sp.constitution` for an AI-powered, spec-driven book project titled:
"Physical AI and Humanoid Robotics"
This project must follow Spec-Kit Plus methodology and be fully compatible with Claude Code CLI.
Project Objectives
1. AI / Spec-Driven Book Creation (Base Requirement – 100 Points)
* The book must be:
   * Written using Spec-Kit Plus
   * Generated and maintained via Claude Code
   * Built with Docusaurus
   * Deployed to GitHub Pages
* The book should follow a modular, spec-first structure:
   * Clear specifications for chapters
   * Reusable prompts
   * Deterministic generation rules
* Content focus:
   * Physical AI
   * Humanoid Robotics
   * Hardware–software co-design
   * Embodied intelligence
   * AI agents in robotics
   * Future research directions
2. Integrated RAG Chatbot (Base Requirement)
Build and embed a Retrieval-Augmented Generation (RAG) chatbot inside the published Docusaurus book.
Technical Stack (Mandatory):
* OpenAI Agents SDK / ChatKit
* FastAPI backend
* Neon Serverless PostgreSQL (for user data + metadata)
* Qdrant Cloud (Free Tier) for vector search
* Deployed and usable inside the book UI
Chatbot Capabilities:
* Answer questions based on the entire book
* Answer questions based only on user-selected text
* Cite chapter/section sources
* Respect strict RAG boundaries (no hallucination beyond retrieved content)
3. Bonus Points – Advanced Intelligence (Up to +50)
* Design and use Claude Code Subagents
* Implement Agent Skills for:
   * Chapter drafting
   * Technical review
   * Robotics domain validation
   * Simplification for beginners
* Subagents must be reusable across chapters
4. Bonus Points – Authentication & Personalization (Up to +50)
* Implement Signup & Signin using https://www.better-auth.com/
* During signup, collect:
   * Software background (Beginner / Intermediate / Advanced)
   * Hardware/Robotics background
   * Learning goals
* Store user profile data securely
* Use this data to personalize:
   * Explanations
   * Examples
   * Difficulty level of content
5. Bonus Points – Chapter-Level Personalization (Up to +50)
* At the start of each chapter, provide a button:
   * "Personalize This Chapter"
* When clicked:
   * Regenerate or adapt the chapter content
   * Match the logged-in user's background and goals
* Personalization must be deterministic and spec-driven
6. Bonus Points – Urdu Translation (Up to +50)
* At the start of each chapter, provide a button:
   * "Translate to Urdu"
* On click:
   * Translate the chapter into clear, professional Urdu
   * Preserve:
      * Technical accuracy
      * Code blocks
      * Terminology consistency
* Translation should be generated dynamically for logged-in users
Constitution Requirements
The `/sp.constitution` must clearly define:
* System goals and non-goals
* Allowed and disallowed behaviors
* Deterministic generation rules
* Agent responsibilities
* Content quality standards
* RAG safety and grounding rules
* Personalization constraints
* Translation constraints
Output Format:
* Generate a complete, production-ready `/sp.constitution`
* Use Spec-Kit Plus style
* Clear sections, bullet points, and enforceable rules
* No explanations—only the constitution content

## Response snapshot

A complete constitution has been created for the "Physical AI and Humanoid Robotics" book project. The constitution includes:

1. Core Principles:
   - Spec-Driven Content Creation
   - AI-Agent Assisted Development
   - Deterministic Generation (NON-NEGOTIABLE)
   - Technical Accuracy and Domain Validation
   - Modular Architecture and Reusability
   - RAG Safety and Grounding Rules

2. Content Quality Standards:
   - Professional academic standards
   - Focus on Physical AI, Humanoid Robotics, hardware-software co-design, embodied intelligence, AI agents in robotics, and future research directions

3. Personalization and Accessibility Constraints:
   - User data privacy requirements
   - Urdu translation preservation requirements

4. Technology Stack Requirements:
   - Docusaurus, GitHub Pages, OpenAI Agents SDK, FastAPI, Neon PostgreSQL, Qdrant Cloud, Better-Auth.com

5. Governance section with compliance requirements

The constitution was written with specific, enforceable rules as required by the project methodology.

## Outcome

- ✅ Impact: Created complete project constitution defining all required principles and constraints
- 🧪 Tests: No tests required for constitution creation
- 📁 Files: Updated .specify/memory/constitution.md with complete constitution
- 🔁 Next prompts: Project can now proceed with spec creation based on this constitution
- 🧠 Reflection: Constitution successfully addresses all requirements from the user input

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS - All constitution requirements met
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A