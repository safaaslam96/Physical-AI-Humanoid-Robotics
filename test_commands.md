# Test Commands for Physical AI & Humanoid Robotics Book Project

## Agent Commands

```bash
# Run BookMasterAgent to fix sidebar
claude agents:run BookMasterAgent --task "Fix sidebar"

# Run BookMasterAgent to translate entire book to Urdu
claude agents:run BookMasterAgent --task "Translate entire book to Urdu"

# Run UIUX_Agent to create modern homepage
claude agents:run UIUX_Agent --task "Create modern homepage with hero section, maroon dark + light complementary mode, footer links, and highlighter pen"

# Run BookOps_Agent to clean chapters
claude agents:run BookOps_Agent --task "Clean all chapters, enable personalization and highlighter feature"

# Run NextJS_Agent to generate components
claude agents:run NextJS_Agent --task "Generate navbar + footer + modules + chat UI + highlighter pen with theme colors"

# Run BookMasterAgent to simulate RAG Chatbot
claude agents:run BookMasterAgent --task "Run RAG Chatbot simulation with user-selected text and highlights"
```

## Development Commands

```bash
# Install dependencies for Docusaurus
cd docusaurus
npm install

# Start Docusaurus development server
npm start

# Build Docusaurus site
npm run build

# Serve built site
npm run serve

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Start backend server
cd backend
uvicorn rag_backend:app --reload --port 8000
```

## Build and Deployment

```bash
# Build the entire project
cd docusaurus
npm run build

# The build will be available in the /build directory
# For production deployment, serve the /build directory
```

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```env
QDRANT_URL=your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
NEON_DB_URL=your-neon-db-connection-string
OPENAI_API_KEY=your-openai-api-key
```

## Project Structure

```
Physical-AI-Humanoid-Robotics/
├── ai/
│   ├── agents/
│   │   ├── BookMasterAgent.json
│   │   ├── UIUX_Agent.json
│   │   ├── NextJS_Agent.json
│   │   └── BookOps_Agent.json
│   └── skills/
│       ├── translateTextSkill.json
│       ├── fixFrontMatterSkill.json
│       ├── imageInjectionSkill.json
│       ├── generateTailwindComponentSkill.json
│       ├── summarizeChapterSkill.json
│       ├── createChatbotMemorySkill.json
│       ├── personalizeChapterSkill.json
│       ├── signupSigninSkill.json
│       └── highlightTextSkill.json
├── backend/
│   ├── rag_backend.py
│   ├── requirements.txt
│   └── package.json
├── docusaurus/
│   ├── docs/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chatbot.tsx
│   │   │   ├── Highlighter.tsx
│   │   │   ├── PersonalizeButton.tsx
│   │   │   └── TranslateButton.tsx
│   │   ├── css/
│   │   │   └── custom.css
│   │   └── pages/
│   │       ├── auth/
│   │       │   ├── signin.tsx
│   │       │   └── signup.tsx
│   │       └── index.tsx
│   ├── pages/
│   │   └── auth/
│   │       ├── signin.tsx
│   │       └── signup.tsx
│   ├── static/
│   │   └── img/
│   │       └── physical-ai-humanoid-robotics-cover.png
│   ├── sidebars.js
│   ├── docusaurus.config.ts
│   └── package.json
├── rag_data/
│   └── summaries.json
└── test_commands.md
```

## Features Verification

1. **Homepage Hero Section**: Check that the cover image is on the left and title/buttons are on the right
2. **Color Scheme**: Verify maroon dark mode (#800000) and light complementary (#FFD6D6)
3. **Auth System**: Verify sign-in/sign-up buttons work
4. **Language Toggle**: Check English/Urdu toggle functionality
5. **Chatbot**: Verify floating chatbot popup works
6. **Sidebar**: Confirm tutorials are removed, proper capitalization
7. **Footer**: Check Learn/Community/About sections
8. **Highlighter**: Verify text highlighting functionality
9. **Personalization**: Check personalization button works when logged in
10. **Translation**: Verify translation button works