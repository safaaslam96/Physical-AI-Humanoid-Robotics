# Chatbot Testing Guide

## ✅ Fixes Applied:

### 1. Fixed jsx Warning
- Changed `<style jsx>` to `<style dangerouslySetInnerHTML={{__html: \`...\`}} />`
- No more React non-boolean attribute warnings

### 2. Fixed 404 Error
- **Corrected API endpoint**: Changed from `/chat` to `/api/chat`
- **Fixed request body**: Changed `query` field to `message` field
- **Added correct fields**: `message`, `selected_text`, `from_selected_text`

### 3. Fixed Gemini API Quota Exhaustion
- **Changed model**: Updated from `gemini-2.5-pro-flash` to `models/gemini-2.5-flash`
- **Root cause**:
  - Experimental/non-existent model name in `.env` file
  - Missing "models/" prefix required by Gemini API
- **Files updated**:
  - `backend/.env`: Changed `GEMINI_MODEL=gemini-2.5-pro-flash` to `GEMINI_MODEL=models/gemini-2.5-flash`
  - `backend/src/core/config.py` line 20: Set default to `models/gemini-2.5-flash`
- **Result**: Chatbot now generates proper AI responses from the book content ✅

### 4. Fixed API Route and Request Body Mismatch
- **API endpoint**: Changed from prefix="" to prefix="/api" in `backend/main.py`
- **Request model**: Updated `FrontendChatRequest` in `backend/src/api/rag_chat_routes.py`:
  - Changed `query` → `message`
  - Changed `history` → `from_selected_text`
- **Result**: Frontend and backend now communicate correctly at `/api/chat` endpoint ✅

## 🧪 How to Test:

### Step 1: Verify Backend is Running
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","version":"1.0.0"}
```

### Step 2: Test Chat Endpoint Directly
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"What is ROS 2?\"}"

# Should return a JSON response with:
# - response: "..." (AI answer)
# - sources: [...] (book chapters)
# - timestamp: "..."
```

### Step 3: Test in Browser
1. Open http://localhost:3000
2. Click the 🤖 button in bottom-right corner
3. Chat popup should open
4. Type: "What is ROS 2?"
5. Press Enter or click ➤
6. Should see AI response from the book!

### Step 4: Test Selected Text Feature
1. Highlight any text on the page
2. Click "Ask AI" button in the highlighter toolbar
3. Chat should open with pre-filled question
4. Submit and get response

## 📋 Expected Behavior:

**Working Chatbot:**
- ✅ Chat button appears (🤖)
- ✅ Click opens glassmorphic chat popup
- ✅ Type message and press Enter
- ✅ Loading indicator appears (typing dots)
- ✅ AI response appears with book content
- ✅ No console errors
- ✅ No 404 errors
- ✅ No jsx warnings

**Error Indicators:**
- ❌ 404 error → Backend not running or wrong endpoint
- ❌ CORS error → Backend CORS not configured
- ❌ Timeout → Backend taking too long (check Gemini/Qdrant)
- ❌ "Field required" → Wrong request body format

## 🔧 Troubleshooting:

### If 404 Error Persists:
```bash
# 1. Check which server is running
netstat -ano | findstr :8000

# 2. Kill old servers
taskkill /PID <PID> /F

# 3. Start correct backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### If CORS Error:
- Backend should already have CORS enabled for all origins
- Check browser console for specific CORS error

### If Slow Response:
- Gemini API might be slow (first time ~1-2 seconds)
- Qdrant search should be fast (<100ms)
- Check backend logs for errors

## 🎯 API Contract:

**Request:**
```json
{
  "message": "Your question here",
  "selected_text": "highlighted text (optional)",
  "from_selected_text": false
}
```

**Response:**
```json
{
  "response": "AI answer from the book",
  "sources": [
    {
      "chapter": "part2_chapter3",
      "content_preview": "...",
      "score": 0.85
    }
  ],
  "timestamp": "2024-12-28T05:54:51.193526"
}
```

## ✅ Success Criteria:

1. No console errors
2. Chat opens smoothly
3. Messages send successfully
4. AI responds with book content
5. Sources are displayed (if implemented in UI)
6. Chat history persists during session
7. Mobile responsive (optional)

---

Last Updated: 2024-12-28
Backend: http://localhost:8000
Frontend: http://localhost:3000
