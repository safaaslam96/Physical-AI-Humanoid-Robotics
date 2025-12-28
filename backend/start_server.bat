@echo off
echo Starting Physical AI RAG Chatbot Backend Server...
echo.
echo IMPORTANT: This will start the correct backend server with RAG chat routes
echo.

cd /d "%~dp0"

echo Checking environment variables...
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('GEMINI_API_KEY:', 'Set' if os.getenv('GEMINI_API_KEY') else 'NOT SET'); print('QDRANT_URL:', 'Set' if os.getenv('QDRANT_URL') else 'NOT SET'); print('QDRANT_API_KEY:', 'Set' if os.getenv('QDRANT_API_KEY') else 'NOT SET'); print('NEON_DATABASE_URL:', 'Set' if os.getenv('NEON_DATABASE_URL') else 'NOT SET')"

echo.
echo Starting FastAPI server on http://localhost:8000
echo Chat endpoint will be available at: http://localhost:8000/chat
echo Health check: http://localhost:8000/health
echo API docs: http://localhost:8000/docs
echo.

uvicorn main:app --reload --host 0.0.0.0 --port 8000
