@echo off
echo ========================================
echo Testing RAG Chatbot with Fixed Gemini Model
echo ========================================
echo.

echo [1/3] Testing Health Endpoint...
curl -s http://localhost:8000/health
echo.
echo.

echo [2/3] Testing RAG Chat Endpoint with "What is ROS 2?"...
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d "{\"message\": \"What is ROS 2?\"}"
echo.
echo.

echo [3/3] Testing RAG Chat Endpoint with "What is Gazebo?"...
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d "{\"message\": \"What is Gazebo?\"}"
echo.
echo.

echo ========================================
echo Testing Complete!
echo ========================================
echo.
echo If you see proper AI responses (not "service temporarily unavailable"),
echo then the chatbot is working correctly!
echo.
pause
