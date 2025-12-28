@echo off
echo Testing RAG Chat Endpoint...
echo.

curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"What is ROS 2?\"}"

echo.
echo.
echo Testing Health Endpoint...
curl http://localhost:8000/health

echo.
