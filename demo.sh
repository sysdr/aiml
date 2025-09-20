#!/bin/bash

echo "🎬 Running AI Chat Assistant Demo..."

# Check if services are running
backend_status=$(curl -s http://localhost:8000/api/v1/health 2>/dev/null | grep -o '"status":"healthy"' || echo "")
frontend_status=$(curl -s http://localhost:3000 2>/dev/null && echo "running" || echo "")

if [ -z "$backend_status" ] || [ -z "$frontend_status" ]; then
    echo "⚠️  Services not running. Starting them first..."
    ./start.sh &
    sleep 10
fi

echo "🧪 Testing API endpoints..."

# Test health endpoint
echo "1. Testing health endpoint..."
curl -X GET http://localhost:8000/api/v1/health

echo -e "\n\n2. Testing chat endpoint..."
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! Can you explain what Python is?",
    "conversation_history": []
  }'

echo -e "\n\n✅ Demo complete!"
echo "🌐 Open http://localhost:3000 to interact with the UI"
echo "📡 API Documentation: http://localhost:8000/docs"
