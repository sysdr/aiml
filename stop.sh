#!/bin/bash

echo "🛑 Stopping AI Chat Assistant..."

# Kill all Node.js processes (React)
pkill -f "react-scripts start" 2>/dev/null

# Kill all Python/uvicorn processes
pkill -f "uvicorn" 2>/dev/null

# Kill any remaining processes on ports 3000 and 8000
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null

echo "✅ All services stopped!"
