#!/bin/bash

# Day 105: Content-Based Filtering - Server Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Content-Based Filtering API Server..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found in current directory."
    exit 1
fi

# Check if port 5001 is already in use
PORT=5001
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port $PORT is already in use."
    echo "   Checking for existing Flask processes..."
    ps aux | grep -E "flask|python.*app.py" | grep -v grep || true
    echo ""
    read -p "Do you want to kill existing processes and continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🛑 Stopping existing processes on port $PORT..."
        lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        echo "❌ Aborted. Please stop the existing process first."
        exit 1
    fi
fi

echo "✅ Starting Flask server..."
echo "📊 Dashboard will be available at: http://localhost:$PORT/"
echo "🔍 API endpoints available at: http://localhost:$PORT/api/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Flask app
export PORT=$PORT
python app.py

