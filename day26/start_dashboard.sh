#!/bin/bash

# Day 26 Dashboard Startup Script
# This script checks for duplicate services and starts the dashboard server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "🔍 Checking for duplicate services..."

# Check if Flask app is already running
if pgrep -f "python.*app.py" > /dev/null; then
    echo "⚠️  Warning: Dashboard server appears to be running already"
    echo "   PIDs: $(pgrep -f 'python.*app.py' | tr '\n' ' ')"
    read -p "   Kill existing processes and start new? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "python.*app.py"
        sleep 2
        echo "✅ Killed existing processes"
    else
        echo "❌ Aborted. Dashboard may already be running on http://localhost:5000"
        exit 1
    fi
fi

# Check if port 5000 is in use
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Warning: Port 5000 is already in use"
    echo "   Another service may be using this port"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./env_setup.sh first"
    exit 1
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Activated virtual environment"
else
    echo "❌ Virtual environment activation script not found"
    exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Run setup.sh first"
    exit 1
fi

# Check if dashboard.html exists
if [ ! -f "dashboard.html" ]; then
    echo "❌ dashboard.html not found. Run setup.sh first"
    exit 1
fi

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask not found. Installing..."
    pip install flask flask-cors
fi

echo ""
echo "🚀 Starting dashboard server..."
echo "📊 Dashboard will be available at: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop"
echo ""

# Start the server
python app.py
