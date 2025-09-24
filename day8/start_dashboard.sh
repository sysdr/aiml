#!/bin/bash

# Day 8: Linear Algebra Dashboard Startup Script
# Starts the web dashboard and all necessary services

echo "🚀 Starting Day 8: Linear Algebra Dashboard..."

# Check if virtual environment exists
if [ ! -d "ai_course_env" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ai_course_env/bin/activate

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    pip install flask==2.3.3
fi

# Check if dashboard.py exists
if [ ! -f "dashboard.py" ]; then
    echo "❌ dashboard.py not found. Please ensure all files are generated."
    exit 1
fi

# Check if templates directory exists
if [ ! -d "templates" ]; then
    echo "❌ templates directory not found. Please ensure all files are generated."
    exit 1
fi

# Check for running processes on port 5000
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 5000 is already in use. Stopping existing process..."
    pkill -f "python.*dashboard.py" || true
    sleep 2
fi

echo "🌐 Starting Linear Algebra Dashboard..."
echo "📊 Dashboard will be available at: http://localhost:5000"
echo "🔢 Press Ctrl+C to stop the dashboard"
echo ""

# Start the dashboard
python dashboard.py
