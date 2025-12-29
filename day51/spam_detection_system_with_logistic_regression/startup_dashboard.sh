#!/bin/bash

# Dashboard Startup Script for Day 51
# This script starts the real-time metrics dashboard

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Day 51 Real-Time Spam Detection Dashboard..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Running setup.sh first..."
    bash setup.sh
    if [ $? -ne 0 ]; then
        echo "❌ Setup failed. Please fix errors and try again."
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    pip install flask --quiet
fi

# Check if dashboard.py exists
if [ ! -f "dashboard.py" ]; then
    echo "❌ dashboard.py not found!"
    exit 1
fi

# Check for duplicate dashboard processes
DASHBOARD_PID=$(ps aux | grep "[d]ashboard.py" | grep -v "grep" | awk '{print $2}')
if [ ! -z "$DASHBOARD_PID" ]; then
    echo "⚠️  Warning: Dashboard process already running (PID: $DASHBOARD_PID)"
    read -p "Do you want to kill the existing process and start a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill $DASHBOARD_PID
        sleep 2
        echo "✅ Old process terminated"
    else
        echo "ℹ️  Keeping existing process. Exiting."
        exit 0
    fi
fi

# Check if port 5000 is in use
if command -v lsof &> /dev/null; then
    PORT_IN_USE=$(lsof -ti:5000 2>/dev/null)
    if [ ! -z "$PORT_IN_USE" ]; then
        echo "⚠️  Port 5000 is already in use (PID: $PORT_IN_USE)"
        read -p "Do you want to kill the process using port 5000? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill $PORT_IN_USE
            sleep 2
            echo "✅ Port 5000 freed"
        else
            echo "❌ Cannot start dashboard. Port 5000 is in use."
            exit 1
        fi
    fi
fi

# Make dashboard.py executable
chmod +x dashboard.py

# Start the dashboard
echo ""
echo "✅ Starting dashboard server..."
echo "📊 Dashboard will be available at: http://localhost:5000"
echo "🔄 Metrics update in real-time every 3 seconds"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

python dashboard.py

