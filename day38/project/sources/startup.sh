#!/bin/bash

# Startup script for Day 38: ML Workflow Dashboard

echo "🚀 Starting Day 38: ML Workflow Dashboard..."
echo "==========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to sources directory
cd "$SCRIPT_DIR" || exit 1

# Check if virtual environment exists in parent directory
if [ ! -d "$PARENT_DIR/ml_workflow_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: cd $PARENT_DIR && ./setup_env.sh first"
    exit 1
fi

# Activate virtual environment
source "$PARENT_DIR/ml_workflow_env/bin/activate"

# Check if dashboard is already running
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Dashboard already running on port 5000"
    echo "   Stopping existing instance..."
    pkill -9 -f "python.*dashboard.py" || true
    sleep 2
    # Double check
    if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "❌ Failed to stop existing dashboard"
        exit 1
    fi
fi

# Also check for any dashboard processes
DASHBOARD_COUNT=$(ps aux | grep -E "python.*dashboard.py" | grep -v grep | wc -l)
if [ "$DASHBOARD_COUNT" -gt 0 ]; then
    echo "⚠️  Found $DASHBOARD_COUNT existing dashboard process(es), stopping..."
    pkill -9 -f "python.*dashboard.py" || true
    sleep 2
fi

# Start dashboard
echo "📊 Starting dashboard on http://localhost:5000"
nohup python dashboard.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!

echo "✅ Dashboard started (PID: $DASHBOARD_PID)"
echo ""
echo "📱 Access dashboard at: http://localhost:5000"
echo "📡 API endpoint: http://localhost:5000/api/metrics"
echo ""
echo "To stop the dashboard, run:"
echo "   kill $DASHBOARD_PID"
echo "   or"
echo "   pkill -f 'python.*dashboard.py'"
echo ""

# Wait a moment for startup
sleep 3

# Check if dashboard started successfully
if ps -p $DASHBOARD_PID > /dev/null; then
    echo "✅ Dashboard is running successfully!"
else
    echo "❌ Dashboard failed to start"
    echo "   Check dashboard.log for errors:"
    tail -20 dashboard.log
    exit 1
fi

