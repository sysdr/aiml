#!/bin/bash

echo "🛑 Stopping AI Dashboard..."

# Find and kill the dashboard process
DASHBOARD_PID=$(ps aux | grep "python dashboard.py" | grep -v grep | awk '{print $2}')

if [ ! -z "$DASHBOARD_PID" ]; then
    echo "📊 Found dashboard process (PID: $DASHBOARD_PID)"
    kill $DASHBOARD_PID
    echo "✅ Dashboard stopped successfully"
else
    echo "ℹ️  No dashboard process found"
fi

# Also kill any Flask processes on port 5000
FLASK_PID=$(lsof -ti:5000 2>/dev/null)

if [ ! -z "$FLASK_PID" ]; then
    echo "🔧 Found Flask process on port 5000 (PID: $FLASK_PID)"
    kill $FLASK_PID
    echo "✅ Flask process stopped"
fi

echo "🎯 All dashboard services stopped"
