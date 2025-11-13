#!/bin/bash

# Startup script for Day 31: NumPy Dashboard
# This script starts the dashboard and checks for duplicate services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_SCRIPT="${SCRIPT_DIR}/dashboard.py"
DEMO_SCRIPT="${SCRIPT_DIR}/demo.py"

echo "=============================================================="
echo "Day 31: NumPy Dashboard Startup Script"
echo "=============================================================="

# Check if we're in the right directory
if [ ! -f "$DASHBOARD_SCRIPT" ]; then
    echo "ERROR: dashboard.py not found at $DASHBOARD_SCRIPT"
    echo "Please run this script from the day31_numpy_intro directory"
    exit 1
fi

# Check if virtual environment exists, otherwise use system Python
if [ -d "${SCRIPT_DIR}/venv" ] && [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    # Activate virtual environment
    echo "Activating virtual environment..."
    source "${SCRIPT_DIR}/venv/bin/activate"
    PYTHON_CMD="python"
else
    echo "Using system Python (venv not available)"
    PYTHON_CMD="python3"
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check for duplicate dashboard processes
echo "Checking for existing dashboard processes..."
EXISTING_PIDS=$(ps aux | grep "[d]ashboard.py" | awk '{print $2}')
if [ ! -z "$EXISTING_PIDS" ]; then
    echo "WARNING: Found existing dashboard processes: $EXISTING_PIDS"
    echo "Killing existing processes..."
    for pid in $EXISTING_PIDS; do
        kill $pid 2>/dev/null
    done
    sleep 2
    # Force kill if still running
    for pid in $EXISTING_PIDS; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null
        fi
    done
    echo "Cleaned up existing processes"
fi

# Check if port 5000 is in use
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "WARNING: Port 5000 is already in use"
    echo "Attempting to free port 5000..."
    PID_ON_PORT=$(lsof -ti:5000)
    if [ ! -z "$PID_ON_PORT" ]; then
        kill $PID_ON_PORT 2>/dev/null
        sleep 2
    fi
fi

# Start dashboard in background
echo "Starting dashboard..."
cd "$SCRIPT_DIR"
nohup $PYTHON_CMD "$DASHBOARD_SCRIPT" > dashboard.log 2>&1 &
DASHBOARD_PID=$!

# Wait a moment for dashboard to start
sleep 3

# Check if dashboard started successfully
if kill -0 $DASHBOARD_PID 2>/dev/null; then
    echo "✓ Dashboard started successfully (PID: $DASHBOARD_PID)"
    echo "✓ Dashboard available at: http://localhost:5000"
    echo "✓ Dashboard log: ${SCRIPT_DIR}/dashboard.log"
    echo ""
    echo "To stop the dashboard, run:"
    echo "  kill $DASHBOARD_PID"
    echo ""
    echo "To run the demo and update metrics:"
    echo "  cd ${SCRIPT_DIR}"
    echo "  source venv/bin/activate"
    echo "  python demo.py"
else
    echo "ERROR: Dashboard failed to start"
    echo "Check dashboard.log for details"
    exit 1
fi

