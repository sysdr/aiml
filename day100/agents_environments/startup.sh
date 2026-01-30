#!/bin/bash

# Startup script for Day 100 RL Lesson
# This script starts the lesson and dashboard

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Day 100: Agents, Environments, and Rewards..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    if [ -f "setup_venv.sh" ]; then
        bash setup_venv.sh
    else
        echo "Error: setup_venv.sh not found"
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Check if dashboard script exists and start it
if [ -f "dashboard.py" ]; then
    echo "Starting dashboard..."
    python dashboard.py &
    DASHBOARD_PID=$!
    echo "Dashboard started with PID: $DASHBOARD_PID"
    sleep 2
fi

# Run the lesson
echo "Running lesson..."
python lesson_code.py

echo "Startup complete!"
