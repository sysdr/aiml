#!/bin/bash

echo "Setting up Python environment for Day 13: Derivatives..."

# Check if Python 3.11+ is available
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 11 ]); then
    echo "❌ Python 3.11+ required. Please install Python 3.11 or later."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo "🎉 Setup complete! Run 'source venv/bin/activate' to activate the environment."
echo "Then run 'python lesson_code.py' to start the lesson."
