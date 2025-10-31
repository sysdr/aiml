#!/bin/bash

echo "Setting up Day 26: Descriptive Statistics Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "To activate environment: source venv/bin/activate"
echo "To run lesson: python lesson_code.py"
echo "To run tests: python test_lesson.py"
echo "To start dashboard: python app.py"
