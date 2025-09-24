#!/bin/bash

echo "Setting up Day 8: Introduction to Linear Algebra environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python $python_version detected (meets requirement: $required_version+)"
else
    echo "✗ Python $required_version+ required. Current version: $python_version"
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv ai_course_env

# Activate virtual environment
echo "Activating virtual environment..."
source ai_course_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Setup complete!"
echo ""
echo "To activate the environment later, run:"
echo "source ai_course_env/bin/activate"
echo ""
echo "Then run the lesson with:"
echo "python lesson_code.py"
