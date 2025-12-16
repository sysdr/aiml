#!/bin/bash

echo "Setting up Day 41: Overfitting and Underfitting environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo "✅ Setup complete! Virtual environment ready."
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the lesson:"
echo "  python lesson_code.py"
echo ""
echo "To run tests:"
echo "  pytest test_lesson.py -v"
