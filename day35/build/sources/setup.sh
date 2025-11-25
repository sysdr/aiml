#!/bin/bash

echo "Setting up Day 35: Data Cleaning and Handling Missing Data environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

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

echo ""
echo "Setup complete! To get started:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run the main script: python lesson_code.py"
echo "3. Run tests: pytest test_lesson.py -v"
echo ""
