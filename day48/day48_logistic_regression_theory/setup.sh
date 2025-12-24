#!/bin/bash

echo "Setting up Day 48: Logistic Regression Theory environment..."

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

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the environment, run: source venv/bin/activate"
echo "To run the lesson code: python lesson_code.py"
echo "To run tests: pytest test_lesson.py -v"
