#!/bin/bash

echo "Setting up Day 44: Simple Linear Regression Theory environment..."

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "Setup complete! ✓"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run lesson: python lesson_code.py --train"
echo "3. Run tests: pytest test_lesson.py -v"
echo ""
