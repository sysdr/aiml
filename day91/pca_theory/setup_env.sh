#!/bin/bash

echo "Setting up Day 91: PCA Theory environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "Setup complete! Activate the environment with:"
echo "  source venv/bin/activate"
echo ""
echo "Then run the lesson with:"
echo "  python lesson_code.py"
echo ""
echo "Or run tests with:"
echo "  pytest test_lesson.py -v"
