#!/bin/bash

echo "Setting up Day 32: NumPy Array Manipulation and Vectorization"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo "Activate environment: source venv/bin/activate"
echo "Run lesson: python lesson_code.py"
echo "Run tests: pytest test_lesson.py -v"
