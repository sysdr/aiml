#!/bin/bash

echo "Setting up Day 62: KNN Theory environment..."

# Check Python version
python3 --version

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Setup complete! Activate environment with: source venv/bin/activate"
echo "Run the lesson with: python lesson_code.py"
echo "Run tests with: pytest test_lesson.py -v"
