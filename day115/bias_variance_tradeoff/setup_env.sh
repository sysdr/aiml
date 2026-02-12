#!/bin/bash

echo "Setting up Day 115: Bias-Variance Tradeoff environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Setup complete! Activate the environment with: source venv/bin/activate"
echo "Then run: python lesson_code.py"
echo "Or run tests: pytest test_lesson.py -v"
