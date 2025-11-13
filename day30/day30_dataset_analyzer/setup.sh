#!/bin/bash

echo "Setting up Day 30: ML Dataset Analyzer environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python version: $python_version"

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

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run: source venv/bin/activate"
echo "To run the analyzer, execute: python lesson_code.py"
echo "To run tests, execute: pytest test_lesson.py -v"
