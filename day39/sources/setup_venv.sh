#!/bin/bash

echo "Setting up Day 39: Supervised vs. Unsupervised Learning"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

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
echo "Setup complete! To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "Then run the lesson:"
echo "python lesson_code.py"
