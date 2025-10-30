#!/bin/bash

echo "Setting up environment for Day 25: Random Variables and Probability Distributions"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

echo "Detected Python version: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"; then
    echo "Error: Python 3.11 or higher required"
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
echo "Setup complete! To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "Then run the lesson code:"
echo "python lesson_code.py"
