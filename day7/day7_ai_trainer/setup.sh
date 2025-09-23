#!/bin/bash

echo "🚀 Setting up AI/ML Python Environment for Day 7"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d" " -f2)
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete! Run 'source venv/bin/activate' (or 'venv\Scripts\activate' on Windows) to activate the environment"
echo "Then run: python lesson_code.py"
