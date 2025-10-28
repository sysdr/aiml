#!/bin/bash

echo "🔧 Setting up Day 24: Conditional Probability and Bayes' Theorem"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete! Activate your environment with: source venv/bin/activate"
echo "📚 Run the lesson with: python lesson_code.py"
echo "🧪 Run tests with: python test_lesson.py"
