#!/bin/bash

echo "🔧 Setting up Day 12: Introduction to Calculus for AI/ML environment..."

# Check if Python 3.11+ is installed
python_version=$(python3 --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
if [[ -z "$python_version" ]]; then
    echo "❌ Python 3 not found. Please install Python 3.11 or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv calculus_env
source calculus_env/bin/activate

# Install dependencies
echo "📚 Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete! Activate environment with: source calculus_env/bin/activate"
echo "🎯 Run the lesson with: python lesson_code.py"
