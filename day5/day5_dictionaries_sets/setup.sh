#!/bin/bash

echo "🔧 Setting up Python environment for Day 5..."

# Check Python version
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "✅ Python $python_version detected (using Python 3.10+ for compatibility)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv ai_env
source ai_env/bin/activate

# Install dependencies
echo "⬇️ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo "💡 To activate environment: source ai_env/bin/activate"
echo "🚀 Run the lesson: python lesson_code.py"
