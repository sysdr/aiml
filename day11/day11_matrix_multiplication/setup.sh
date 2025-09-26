#!/bin/bash

echo "🔧 Setting up Python environment for Day 11: Matrix Multiplication and Dot Products..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= 3.10 required)"
else
    echo "❌ Python 3.10+ required. Current version: $python_version"
    echo "Please install Python 3.10+ and try again."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv matrix_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source matrix_env/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "💡 To activate the environment manually, run: source matrix_env/bin/activate"
echo "🎯 To start the lesson, run: python lesson_code.py"
