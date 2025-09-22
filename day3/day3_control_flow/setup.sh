#!/bin/bash

echo "Setting up Python environment for Day 3: Control Flow"
echo "======================================================"

# Check if Python 3.11+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Found Python $PYTHON_VERSION"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv ai_env

# Activate virtual environment
echo "🔗 Activating virtual environment..."
source ai_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing required packages..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete! To activate your environment, run:"
echo "   source ai_env/bin/activate"
echo ""
echo "🎯 To start the lesson, run:"
echo "   python lesson_code.py"
echo ""
echo "🧪 To run tests, execute:"
echo "   python test_lesson.py"
