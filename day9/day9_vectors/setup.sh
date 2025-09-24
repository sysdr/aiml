#!/bin/bash

echo "🔧 Setting up Day 9: Vectors and Vector Operations environment..."

# Check if Python 3.11+ is installed
python_version=$(python3 --version 2>&1 | grep -o '3\.[0-9]*' | head -1)
if [ -z "$python_version" ]; then
    echo "❌ Python 3.11+ is required. Please install Python first."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📁 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the lesson:"
echo "  python lesson_code.py"
echo ""
echo "To run tests:"
echo "  python test_lesson.py"
