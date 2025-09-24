#!/bin/bash

echo "Setting up Day 10: Matrices and Matrix Operations environment..."

# Check if Python 3.11+ is available
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    python_version=$(python3.11 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
else
    echo "❌ Python 3.11+ required but not found"
    echo "Please install Python 3.11 or higher"
    exit 1
fi

major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 10 ]); then
    echo "❌ Python 3.10+ required. Current version: $python_version"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

echo "✅ Python $python_version detected (using $PYTHON_CMD)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete! To get started:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run the lesson: python lesson_code.py"
echo "3. Run tests: python test_lesson.py"
echo "4. Open Jupyter notebook: jupyter notebook matrices_lesson.ipynb"
