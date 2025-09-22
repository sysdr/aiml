#!/bin/bash

echo "🔧 Setting up Python environment for Day 6..."

# Check if Python 3.11+ is available
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    python_version=$(python3.11 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
    major_version=$(echo $python_version | cut -d. -f1)
    minor_version=$(echo $python_version | cut -d. -f2)
    
    if [ "$major_version" -lt 3 ] || { [ "$major_version" -eq 3 ] && [ "$minor_version" -lt 10 ]; }; then
        echo "❌ Python 3.10+ required. Current version: $python_version"
        echo "Please install Python 3.10 or higher"
        exit 1
    fi
else
    echo "❌ Python not found. Please install Python 3.10 or higher"
    exit 1
fi

echo "✅ Python version $python_version detected (using $PYTHON_CMD)"

# Create virtual environment
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv ai_course_env

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source ai_course_env/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🎉 Setup complete! To activate the environment, run:"
echo "    source ai_course_env/bin/activate"
echo ""
echo "Then run the lesson:"
echo "    python lesson_code.py"
