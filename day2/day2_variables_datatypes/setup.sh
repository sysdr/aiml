#!/bin/bash

echo "🔧 Setting up Python environment for Day 2: Variables, Data Types, and Operators"

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv ai_course_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source ai_course_env/Scripts/activate
else
    source ai_course_env/bin/activate
fi

# Install requirements
echo "📥 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete! Run 'source ai_course_env/bin/activate' to start coding."
echo "🎯 Then run: python lesson_code.py"
