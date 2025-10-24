#!/bin/bash

echo "🔧 Setting up Day 23: Introduction to Probability"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📍 Found Python $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo "  1. source venv/bin/activate"
echo "  2. python lesson_code.py"
echo "  3. jupyter notebook (optional, for interactive learning)"
