#!/bin/bash

echo "🏗️  Setting up Day 46: Multiple Linear Regression Environment..."

# Check Python version
python_version=$(python3.11 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.11+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the lesson code:"
echo "  python lesson_code.py"
echo ""
echo "To run tests:"
echo "  pytest test_lesson.py -v"
