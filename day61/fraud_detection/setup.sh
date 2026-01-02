#!/bin/bash

echo "🔧 Setting up Credit Card Fraud Detection Environment..."

# Check Python version
python3 --version || { echo "Python 3.11+ required"; exit 1; }

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo "📊 To activate: source venv/bin/activate"
echo "▶️  To run: python lesson_code.py"
echo "🧪 To test: pytest test_lesson.py -v"
