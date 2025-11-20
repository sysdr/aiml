#!/bin/bash
set -e

echo "🔧 Setting up Day 33: Introduction to Pandas..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "Run: source venv/bin/activate"
echo "Then: python lesson_code.py"
