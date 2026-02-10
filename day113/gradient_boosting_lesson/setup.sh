#!/bin/bash

echo "Setting up Gradient Boosting Machines learning environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✅ Setup complete! Activate environment with: source venv/bin/activate"
echo "📚 Run the lesson: python lesson_code.py"
echo "🧪 Run tests: pytest test_lesson.py -v"
