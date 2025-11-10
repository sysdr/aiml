#!/bin/bash

echo "Setting up Day 29: Central Limit Theorem environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if (( $(echo "$python_version < $required_version" | bc -l) )); then
    echo "Warning: Python 3.11+ recommended. You have Python $python_version"
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then run the lesson:"
echo "  python lesson_code.py"
echo ""
echo "Or run tests:"
echo "  pytest test_lesson.py -v"
