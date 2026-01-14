#!/bin/bash

echo "Setting up Scikit-learn Ecosystem Lesson Environment..."

# Check Python version - try python3.11 first, then python3
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    python_version=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    required_version="3.11"
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        echo "Error: Python 3.11+ required. Found: $python_version"
        exit 1
    fi
else
    echo "Error: Python 3 not found"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate  # Linux/Mac"
echo "  venv\\Scripts\\activate    # Windows"
echo ""
echo "To run the lesson:"
echo "  python lesson_code.py"
echo ""
echo "To run tests:"
echo "  pytest test_lesson.py -v"
