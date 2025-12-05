#!/bin/bash

echo "Setting up Day 38: Machine Learning Workflow Environment..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv ml_workflow_env

# Activate virtual environment
source ml_workflow_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source ml_workflow_env/bin/activate"
echo ""
echo "Then run the lesson:"
echo "  python lesson_code.py"
echo ""
echo "Or run tests:"
echo "  pytest test_lesson.py -v"
