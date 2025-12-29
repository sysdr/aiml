#!/bin/bash

echo "Setting up Day 58: Decision Trees Theory Environment"
echo "===================================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "✓ Virtual environment created"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ Pip upgraded"

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed:"
pip list | grep -E "numpy|pandas|matplotlib|scikit-learn|pytest"

echo ""
echo "===================================================="
echo "Setup complete! Activate the environment with:"
echo "  source venv/bin/activate   (Linux/Mac)"
echo "  venv\Scripts\activate      (Windows)"
echo "===================================================="
