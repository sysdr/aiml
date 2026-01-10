#!/bin/bash

echo "Setting up Day 63: KNN with Scikit-learn..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✓ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate  (Linux/Mac)"
echo "  venv\\Scripts\\activate    (Windows)"
echo ""
echo "Then run: python lesson_code.py --mode all"
