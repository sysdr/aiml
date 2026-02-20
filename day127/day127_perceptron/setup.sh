#!/bin/bash
echo "Setting up Day 127 environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo ""
echo "Setup complete. Activate with: source venv/bin/activate"
echo "Then run: python lesson_code.py"
