#!/bin/bash
echo "Setting up Day 131 environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Setup complete. Run: source venv/bin/activate"
