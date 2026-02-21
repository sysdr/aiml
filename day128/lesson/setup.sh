#!/bin/bash
echo "Setting up Day 128 environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Environment ready. Run: source venv/bin/activate"
