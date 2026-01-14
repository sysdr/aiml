#!/bin/bash

echo "Setting up Day 72: Data Preprocessing and Feature Scaling"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Activate environment with: source venv/bin/activate"
