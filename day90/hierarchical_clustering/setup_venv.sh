#!/bin/bash

echo "Setting up Hierarchical Clustering environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Setup complete! Activate the environment with: source venv/bin/activate"
