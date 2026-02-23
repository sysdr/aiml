#!/usr/bin/env bash
set -e
echo "Setting up Day 129 environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Setup complete. Activate with: source .venv/bin/activate"
