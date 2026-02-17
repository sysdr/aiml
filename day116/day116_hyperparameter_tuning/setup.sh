#!/bin/bash
# Day 116 Environment Setup
set -e

echo "Setting up Day 116: Hyperparameter Tuning Theory"

# Check Python version
python3 --version | grep -E "3\.(11|12)" > /dev/null 2>&1 || {
  echo "WARNING: Python 3.11+ recommended. Continuing anyway..."
}

# Create virtual environment
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "[OK] Virtual environment created"
fi

# Activate and install
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "[OK] Dependencies installed"

echo ""
echo "Setup complete. Run:"
echo "  source .venv/bin/activate"
echo "  python lesson_code.py       # full demo"
echo "  pytest test_lesson.py -v    # run tests"
echo "  jupyter notebook            # interactive"
