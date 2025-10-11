#!/bin/bash

echo "🔧 Setting up Day 14: Chain Rule and Partial Derivatives environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "⚠️  Warning: Python $required_version+ recommended. Current: $python_version"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv chain_rule_env
source chain_rule_env/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo "💡 To activate: source chain_rule_env/bin/activate"
echo "🎯 To run lesson: python lesson_code.py"
