#!/bin/bash

echo "🎬 Setting up Movie Recommender System Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download MovieLens 100K dataset
echo "Downloading MovieLens 100K dataset..."
if [ ! -d "data/ml-100k" ]; then
    mkdir -p data
    cd data
    curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
    unzip -q ml-100k.zip
    rm ml-100k.zip
    cd ..
    echo "✅ Dataset downloaded and extracted"
else
    echo "✅ Dataset already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the recommender system:"
echo "  python main.py"
echo ""
echo "To run tests:"
echo "  pytest -v tests/ --cov=. --cov-report=term-missing"
