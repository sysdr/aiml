#!/bin/bash

echo "🔍 Verifying fraud detection system setup..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check Python packages
echo "📦 Checking required packages..."
python3 -c "import sklearn; print('✓ scikit-learn')" || echo "✗ scikit-learn"
python3 -c "import imblearn; print('✓ imbalanced-learn')" || echo "✗ imbalanced-learn"
python3 -c "import pandas; print('✓ pandas')" || echo "✗ pandas"
python3 -c "import numpy; print('✓ numpy')" || echo "✗ numpy"

echo "✅ Setup verification complete!"
