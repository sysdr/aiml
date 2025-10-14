#!/bin/bash

echo "🔧 Setting up Day 15: Gradients and Gradient Descent"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (compatible)"
else
    echo "❌ Python 3.11+ required. Current: $python_version"
    echo "Please install Python 3.11+ and try again."
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv gradient_env
source gradient_env/bin/activate || . gradient_env/Scripts/activate

# Install requirements
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
echo "🧪 Testing installation..."
python3 -c "import numpy, matplotlib; print('✅ All dependencies installed successfully')"

# Create Jupyter kernel
echo "🔬 Setting up Jupyter kernel..."
python3 -m ipykernel install --user --name gradient_env --display-name "Day 15 - Gradients"

echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "source gradient_env/bin/activate  # Linux/Mac"
echo ". gradient_env/Scripts/activate   # Windows Git Bash"
echo ""
echo "To run the lesson:"
echo "python3 lesson_code.py"
echo ""
echo "To run tests:"
echo "python3 test_lesson.py"
echo ""
echo "To start Jupyter:"
echo "jupyter notebook"
