#!/bin/bash

# Day 99: Introduction to Reinforcement Learning - File Generator
# This script creates all necessary files for the RL lesson

echo "Generating Day 99: Introduction to Reinforcement Learning files..."

# Create setup.sh
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 99: Introduction to Reinforcement Learning environment..."

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

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! Environment ready."
echo "To activate the environment, run: source venv/bin/activate"
echo "To run the lesson: python lesson_code.py"
echo "To run tests: python -m pytest test_lesson.py -v"
EOF

chmod +x setup.sh
echo "✓ Created setup.sh"

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.26.4
matplotlib==3.8.3
gym==0.26.2
pytest==8.0.2
tabulate==0.9.0
EOF
echo "✓ Created requirements.txt"

echo "All files generated successfully!"
