#!/bin/bash

# Day 104: Collaborative Filtering - Implementation Package Generator
# This script creates all necessary files for the lesson

set -e  # Exit on error

echo "Generating Day 104: Collaborative Filtering implementation files..."

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.26.4
scipy==1.12.0
pandas==2.2.1
scikit-learn==1.4.1
matplotlib==3.8.3
seaborn==0.13.2
pytest==8.0.2
pytest-cov==4.1.0
jupyter==1.0.0
flask==3.0.0
flask-cors==4.0.0
requests==2.31.0
EOF

# Create setup.sh (for Python venv setup)
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 104: Collaborative Filtering environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Setup complete! Activate the environment with: source venv/bin/activate"
EOF

chmod +x setup.sh

# Create lesson_code.py (truncated for brevity - will add full content)
echo "Creating lesson_code.py..."
