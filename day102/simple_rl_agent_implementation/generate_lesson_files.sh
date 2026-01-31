#!/bin/bash

echo "Generating Day 102 implementation files..."

# Create setup.sh (nested)
cat > setup.sh << 'SETUP_EOF'
#!/bin/bash

echo "Setting up Day 102: Simple RL Agent Environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Setup complete! Activate environment with: source venv/bin/activate"
SETUP_EOF

chmod +x setup.sh

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.24.3
matplotlib==3.7.2
pytest==7.4.0
pytest-cov==4.1.0
