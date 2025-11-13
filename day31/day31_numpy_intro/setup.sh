#!/bin/bash

echo "Setting up Day 31: Introduction to NumPy"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Try to create virtual environment
echo "Creating virtual environment..."
if python3 -m venv venv 2>/dev/null; then
    VENV_CREATED=true
else
    echo "Warning: Could not create venv, trying alternative method..."
    # Try installing pip first with --user
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &>/dev/null; then
        echo "Installing pip for user..."
        curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py --user --break-system-packages 2>/dev/null || python3 get-pip.py --user 2>/dev/null
        rm -f get-pip.py
        export PATH="$HOME/.local/bin:$PATH"
    fi
    # Try venv again
    python3 -m venv venv 2>/dev/null && VENV_CREATED=true || VENV_CREATED=false
fi

# Activate virtual environment if created, otherwise use system python
if [ "$VENV_CREATED" = true ]; then
    source venv/bin/activate
    PIP_CMD="pip"
    PYTHON_CMD="python"
else
    echo "Using system Python with --user flag for packages"
    PIP_CMD="python3 -m pip install --user"
    PYTHON_CMD="python3"
    export PATH="$HOME/.local/bin:$PATH"
fi

# Upgrade pip
echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip --quiet --break-system-packages 2>/dev/null || \
$PYTHON_CMD -m pip install --upgrade pip --quiet --user 2>/dev/null || \
$PYTHON_CMD -m pip install --upgrade pip --quiet 2>/dev/null || true

# Install requirements
echo "Installing dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt --quiet --break-system-packages 2>/dev/null || \
$PYTHON_CMD -m pip install -r requirements.txt --quiet --user 2>/dev/null || \
$PYTHON_CMD -m pip install -r requirements.txt --quiet 2>/dev/null

echo ""
echo "Setup complete! To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then run the main lesson code:"
echo "  python lesson_code.py"
echo ""
echo "Or run tests:"
echo "  pytest test_lesson.py -v"
