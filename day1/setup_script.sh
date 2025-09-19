#!/bin/bash

echo "🚀 Setting up your AI/ML Python Environment - Day 1"
echo "================================================="

# Check if Python 3.11+ is installed
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✅ Python found: $python_version"
else
    echo "❌ Python 3.11+ required. Please install Python from https://python.org"
    exit 1
fi

# Create virtual environment for AI course
echo "📦 Creating virtual environment..."
python3 -m venv ai_course_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ai_course_env/bin/activate || ai_course_env\Scripts\activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📚 Installing required packages..."
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found. Installing basic packages..."
    pip install jupyter ipython requests python-dotenv
fi

# Create necessary directories
echo "📁 Creating project structure..."
mkdir -p data
mkdir -p notebooks
mkdir -p tests

# Create a simple .env template
echo "🔑 Creating environment template..."
cat > .env << EOL
# Add your API keys here (Day 1 doesn't require any)
# GEMINI_API_KEY=your_api_key_here
# OPENAI_API_KEY=your_api_key_here

# Course settings
COURSE_DAY=1
STUDENT_NAME=Your Name Here
EOL

echo ""
echo "🎉 Setup Complete! Your AI/ML environment is ready!"
echo ""
echo "Next steps:"
echo "1. Run: source ai_course_env/bin/activate (Linux/Mac) or ai_course_env\\Scripts\\activate (Windows)"
echo "2. Run: python lesson_code.py"
echo "3. Open: jupyter notebook (for interactive learning)"
echo ""
echo "Happy coding! 🐍✨"