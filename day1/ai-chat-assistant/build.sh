#!/bin/bash

echo "🚀 Building AI Chat Assistant..."

# Build Backend
echo "🐍 Building Python Backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run backend tests
echo "🧪 Running backend tests..."
python -m pytest tests/ -v

echo "✅ Backend build complete!"

cd ..

# Build Frontend
echo "⚛️ Building React Frontend..."
cd frontend

# Install Node.js dependencies
npm install

# Run frontend tests
echo "🧪 Running frontend tests..."
npm test -- --coverage --watchAll=false

# Build for production
echo "📦 Building for production..."
npm run build

echo "✅ Frontend build complete!"

cd ..

echo "🎉 Build completed successfully!"
echo "📋 Next steps:"
echo "   1. Run './start.sh' to start the application"
echo "   2. Open http://localhost:3000 in your browser"
echo "   3. Start chatting with your AI assistant!"
