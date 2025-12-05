#!/bin/bash

echo "🚀 Starting Day 38: Machine Learning Workflow..."
echo ""

# Check if virtual environment exists
if [ ! -d "ml_workflow_env" ]; then
    echo "Virtual environment not found. Running setup..."
    chmod +x setup_env.sh
    ./setup_env.sh
    if [ $? -ne 0 ]; then
        echo "❌ Setup failed"
        exit 1
    fi
fi

# Run the ML workflow using virtual environment's Python
echo "Running the complete ML workflow..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ml_workflow_env/bin/python lesson_code.py

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Workflow completed successfully!"
    echo ""
    echo "📊 Model artifacts saved in ./models/"
    echo ""
    echo "To verify your understanding, run:"
    echo "  ml_workflow_env/bin/pytest test_lesson.py -v"
else
    echo "❌ Workflow failed"
    exit 1
fi

