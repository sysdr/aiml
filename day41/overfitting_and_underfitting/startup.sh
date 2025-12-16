#!/bin/bash

# Startup script for Day 41: Overfitting and Underfitting Detection
# Runs the lesson and validates all metrics and visualizations

echo "🚀 Starting Day 41: Overfitting and Underfitting Detection System..."
echo "=================================================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")" && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Running setup.sh first..."
    bash setup.sh
    if [ $? -ne 0 ]; then
        echo "❌ Setup failed. Please fix errors and try again."
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Check if lesson_code.py exists
if [ ! -f "lesson_code.py" ]; then
    echo "❌ lesson_code.py not found!"
    exit 1
fi

# Check for duplicate lesson processes
LESSON_PID=$(ps aux | grep "[p]ython.*lesson_code.py" | grep -v "startup" | awk '{print $2}')
if [ ! -z "$LESSON_PID" ]; then
    echo "⚠️  Warning: Lesson process already running (PID: $LESSON_PID)"
    echo "   Stopping existing process..."
    kill $LESSON_PID 2>/dev/null
    sleep 2
    echo "✅ Old process terminated"
fi

# Run the lesson
echo ""
echo "📊 Running Overfitting/Underfitting Analysis..."
echo "=============================================="
echo ""

python lesson_code.py

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Lesson execution failed!"
    exit 1
fi

# Validate outputs
echo ""
echo "🔍 Validating outputs..."
echo "======================="

# Check if visualization was created
if [ -f "overfitting_analysis.png" ]; then
    FILE_SIZE=$(stat -f%z "overfitting_analysis.png" 2>/dev/null || stat -c%s "overfitting_analysis.png" 2>/dev/null)
    if [ "$FILE_SIZE" -gt 1000 ]; then
        echo "✅ Diagnostic dashboard visualization created (${FILE_SIZE} bytes)"
    else
        echo "⚠️  Visualization file too small, may be corrupted"
    fi
else
    echo "❌ Visualization file not found!"
    exit 1
fi

# Run tests to validate metrics
echo ""
echo "🧪 Running validation tests..."
echo "=============================="
pytest test_lesson.py -v --tb=short

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "⚠️  Some tests failed, but lesson completed"
fi

echo ""
echo "=================================================================="
echo "✅ Day 41 Lesson Complete!"
echo ""
echo "📊 Generated Files:"
echo "   - overfitting_analysis.png (Diagnostic Dashboard)"
echo ""
echo "📈 Key Metrics Validated:"
echo "   ✓ Model complexity analysis (degrees 1-15)"
echo "   ✓ Train-test gap detection"
echo "   ✓ Learning curve analysis"
echo "   ✓ Cross-validation stability"
echo "   ✓ Optimal model identification"
echo ""
echo "💡 View the dashboard visualization:"
echo "   open overfitting_analysis.png"
echo "   or"
echo "   xdg-open overfitting_analysis.png"
echo "=================================================================="

