#!/bin/bash

echo "🎲 Starting Probability Dashboard..."
echo "📊 Dashboard will be available at: http://localhost:5000"
echo "🔄 Real-time simulations will update automatically"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

cd /home/sds/git/aiml/day23
source venv/bin/activate
python dashboard.py
