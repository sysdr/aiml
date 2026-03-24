#!/usr/bin/env bash
set -e
echo "==> Setting up Day 170 environment..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt -q

python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
python -m spacy download en_core_web_sm --quiet

echo ""
echo "✓ Environment ready."
echo "  Activate with: source .venv/bin/activate"
echo "  Run lesson:    python lesson_code.py"
echo "  Run tests:     pytest test_lesson.py -v"
