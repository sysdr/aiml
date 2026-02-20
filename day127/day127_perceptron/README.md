# Day 127: The Perceptron and Its Limitations
### 180-Day AI/ML Course — Module 4: Deep Learning

## Quick Start

```bash
# 1. Set up environment
chmod +x setup.sh && ./setup.sh
source venv/bin/activate

# 2. Run the lesson demo
python lesson_code.py

# 3. Run the test suite
pytest test_lesson.py -v

# Expected test output:
#   23 passed in <1s
```

## Without Virtual Environment (system Python)

```bash
pip install numpy matplotlib pytest
python lesson_code.py
pytest test_lesson.py -v
```

## With Docker

```bash
docker run --rm -v $(pwd):/app -w /app python:3.11-slim bash -c \
  "pip install -r requirements.txt -q && python lesson_code.py"
```

## What You'll See

- **AND gate training**: Perceptron converges in ~5 epochs, 100% accuracy
- **XOR gate training**: Perceptron oscillates, maxes at 75% accuracy
- **Visualization**: `perceptron_results.png` — decision boundaries + training error curves
- **Key insight**: The XOR failure is not a bug. It proved that *stacking* neurons
  (hidden layers) is required for non-linear problems — the birth of deep learning.

## Files

| File | Purpose |
|------|---------|
| `lesson_code.py` | Perceptron class + AND/XOR training demo |
| `test_lesson.py` | 20 pytest tests covering all core behaviors |
| `requirements.txt` | NumPy, Matplotlib, Pytest |
| `setup.sh` | Automated venv setup |

## Next: Day 128 — Activation Functions
We replace the step function with sigmoid, tanh, and ReLU,
enabling gradient flow through multiple layers.
