# Day 129: Multi-Layer Perceptrons (MLPs)

## Quick Start (< 5 minutes)

```bash
# 1. Setup environment
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate

# 2. Run the lesson demos
python lesson_code.py

# 3. Run the test suite
pytest test_lesson.py -v

# 4. Check output
open activation_distributions.png   # macOS
xdg-open activation_distributions.png  # Linux
```

## Without Docker

Requirements: Python 3.11+

```bash
pip install -r requirements.txt
python lesson_code.py
pytest test_lesson.py -v
```

## With Docker

```bash
docker run --rm -v $(pwd):/app -w /app python:3.11-slim bash -c \
  "pip install -r requirements.txt -q && python lesson_code.py && pytest test_lesson.py -v"
```

## Expected Output

```
==================================================
Demo 1: XOR Problem — Why Layers Matter
==================================================
 MLP Architecture Summary
────────────────────────────────────────
  Layer 1 (Hidden 1): (2, 4) ...
  Layer 2 (Output):   (4, 1) ...
  Total trainable parameters: 17

Forward Pass Results (untrained — random weights):
  Input: [0 0]  Expected: 0  Predicted: 0.XXXX
  ...

==================================================
Demo 2: Fraud Detection MLP (Production Simulation)
==================================================
...
Processed 100 transactions: ...

Saved: activation_distributions.png
Day 129 complete. Tomorrow: Backpropagation.
```

## Test Results

```
test_lesson.py::TestActivationFunctions::test_relu_positive_passthrough PASSED
test_lesson.py::TestActivationFunctions::test_sigmoid_zero_input PASSED
... (20+ tests)
```

## Key Concepts Covered

- **Why single layers fail**: Linear separability limitation
- **MLP architecture**: Input → Hidden(s) → Output layer flow
- **Weight matrices**: Shape rules and He initialization
- **Forward propagation**: Z = X·W + b → A = activation(Z)
- **Cache**: Why we store Z and A (needed for Day 130 backprop)

## Next Lesson

Day 130: The Backpropagation Algorithm — gradient computation via chain rule.
