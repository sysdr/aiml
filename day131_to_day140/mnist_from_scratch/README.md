# Day 131: MNIST Neural Network — Project Kickoff

## What You're Building
A 3-layer neural network (784 → 128 → 64 → 10) trained from scratch to classify
handwritten digits with no ML framework dependencies.

## Quick Start

```bash
# 1. Install dependencies
bash setup.sh
source venv/bin/activate

# 2. Run the forward pass demo
python lesson_code.py

# 3. Run all tests
pytest tests/test_lesson.py -v
```

## Expected Output

```
=== Day 131: MNIST Neural Network — Forward Pass Demo ===

Loading MNIST (first run downloads ~12MB)...
Train: (60000, 784)  |  Test: (10000, 784)
Label encoding check: y=7 → [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]

Network parameters: 109,386

--- Shape Verification ---
Input  : (32, 784)   expected (32, 784)
Output : (32, 10)    expected (32, 10)
Row sums (should be 1.0): [1. 1. 1. 1.]

Sample 0 — True: 5  |  Pred: ?  |  Confidence: ?%
(Untrained network — predictions are random)

✓ Forward pass verified. Ready for Day 132: Loss + Backprop.
```

## Architecture

```
Input (784)
    ↓  W1 (784×128) + b1
Hidden Layer 1 (128) — ReLU
    ↓  W2 (128×64) + b2
Hidden Layer 2 (64)  — ReLU
    ↓  W3 (64×10) + b3
Output (10)          — Softmax → Probabilities
```

## Project Timeline (Days 131–140)
| Day | Topic |
|-----|-------|
| 131 | Architecture, data pipeline, forward pass (today) |
| 132 | Loss function (cross-entropy) + backpropagation |
| 133 | Training loop + gradient descent |
| 134 | Batch training + learning curves |
| 135 | Evaluation: accuracy, confusion matrix |
| 136 | Regularization (dropout, L2) |
| 137 | Hyperparameter tuning |
| 138 | Visualization: weights, activations |
| 139 | Optimization: momentum, Adam |
| 140 | Final polish + 98%+ accuracy target |
