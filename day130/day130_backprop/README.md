# Day 130: The Backpropagation Algorithm

## Quick Start

```bash
# Option 1: Direct
pip install -r requirements.txt
python lesson_code.py
pytest test_lesson.py -v

# Option 2: Virtual environment
./setup.sh
source venv/bin/activate
python lesson_code.py

# Option 3: Docker
docker build -t day130 .
docker run --rm day130
```

## What You'll See

```
[1] Running gradient check...
    Relative error: 3.2e-08  ✓ PASS

[2] Training on XOR dataset...
    Epoch    0 | Loss: 0.312847
    Epoch  100 | Loss: 0.249012
    Epoch  500 | Loss: 0.023441
    Epoch  900 | Loss: 0.001832

[3] Final predictions vs ground truth:
    Input: [0. 0.] | Target: 0 | Predicted: 0.0094
    Input: [0. 1.] | Target: 1 | Predicted: 0.9887
    Input: [1. 0.] | Target: 1 | Predicted: 0.9901
    Input: [1. 1.] | Target: 0 | Predicted: 0.0071

[4] sigmoid final loss: 0.249921
    relu    final loss: 0.002341
```

## Key Concepts

| Concept | What it does |
|---|---|
| Forward pass | Compute predictions, cache intermediate values |
| Loss | Measure prediction error (MSE) |
| Backward pass | Chain rule computes dL/dW for every weight |
| Gradient check | Numerical verification of analytic gradients |
| ReLU vs Sigmoid | ReLU prevents vanishing gradients in deep nets |

## File Structure

```
day130_backprop/
├── lesson_code.py     # Full backprop implementation (no ML libs)
├── test_lesson.py     # 20 tests covering all components
├── requirements.txt
├── setup.sh
└── README.md
```

## Tests

```bash
pytest test_lesson.py -v
# 20 tests: activations, loss, forward pass, gradient check, convergence
```
