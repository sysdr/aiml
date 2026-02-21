# Day 128: Activation Functions

**Module 4: Deep Learning | Week 19-20: Neural Networks from Scratch**

## Quick Start

```bash
# 1. Set up environment
chmod +x setup.sh && ./setup.sh
source venv/bin/activate

# 2. Run all tests
python test_lesson.py

# 3. Visualize activation functions + XOR demo
python lesson_code.py --demo

# 4. Verify gradients
python lesson_code.py --check
```

## Docker

```bash
docker build -t day128 .
docker run day128 python test_lesson.py
```

## What You'll Build

| File | Description |
|------|-------------|
| `lesson_code.py` | ReLU, Sigmoid, Tanh, Softmax with forward + backward |
| `test_lesson.py` | 20+ tests including PyTorch comparison & gradient checks |

## Key Concepts

- **Why activations?** Without them, deep networks collapse to a single linear layer
- **ReLU family** — hidden layers in 95% of production models
- **Sigmoid** — binary outputs, LSTM/GRU gates
- **Tanh** — zero-centered, LSTM cell state
- **Softmax** — multiclass output, LLM token prediction

## Next: Day 129 — Multi-Layer Perceptrons

Wire these activations between configurable layers, train on real data, solve XOR.
