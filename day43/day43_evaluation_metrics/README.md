# Day 43: Model Evaluation Metrics

## Quick Start

### Setup
```bash
chmod +x setup.sh
./setup.sh
```

### Run Main Implementation
```bash
python lesson_code.py
```

### Run Tests
```bash
pytest test_lesson.py -v
```

## What You'll Learn

- Calculate accuracy, precision, and recall from scratch
- Understand the precision-recall tradeoff
- Visualize confusion matrices
- Apply metrics to real-world scenarios

## Project Structure

```
day43_evaluation_metrics/
├── lesson_code.py       # Main implementation
├── test_lesson.py       # Test suite
├── requirements.txt     # Dependencies
├── setup.sh            # Setup script
└── README.md           # This file
```

## Key Concepts

1. **Accuracy**: Overall correctness, dangerous with imbalanced data
2. **Precision**: Trust metric - when model says YES, how often is it right?
3. **Recall**: Safety metric - of all actual YES cases, how many did we catch?

## Real-World Applications

- Healthcare: High recall for disease detection
- Spam filtering: High precision to avoid blocking legitimate emails
- Fraud detection: Balance both based on business cost tradeoffs
