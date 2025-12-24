# Day 48: Logistic Regression Theory

## Overview
Understand the mathematical foundations of logistic regression—the most deployed classification algorithm in production AI systems. Learn why it's called "regression" but does classification, and how sigmoid functions and log-loss enable probabilistic predictions.

## Learning Objectives
- Master the sigmoid function and its properties
- Understand log-loss (binary cross-entropy)
- Analyze decision boundaries and threshold tuning
- Connect theory to real-world applications (spam filters, fraud detection, recommendation systems)

## Quick Start

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run the Lesson
```bash
python lesson_code.py
```

Expected output:
- Linear to probability conversion examples
- Log-loss behavior demonstration
- Threshold analysis
- Sigmoid visualization (saved as `sigmoid_curve.png`)

### 3. Run Tests
```bash
pytest test_lesson.py -v
```

All 30+ tests should pass, validating:
- Sigmoid function properties
- Log-loss calculations
- Decision boundary behavior
- Mathematical relationships

## Key Concepts

### 1. The Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
```
Maps any real number to probability (0, 1)

### 2. Log-Loss (Binary Cross-Entropy)
```
Loss = -[y·log(p) + (1-y)·log(1-p)]
```
Penalizes confident wrong predictions exponentially

### 3. Decision Boundaries
- Standard threshold: 0.5
- Production: Tuned based on costs
- Example: Spam filter uses 0.3, fraud detection uses 0.8

## Real-World Applications

**Gmail Spam Filter**: Logistic regression with engineered features
**Netflix Recommendations**: Predicts watch probability
**Credit Card Fraud**: High-confidence blocking only
**Tesla Autopilot**: Object classification with confidence scores

## File Structure
```
day48_logistic_regression_theory/
├── setup.sh              # Environment setup
├── requirements.txt      # Dependencies
├── lesson_code.py        # Core implementations
├── test_lesson.py        # Comprehensive tests
├── README.md            # This file
└── sigmoid_curve.png    # Generated visualization
```

## What's Next?

**Day 49**: Implement logistic regression from scratch and with scikit-learn for binary classification. Build a spam email classifier and learn hyperparameter tuning.

## Production Insights

- **Speed**: Predictions in microseconds (billions daily)
- **Interpretability**: Can explain why a prediction was made
- **Probabilistic**: Confidence scores enable threshold tuning
- **Scalable**: From prototype to production without rewrite

## Common Questions

**Q: Why is it called "regression" if it does classification?**
A: Historical reasons. It uses regression (linear combination) then transforms output via sigmoid.

**Q: When should I use deep learning instead?**
A: When you have massive data, complex patterns, or need to learn features automatically. For structured data with engineered features, logistic regression often wins.

**Q: How do I handle imbalanced data?**
A: Covered in Day 49. Preview: adjust class weights, use SMOTE, or tune threshold.

## Resources
- Course documentation: Day 47 (Housing Prices) ← Previous | Next → Day 49 (Implementation)
- Scikit-learn logistic regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Stanford CS229 Lecture Notes on Classification

---

**Remember**: Every binary decision in modern AI likely uses logistic regression somewhere in the pipeline. You're learning the foundation of production AI systems.
