# Day 50: Multi-Class Classification with Logistic Regression

Complete implementation of One-vs-Rest and Softmax strategies for multi-class classification.

## Quick Start

```bash
# Setup environment
bash setup.sh

# Activate virtual environment
source venv/bin/activate

# Run the news categorizer
python lesson_code.py

# Run tests
pytest test_lesson.py -v
```

## What's Included

- **One-vs-Rest (OvR) Classifier**: Trains separate binary models for each class
- **Softmax Classifier**: Single model with multi-output layer
- **Performance Comparison**: Side-by-side evaluation of both strategies
- **News Categorization**: Real-world example with 4 categories
- **Visualization**: Confusion matrices for both strategies

## Key Concepts

1. **Multi-Class Extension**: Converting binary classification to handle 3+ classes
2. **OvR Strategy**: Train N binary classifiers for N classes
3. **Softmax Strategy**: Single model outputting probabilities that sum to 1.0
4. **Feature Importance**: Understanding which words indicate each category

## Expected Output

- Training time for both strategies
- Accuracy comparison (typically 85-95% on synthetic data)
- Confusion matrices showing prediction patterns
- Top features per category
- Sample predictions with probability distributions

## Production Applications

- Gmail priority categorization
- Netflix genre classification
- Google Photos face recognition
- Amazon product categorization
- Medical diagnosis systems

## Build Time

Complete setup and execution: ~3 minutes

## Files Generated

- `one-vs-rest_confusion_matrix.png`: OvR visualization
- `softmax_confusion_matrix.png`: Softmax visualization
- Test results and performance metrics
