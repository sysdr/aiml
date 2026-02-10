# Day 113: Gradient Boosting Machines

## Overview

This lesson implements production-grade Gradient Boosting from scratch, demonstrating sequential ensemble learning for binary classification. Build fraud detection systems mirroring real-world transaction monitoring at PayPal, Stripe, and Square.

## Quick Start

```bash
# Setup environment
./setup.sh
source venv/bin/activate

# Run the lesson
python lesson_code.py

# Run tests
pytest test_lesson.py -v
```

## What You'll Learn

1. **Sequential Error Correction**: How boosting builds ensembles where each model targets previous errors
2. **Weak Learners**: Why shallow decision trees (depth 3-6) outperform deep trees in boosting
3. **Gradient Descent in Function Space**: Training as optimization through residual targeting
4. **Production Implementation**: Real-world patterns from fraud detection systems processing millions of transactions

## Architecture

```
GradientBoostingClassifier
├── Initialization: Log-odds of positive class
├── Sequential Training Loop:
│   ├── Compute residuals (actual - predicted)
│   ├── Train weak learner on residuals
│   ├── Update predictions with learning_rate * new_predictions
│   └── Calculate training loss
└── Prediction: Aggregate all trees with sigmoid transformation

FraudDetectionSystem
├── Data Generation: Realistic transaction patterns
├── Model Training: 50-100 trees, learning_rate 0.05-0.1
├── Evaluation: Accuracy, precision, recall, F1, ROC-AUC
└── Visualization: Loss curves, feature importance
```

## Key Concepts

### Boosting Fundamentals
- **Sequential Learning**: Each tree corrects errors from previous ensemble
- **Weak Learners**: Shallow trees (depth 3-6) prevent overfitting
- **Learning Rate**: Controls contribution of each tree (0.01-0.3)
- **Residual Targeting**: New models fit to (actual - predicted)

### Loss Functions
- **Log Loss (Binary)**: Quantifies prediction confidence errors
- **Gradient Computation**: Residuals = actual - predicted_probability
- **Optimization**: Minimize loss through sequential corrections

### Hyperparameters
- `n_estimators`: Number of trees (100-500 typical)
- `learning_rate`: Shrinkage parameter (0.01-0.3)
- `max_depth`: Tree depth (3-6 for weak learners)
- `subsample`: Data fraction per tree (0.5-1.0)

## Production Patterns

### Real-World Applications
- **Google Search**: Result ranking optimization
- **Uber ETA**: Trip duration prediction
- **Stripe Fraud**: Transaction risk scoring
- **Netflix**: Recommendation refinement

### Performance Characteristics
- **Latency**: <10ms per prediction (single transaction)
- **Throughput**: 10,000+ predictions/second
- **Accuracy**: 20-40% improvement over single models
- **Interpretability**: Traceable through tree sequences

## Implementation Details

### Custom GBM
```python
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Fraud Detection Pipeline
```python
system = FraudDetectionSystem(n_estimators=50)
X, y = system.generate_transaction_data(n_samples=5000)
system.train(X_train, y_train)
metrics = system.evaluate(X_test, y_test)
```

## Testing

Comprehensive test suite validates:
- ✅ Model initialization and configuration
- ✅ Training convergence and loss reduction
- ✅ Prediction accuracy and consistency
- ✅ Feature importance calculation
- ✅ Edge cases (single sample, binary features)
- ✅ Production requirements (latency, serialization)

Run tests: `pytest test_lesson.py -v`

## Expected Results

### Performance Metrics
- **Accuracy**: 85-95% on fraud detection
- **ROC-AUC**: 0.90+ typical
- **Training Time**: 2-5 seconds (50 trees, 2000 samples)
- **Prediction Latency**: <5ms per transaction

### Comparison with scikit-learn
Custom implementation matches scikit-learn within 2-3% accuracy, validating architecture correctness.

## Next Steps

**Day 114: XGBoost and LightGBM**
- Industrial-strength implementations
- Histogram-based splitting (5-10x faster)
- GPU acceleration
- Advanced regularization
- Real-time serving infrastructure

## Resources

- Lesson Article: `lesson_article.md`
- Implementation: `lesson_code.py`
- Tests: `test_lesson.py`
- Visualizations: `gbm_training_analysis.png`

## Common Issues

**Slow Training**: Reduce `n_estimators` or increase `learning_rate`
**Overfitting**: Decrease `max_depth` or increase regularization
**Poor Performance**: Increase `n_estimators` or adjust `learning_rate`

## Production Deployment

```python
# Save model
import joblib
joblib.dump(model, 'fraud_detector_v1.pkl')

# Load and predict
model = joblib.load('fraud_detector_v1.pkl')
risk_score = model.predict_proba(transaction_features)[:, 1]
```

---

**Completion Time**: 2-3 hours
**Difficulty**: Intermediate
**Prerequisites**: Decision Trees, Ensemble Methods
