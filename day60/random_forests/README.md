# Day 60: Random Forests and Ensemble Methods

Production-grade implementation of Random Forest classifiers for customer churn prediction, demonstrating why ensemble methods dominate real-world AI systems.

## What You'll Learn

- Why single decision trees fail in production and how ensembles solve this
- Bagging and bootstrap aggregating for model robustness
- Random Forest algorithm with feature randomness
- Out-of-Bag (OOB) error for free validation
- Feature importance analysis for production deployment
- Ensemble diversity and why crowds beat experts

## Real-World Applications

- **Spotify**: 500-tree Random Forests for music recommendations
- **DoorDash**: Delivery time prediction with ensemble methods
- **Credit Karma**: Credit score predictions using 1000-tree forests
- **Netflix**: Content recommendation systems
- **Airbnb**: Price prediction models

## Quick Start

### Setup (5 minutes)

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Run the Lesson

```bash
# Execute main lesson code
python lesson_code.py
```

Expected output:
- Comparison of single tree vs Random Forest performance
- Feature importance analysis and visualization
- Ensemble diversity demonstration
- Production deployment insights

### Run Tests

```bash
# Run comprehensive test suite
pytest test_lesson.py -v
```

All 18 tests should pass, covering:
- Data generation and validation
- Model training and hyperparameters
- Ensemble superiority over single trees
- Evaluation metrics (accuracy, precision, recall, F1, ROC AUC)
- OOB score calculation
- Feature importance analysis
- Prediction consistency and diversity
- Error handling
- Scalability

## Project Structure

```
.
├── setup.sh              # Environment setup
├── requirements.txt      # Python dependencies
├── lesson_code.py        # Main implementation
├── test_lesson.py        # Test suite (18 tests)
├── README.md            # This file
└── feature_importance.png  # Generated visualization
```

## Key Concepts

### 1. The Ensemble Advantage

Single decision tree: One expert's opinion (prone to overfitting)
Random Forest: 200 experts voting (robust and reliable)

### 2. Bagging (Bootstrap Aggregating)

```
Original data: 10,000 samples
↓
Create 200 bootstrap samples (with replacement)
↓
Train 200 decision trees
↓
Combine via voting (classification) or averaging (regression)
```

### 3. Feature Randomness

At each split, only consider √k random features (if k total features)
→ Ensures tree diversity
→ Prevents dominance by single powerful feature

### 4. Out-of-Bag (OOB) Score

~37% of data excluded from each bootstrap sample
→ Use for validation without separate validation set
→ Free performance estimate during training

## Production Hyperparameters

```python
RandomForestClassifier(
    n_estimators=200,        # 100-500 in production
    max_depth=15,            # Prevent overfitting
    min_samples_split=10,    # Require 10 samples to split
    min_samples_leaf=5,      # Require 5 samples in leaf
    max_features='sqrt',     # √k feature randomness
    bootstrap=True,          # Enable bagging
    oob_score=True,         # Calculate OOB error
    n_jobs=-1,              # Parallel training
    random_state=42
)
```

## Expected Results

### Performance Comparison

| Metric    | Single Tree | Random Forest | Improvement |
|-----------|-------------|---------------|-------------|
| Accuracy  | ~82-85%     | ~92-95%       | +10%        |
| Precision | ~78-82%     | ~89-93%       | +12%        |
| Recall    | ~80-84%     | ~91-94%       | +11%        |
| F1 Score  | ~79-83%     | ~90-93%       | +11%        |
| ROC AUC   | ~85-88%     | ~95-97%       | +10%        |

### Feature Importance (Top 5)

1. days_since_last_purchase (18.2%)
2. support_tickets (14.7%)
3. login_frequency (12.3%)
4. satisfaction_score (10.9%)
5. total_purchases (9.1%)

## Why Random Forests Dominate Production

1. **Robust to noise**: Outliers only affect a few trees
2. **No feature scaling**: Works with raw features
3. **Handles mixed types**: Categorical + numerical together
4. **Interpretable**: Feature importance for debugging
5. **Parallelizable**: Each tree trains independently

## Common Issues & Solutions

### Issue: OOB score significantly different from test accuracy

**Solution**: Check for data leakage or temporal ordering issues

### Issue: All trees make same predictions (low diversity)

**Solution**: Increase `max_features` randomness or reduce feature correlation

### Issue: Training too slow

**Solution**: Reduce `n_estimators` or `max_depth`, or use `n_jobs=-1`

### Issue: High variance between runs

**Solution**: Set `random_state` parameter for reproducibility

## Tomorrow's Lesson

**Day 61: Credit Card Fraud Detection Project**

Apply Random Forests to highly imbalanced data (<0.1% fraud rate) using:
- SMOTE for class balancing
- Precision-recall optimization
- Cost-sensitive learning
- Production deployment considerations

## Resources

- Scikit-learn Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forest
- Original Random Forest Paper: Breiman (2001)
- Production ML Best Practices: https://ml-ops.org/

## Questions?

Remember: One expert makes mistakes. A diverse crowd of experts, when they agree, is almost always right. This is why ensemble learning powers every high-stakes AI system in production.
