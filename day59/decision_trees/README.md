# Day 59: Decision Trees with Scikit-learn

## Overview
Learn to implement production-ready decision tree classifiers for customer churn prediction, similar to systems used at Netflix, Spotify, and major streaming services.

## Quick Start

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run Main Lesson
```bash
python lesson_code.py
```

Expected output:
- Model training with cross-validation
- Feature importance rankings
- Performance metrics (ROC-AUC, accuracy, confusion matrix)
- Visualizations saved as PNG files

### 3. Run Tests
```bash
python test_lesson.py
```

All 20+ tests should pass, validating:
- Data generation quality
- Model training correctness
- Imbalanced dataset handling
- Feature importance calculation
- Production metrics

## What You'll Learn

### Core Concepts
1. **Production Decision Trees**: Using scikit-learn's battle-tested implementation
2. **Imbalanced Data Handling**: class_weight='balanced' for real-world datasets
3. **Feature Importance**: Understanding which customer behaviors predict churn
4. **Cross-Validation**: Reliable performance estimation
5. **Hyperparameter Tuning**: GridSearchCV for optimal parameters

### Real-World Applications
- Netflix: Subscription cancellation prediction
- Spotify: At-risk user identification
- Amazon: Prime membership renewal forecasting
- PayPal: Fraud detection with imbalanced data

## Files Generated
- `lesson_code.py`: Complete churn prediction implementation
- `test_lesson.py`: Comprehensive test suite (20+ tests)
- `churn_analysis.png`: Performance visualization
- `decision_tree_structure.png`: Tree structure diagram

## Key Features Demonstrated

### 1. Class Imbalance Handling
```python
clf = DecisionTreeClassifier(
    class_weight='balanced',  # Automatically adjusts for imbalance
    max_depth=10,
    min_samples_split=50
)
```

### 2. Cross-Validation
```python
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
```

### 3. Feature Importance
```python
importance = clf.feature_importances_
# Reveals which features drive churn predictions
```

### 4. Grid Search Tuning
```python
param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [20, 50, 100]
}
grid_search = GridSearchCV(clf, param_grid, cv=5)
```

## Production Insights

### Why Scikit-learn?
- **Battle-tested**: Used by thousands of production systems
- **Optimized**: C-based implementation for speed
- **Complete**: Handles edge cases you'd spend months discovering
- **Maintained**: Active development and bug fixes

### Key Hyperparameters
- `max_depth`: Controls overfitting (typical: 5-20)
- `min_samples_split`: Ensures statistical significance (typical: 20-100)
- `class_weight='balanced'`: Essential for imbalanced data
- `random_state`: Reproducibility for production

### Performance Metrics
- **Accuracy**: Often misleading for imbalanced data
- **ROC-AUC**: Better metric for classification quality
- **Confusion Matrix**: Shows false positive/negative trade-offs
- **Feature Importance**: Business insights for product teams

## Common Issues

### Issue: Low Performance on Test Set
**Solution**: Increase min_samples_split or reduce max_depth

### Issue: Only Predicts Majority Class
**Solution**: Use class_weight='balanced'

### Issue: High Variance in Cross-Validation
**Solution**: More data or stronger regularization

## Next Steps
Tomorrow (Day 60): Random Forests and Ensemble Methods
- Combining multiple decision trees
- Reducing overfitting through bagging
- Production-scale ensemble systems

## Resources
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/tree.html
- Feature Importance Guide: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
- Imbalanced Data: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

## Success Criteria
✓ Understand scikit-learn decision tree API
✓ Handle imbalanced datasets with class_weight
✓ Interpret feature importance for business insights
✓ Perform cross-validation for reliable estimates
✓ Tune hyperparameters with GridSearchCV
✓ Generate production-quality visualizations

**Estimated completion time**: 2-3 hours
