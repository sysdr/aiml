# Day 65: SVMs with Scikit-learn - Production Fraud Detection

## Overview

Learn to implement production-ready Support Vector Machine classifiers using scikit-learn. Build a complete fraud detection system with proper preprocessing, hyperparameter tuning, and deployment patterns.

## What You'll Learn

- Feature scaling pipelines for SVM (critical for performance)
- Hyperparameter tuning with GridSearchCV
- Different SVM kernels (linear, RBF, polynomial)
- Handling class imbalance with `class_weight='balanced'`
- Model persistence for production deployment
- Comprehensive evaluation metrics

## Prerequisites

- Python 3.11+
- Understanding of SVM theory (Day 64)
- Basic scikit-learn knowledge

## Quick Start

### 1. Setup Environment

```bash
# Run setup script
chmod +x setup_env.sh
./setup_env.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Run Main Lesson

```bash
python lesson_code.py
```

Expected output:
- Feature scaling impact demonstration
- Kernel comparison results
- GridSearchCV training progress
- Classification metrics
- Saved model file: `fraud_detector_svm_v1.pkl`
- Visualization: `svm_fraud_detection_results.png`

### 3. Run Tests

```bash
pytest test_lesson.py -v
```

All 11 tests should pass, verifying:
- Dataset generation
- SVM training with/without scaling
- Different kernel types
- Probability predictions
- Model persistence
- Class imbalance handling

## Project Structure

```
day-65-svms-scikit-learn/
├── lesson_code.py          # Main implementation
├── test_lesson.py          # Comprehensive tests
├── requirements.txt        # Dependencies
├── setup_env.sh           # Environment setup
├── README.md              # This file
└── venv/                  # Virtual environment (created by setup)
```

## Key Concepts Demonstrated

### 1. Feature Scaling Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Always scale for SVM!
    ('svm', SVC(kernel='rbf'))
])
```

**Why it matters**: SVMs calculate distances. Unscaled features dominate the distance calculation.

### 2. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 0.01, 0.1],
    'svm__kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
```

### 3. Handling Class Imbalance

```python
svm = SVC(class_weight='balanced')  # Automatically adjusts for imbalance
```

## Real-World Application

The fraud detection system demonstrates:

- **PayPal-style** transaction monitoring
- **100K+ transactions/day** processing patterns
- **4 key features**: amount, hour, distance, merchant risk
- **5% fraud rate** (realistic imbalance)
- **Production deployment** with model persistence

## Performance Benchmarks

With proper scaling and tuning:
- **Accuracy**: 95%+
- **F1 Score**: 0.75+ (on imbalanced data)
- **ROC-AUC**: 0.95+
- **Training time**: <30 seconds on 10K samples

## Common Issues & Solutions

### Issue: Poor accuracy despite good code
**Solution**: Ensure `StandardScaler` is in the pipeline. SVMs require scaling.

### Issue: GridSearchCV takes too long
**Solution**: Reduce parameter grid or use `RandomizedSearchCV` for large grids.

### Issue: Model predicts only majority class
**Solution**: Use `class_weight='balanced'` for imbalanced datasets.

### Issue: "Kernel not supported" error
**Solution**: Use 'linear', 'rbf', 'poly', or 'sigmoid' - these are the supported kernels.

## Next Steps

1. **Experiment**: Try different parameter combinations
2. **Add Features**: Include transaction velocity, user history
3. **Compare**: Test against Random Forest, XGBoost
4. **Deploy**: Load saved model in a Flask API
5. **Monitor**: Track false positives/negatives over time

## Resources

- [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [GridSearchCV Guide](https://scikit-learn.org/stable/modules/grid_search.html)
- [Pipeline Tutorial](https://scikit-learn.org/stable/modules/compose.html)

## Troubleshooting

### Setup Issues

```bash
# If setup_env.sh fails, manually create environment:
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Import Errors

```bash
# Ensure virtual environment is activated:
which python  # Should show: /path/to/venv/bin/python

# Reinstall dependencies:
pip install -r requirements.txt --force-reinstall
```

## Time Estimate

- **Setup**: 5 minutes
- **Run lesson**: 3-5 minutes
- **Review outputs**: 10 minutes
- **Experimentation**: 30+ minutes
- **Total**: ~1 hour

## Success Criteria

You've mastered this lesson when you can:
- ✓ Explain why feature scaling is critical for SVMs
- ✓ Tune C and gamma parameters effectively
- ✓ Choose appropriate kernels for different problems
- ✓ Handle imbalanced datasets with class weights
- ✓ Deploy models for production use

---

**Next Lesson**: Day 71 - The Scikit-learn Ecosystem (unified API patterns)

**Questions?** Review the code comments and test cases for detailed explanations.
