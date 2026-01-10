# Day 63: KNN with Scikit-learn

Production-grade K-Nearest Neighbors classification using scikit-learn, the industry-standard
machine learning library used by companies like Netflix, Spotify, and Airbnb.

## Quick Start

```bash
# 1. Setup environment
chmod +x install.sh && ./install.sh
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Run complete pipeline
python lesson_code.py --mode all

# 3. Run tests
pytest test_lesson.py -v
```

## What You'll Learn

- **Production ML Pipeline**: Complete workflow from data loading to model deployment
- **Hyperparameter Optimization**: Grid search with cross-validation (140 model configurations)
- **Feature Scaling**: Critical preprocessing for distance-based algorithms
- **Model Evaluation**: Comprehensive metrics used by industry leaders

## Pipeline Modes

```bash
# Explore dataset statistics
python lesson_code.py --mode explore

# Test feature scaling
python lesson_code.py --mode scale

# Train baseline model
python lesson_code.py --mode train --k 5

# Optimize hyperparameters (takes ~2 minutes)
python lesson_code.py --mode optimize

# Evaluate with visualizations
python lesson_code.py --mode evaluate

# Run everything
python lesson_code.py --mode all
```

## Expected Output

- **confusion_matrix.png**: Which classes get confused with each other
- **decision_boundary.png**: Visual representation of classification regions
- **Classification report**: Precision, recall, F1-score per class
- **Cross-validation scores**: Model stability across different data splits

## Real-World Connections

- **Spotify**: Music recommendations using k=50 neighbors in 450-dimensional space
- **Airbnb**: Listing search with approximate nearest neighbors (7M listings)
- **Amazon**: "Customers who bought this" using hierarchical KNN
- **Pandora**: Music Genome Project with distance-weighted KNN

## Testing

The test suite covers 15+ scenarios including:
- Edge cases (single samples, large k values)
- Data leakage detection (scaler fitting)
- Reproducibility verification
- Stratified splitting validation

```bash
pytest test_lesson.py -v          # Verbose output
pytest test_lesson.py -k "edge"   # Run edge case tests only
```

## Key Takeaways

1. **Scikit-learn is production-ready**: Same API from prototype to deployment
2. **Feature scaling is critical**: Without it, KNN fails completely
3. **Hyperparameter tuning is automated**: Grid search replaces manual guessing
4. **Comprehensive evaluation prevents overfitting**: Multiple metrics catch issues

## Next Steps

Tomorrow (Day 64): Support Vector Machines - solving KNN's scaling problem by finding
optimal decision boundaries instead of storing all training data.

## Dependencies

- Python 3.11+
- scikit-learn 1.4.0
- numpy 1.26.3
- pandas 2.2.0
- matplotlib 3.8.2
- pytest 7.4.4

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'sklearn'"
**Solution**: Activate virtual environment first: `source venv/bin/activate`

**Issue**: Grid search takes too long
**Solution**: Reduce cv_folds in optimize_hyperparameters() or use fewer parameter combinations

**Issue**: Visualizations not displaying
**Solution**: Check that matplotlib backend is configured. Add `plt.show()` if needed.
