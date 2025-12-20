# Day 45: Linear Regression with Scikit-learn

Welcome to Day 45 of the 180-Day AI/ML Course! Today you'll learn how to implement production-ready linear regression using scikit-learn.

## What You'll Learn

- Scikit-learn's standardized fit/predict interface
- Training linear regression models on real data
- Evaluating model performance with multiple metrics
- Making predictions on new data
- Production ML workflow patterns

## Prerequisites

- Python 3.11+
- Completion of Day 44 (Simple Linear Regression Theory)
- Basic understanding of linear regression concepts

## Quick Start

### 1. Setup Environment

```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Run Lesson Code

```bash
python lesson_code.py
```

Expected output:
- Data exploration statistics
- Model training confirmation
- Performance metrics (R² > 0.80 target)
- Predictions for new data points
- Visualization saved as `regression_analysis.png`

### 3. Run Tests

```bash
pytest test_lesson.py -v
```

All 20+ tests should pass, verifying:
- Data preparation correctness
- Model training success
- Prediction accuracy
- Production readiness

## Files Overview

- `lesson_code.py` - Complete implementation with detailed comments
- `test_lesson.py` - Comprehensive test suite (20+ tests)
- `setup.sh` - Automated environment setup
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Key Concepts

### 1. The Fit/Predict Pattern

```python
model = LinearRegression()
model.fit(X_train, y_train)      # Learn from data
predictions = model.predict(X_test)  # Make predictions
```

### 2. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. Model Evaluation

```python
r2 = r2_score(y_test, y_pred)      # Coefficient of determination
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root mean squared error
```

## Success Criteria

✓ Model achieves R² > 0.80 on test data
✓ All tests pass
✓ Visualizations generated successfully
✓ Can make predictions for new data points

## Real-World Applications

This pattern is used for:
- **Salary prediction** (HR systems)
- **Sales forecasting** (retail)
- **Delivery time estimation** (logistics)
- **Price prediction** (real estate, e-commerce)
- **Resource allocation** (cloud computing)

Companies like Netflix, Uber, and DoorDash use these exact patterns scaled to millions of predictions per second.

## Troubleshooting

### Issue: Import errors
**Solution:** Ensure virtual environment is activated: `source venv/bin/activate`

### Issue: Low R² score (<0.60)
**Solution:** Check data quality, verify no missing values, ensure proper train/test split

### Issue: Tests failing
**Solution:** Run `python lesson_code.py` first to generate required files

## Next Steps

Tomorrow (Day 46), you'll extend to multiple linear regression with multiple features:
- Adding education level, location, job title
- Feature engineering and scaling
- Handling categorical variables
- More complex real-world scenarios

The fit/predict pattern remains identical - only the input complexity changes!

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Understanding R² Score](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score)
- [Train-Test Split Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)

## Questions or Issues?

The code includes extensive comments and error handling. If something isn't clear:
1. Review the inline comments in `lesson_code.py`
2. Check the test file for expected behavior
3. Refer to the main lesson article for conceptual understanding

Happy learning! 🚀
