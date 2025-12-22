# Day 46: Multiple Linear Regression

## Overview

Learn how to build multi-feature prediction systems like Netflix, Google, and Tesla using multiple linear regression. This lesson extends simple linear regression to handle multiple input features simultaneously, enabling more accurate and realistic predictions.

## What You'll Learn

- Extend linear regression from one feature to multiple features
- Understand coefficient interpretation and feature importance
- Build production-style prediction systems
- Evaluate model performance with multiple metrics
- Visualize multi-dimensional relationships

## Prerequisites

- Completion of Day 45 (Linear Regression with Scikit-learn)
- Python 3.11 or higher
- Basic understanding of linear regression concepts

## Quick Start

### 1. Setup Environment

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup (creates venv and installs dependencies)
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Run the Lesson Code

```bash
python lesson_code.py
```

Expected output:
- Dataset generation with 500 house records
- Model training metrics (R², RMSE, MAE)
- Feature importance rankings
- Sample predictions
- Three visualization files (PNG)

### 3. Run Tests

```bash
# Run all tests with verbose output
pytest test_lesson.py -v

# Run specific test class
pytest test_lesson.py::TestModelTraining -v

# Run with coverage
pytest test_lesson.py --cov=lesson_code
```

Expected: 25+ tests passing

## Project Structure

```
day46_multiple_linear_regression/
├── lesson_code.py          # Main implementation
├── test_lesson.py          # Comprehensive test suite (25+ tests)
├── requirements.txt        # Python dependencies
├── setup.sh               # Environment setup script
├── README.md              # This file
├── feature_importance.png # Generated visualization
├── predictions.png        # Generated visualization
└── residuals.png          # Generated visualization
```

## Key Concepts

### Multiple Features

Unlike simple linear regression (one feature), multiple linear regression uses several features:

```
Simple:   y = b₀ + b₁x₁
Multiple: y = b₀ + b₁x₁ + b₂x₂ + b₃x₃ + ... + bₙxₙ
```

For house prices:
- x₁ = square_feet
- x₂ = bedrooms
- x₃ = age_years
- x₄ = location_score
- ... (8 features total)

### Feature Importance

Coefficients reveal which features matter most:

- **Positive coefficient**: Feature increases prediction
- **Negative coefficient**: Feature decreases prediction
- **Magnitude**: Relative importance

Example: If location_score coefficient is $50,000 and pool coefficient is $30,000, location matters more for pricing.

### Model Evaluation

Four key metrics:

1. **R² Score** (0-1): Proportion of variance explained (higher is better)
2. **RMSE**: Root Mean Squared Error in dollars (lower is better)
3. **MAE**: Mean Absolute Error in dollars (lower is better)
4. **MAPE**: Mean Absolute Percentage Error (lower is better)

## Usage Examples

### Basic Usage

```python
from lesson_code import MultipleLinearRegressionModel

# Initialize model
model = MultipleLinearRegressionModel()

# Generate dataset
df = model.generate_realistic_dataset(n_samples=500)

# Prepare data
X, y = model.prepare_data(df)

# Train model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"R² Score: {metrics['r2_score']:.4f}")

# Make predictions
predictions = model.predict(X_test)
```

### Feature Importance Analysis

```python
# Get feature importance
importance_df = model.get_feature_importance()
print(importance_df)

# Visualize
model.visualize_coefficients(save_path='importance.png')
```

### Model Persistence

```python
# Save trained model
model.save_model('my_model.joblib')

# Load later
new_model = MultipleLinearRegressionModel()
new_model.load_model('my_model.joblib')
predictions = new_model.predict(X_new)
```

## Real-World Applications

### Netflix Content Ranking
Uses 50+ features per title:
- Watch history
- Time of day
- Device type
- Completion rates
- Pause patterns
- Search behavior

### Google Search Ranking
Uses 200+ features:
- Keyword relevance
- Page authority
- Freshness
- User engagement
- Mobile-friendliness
- Loading speed

### Tesla Autopilot
Uses 30+ sensor inputs:
- Radar distance
- Camera detection
- GPS position
- Vehicle speed
- Steering angle
- Road curvature

## Common Issues & Solutions

### Issue: Low R² Score
**Solution**: 
- Check for missing important features
- Examine feature correlations (multicollinearity)
- Consider feature engineering

### Issue: Large Residuals
**Solution**:
- Check residual plots for patterns
- Consider non-linear relationships
- Remove outliers

### Issue: Unstable Coefficients
**Solution**:
- Check for multicollinearity (highly correlated features)
- Use feature selection
- Consider regularization (covered in later lessons)

## Performance Benchmarks

On a modern laptop (M1/M2 Mac or recent Intel):

| Operation | Time | Notes |
|-----------|------|-------|
| Generate 500 samples | <1s | Includes realistic noise |
| Train model | <0.1s | Linear time complexity |
| Predict 100 samples | <0.01s | Very fast inference |
| Full test suite | <5s | 25+ comprehensive tests |

## Next Steps

Tomorrow (Day 47): **Project Day - Predict Housing Prices**

You'll build a complete end-to-end system:
- Load real-world dataset
- Feature engineering
- Model training and evaluation
- API deployment for predictions
- Production-ready error handling

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [Multiple Linear Regression Theory](https://en.wikipedia.org/wiki/Linear_regression)
- Course Article: `lesson_article.md`

## Support

If you encounter issues:
1. Check all tests pass: `pytest test_lesson.py -v`
2. Verify Python version: `python --version` (should be 3.11+)
3. Review error messages carefully
4. Compare your output with expected results in README

## License

Part of the 180-Day AI/ML Course from Scratch
Educational use only
