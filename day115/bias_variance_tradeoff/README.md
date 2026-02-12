# Day 115: Bias-Variance Tradeoff

Production-grade diagnostic system for analyzing and resolving bias-variance issues in machine learning models.

## Overview

This implementation provides comprehensive tools used by ML teams at companies like Netflix, Google, and Airbnb to diagnose whether models are underfitting (high bias) or overfitting (high variance), and provides actionable recommendations for fixing these issues.

## Features

- **Learning Curve Analysis**: Diagnose bias vs variance by plotting training and validation errors
- **Bootstrap Variance Estimation**: Quantify prediction uncertainty through bootstrap sampling
- **Model Complexity Sweep**: Find optimal model complexity before overfitting begins
- **Cross-Validation Stability**: Measure performance consistency across different data splits
- **Automated Diagnosis**: Get actionable recommendations based on error patterns
- **Production Visualizations**: Generate publication-quality diagnostic plots

## Quick Start

### Installation

```bash
# Run the setup script
chmod +x setup_env.sh
./setup_env.sh

# Activate virtual environment
source venv/bin/activate
```

### Run Main Demo

```bash
python lesson_code.py
```

This will:
1. Generate synthetic datasets
2. Analyze high bias (underfitting) models
3. Analyze high variance (overfitting) models
4. Demonstrate well-balanced models
5. Generate diagnostic visualizations
6. Provide actionable recommendations

### Run Tests

```bash
pytest test_lesson.py -v
```

Expected output: 20 tests passing, covering all diagnostic components.

## Key Concepts

### Bias-Variance Decomposition

Total Error = Bias² + Variance + Irreducible Error

- **Bias**: Systematic error from overly simple models
- **Variance**: Error from memorizing training data noise
- **Tradeoff**: Decreasing one increases the other

### Diagnostic Patterns

**High Bias (Underfitting)**:
- Training error: High
- Validation error: High
- Gap between errors: Small
- Solution: Increase model complexity

**High Variance (Overfitting)**:
- Training error: Low
- Validation error: High
- Gap between errors: Large
- Solution: Regularization or more data

**Well-Balanced**:
- Training error: Reasonable
- Validation error: Close to training error
- Gap between errors: Small
- Solution: Current configuration is good

## Usage Examples

### Basic Diagnosis

```python
from lesson_code import BiasVarianceAnalyzer

analyzer = BiasVarianceAnalyzer()
X, y = analyzer.generate_synthetic_data(n_samples=200)

model = LinearRegression()
diagnosis = analyzer.diagnose_model(model, X, y)

print(f"Issue: {diagnosis['issue']}")
print(f"Recommendations: {diagnosis['recommendations']}")
```

### Learning Curves

```python
curves = analyzer.compute_learning_curves(model, X, y)

# Check for high variance
if curves['val_scores_mean'][-1] > 2 * curves['train_scores_mean'][-1]:
    print("High variance detected - consider regularization")
```

### Model Complexity Analysis

```python
results = analyzer.model_complexity_analysis(X, y, max_degree=10)
optimal_degree = results['degrees'][np.argmin(results['val_errors'])]
print(f"Optimal polynomial degree: {optimal_degree}")
```

### Bootstrap Variance

```python
bootstrap_results = analyzer.bootstrap_variance_analysis(
    model, X_train, y_train, X_test, n_bootstraps=100
)

avg_variance = np.mean(bootstrap_results['variance'])
print(f"Average prediction variance: {avg_variance}")
```

## Production Applications

### Model Selection
- Compare models across complexity spectrum
- Identify optimal point before overfitting
- Guide hyperparameter tuning decisions

### Monitoring
- Track train/val error gap in production
- Detect model drift and concept drift
- Trigger retraining when gaps grow

### AutoML Integration
- Automatically select model complexity
- Apply appropriate regularization
- Optimize ensemble configurations

### A/B Testing
- Predict production performance from offline metrics
- Identify models likely to fail in production
- Guide experimental design

## Generated Outputs

The demo generates four diagnostic plots:

1. **learning_curves_high_bias.png**: Shows plateau in both errors
2. **learning_curves_high_variance.png**: Shows large train/val gap
3. **complexity_analysis.png**: Identifies optimal complexity
4. **bootstrap_variance.png**: Visualizes prediction uncertainty

## Requirements

- Python 3.11+
- NumPy 1.26.4
- Pandas 2.2.1
- Matplotlib 3.8.3
- Seaborn 0.13.2
- Scikit-learn 1.4.1.post1
- Pytest 8.1.1
- SciPy 1.12.0

## Testing

The test suite includes 20 comprehensive tests:

- Data generation validation
- Learning curve computation
- Bootstrap sampling correctness
- Complexity analysis trends
- Cross-validation mechanics
- Diagnostic accuracy
- Visualization rendering
- Integration workflows
- Edge case handling

## Real-World Connections

- **Netflix**: Recommendation model complexity selection
- **Google Photos**: Face recognition generalization
- **Uber**: Demand forecasting confidence intervals
- **Tesla**: Perception model regularization
- **Airbnb**: Pricing model bias-variance balance

## Next Steps

Tomorrow (Day 116) we'll apply this framework to systematic hyperparameter tuning, using bias-variance insights to guide grid search, random search, and Bayesian optimization.

## License

Educational purposes - Part of 180-Day AI/ML Course
