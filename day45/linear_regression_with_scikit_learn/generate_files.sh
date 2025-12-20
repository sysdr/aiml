#!/bin/bash

# Day 45: Linear Regression with Scikit-learn - Complete Implementation Package Generator
# This script creates all necessary files for the lesson

set -e  # Exit on any error

echo "========================================"
echo "Day 45: Linear Regression with Scikit-learn"
echo "Generating lesson files..."
echo "========================================"

# Create setup.sh (environment setup script)
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 45: Linear Regression with Scikit-learn"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "Setup complete! ✓"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the lesson code:"
echo "  python lesson_code.py"
echo ""
echo "To run tests:"
echo "  pytest test_lesson.py -v"
EOF

chmod +x setup.sh

# Create requirements.txt
cat > requirements.txt << 'EOF'
scikit-learn==1.5.2
pandas==2.2.3
numpy==1.26.4
matplotlib==3.9.2
pytest==8.3.3
joblib==1.3.2
EOF

# Create lesson_code.py
cat > lesson_code.py << 'EOF'
"""
Day 45: Linear Regression with Scikit-learn
A production-ready implementation of linear regression for salary prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def create_sample_data():
    """
    Generate sample salary data
    In production, this would come from databases or APIs
    """
    np.random.seed(42)
    years_experience = np.array([1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 
                                  3.2, 3.7, 3.9, 4.0, 4.0, 4.1, 4.5, 4.9,
                                  5.1, 5.3, 5.9, 6.0, 6.8, 7.1, 7.9, 8.2,
                                  8.7, 9.0, 9.5, 9.6, 10.3, 10.5])
    
    # Base salary with realistic noise
    base_salary = 30000 + years_experience * 9500
    noise = np.random.normal(0, 3000, len(years_experience))
    salary = base_salary + noise
    
    df = pd.DataFrame({
        'YearsExperience': years_experience,
        'Salary': salary
    })
    
    return df


def explore_data(df):
    """
    Exploratory Data Analysis
    Production systems always start with EDA
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"\nDataset Shape: {df.shape}")
    print(f"Samples: {len(df)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nStatistical Summary:")
    print(df.describe())
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nData Types:")
    print(df.dtypes)
    

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepare data for modeling
    Split into train/test sets for honest evaluation
    """
    X = df[['YearsExperience']].values
    y = df['Salary'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    print(f"Training samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"Testing samples: {len(X_test)} ({test_size*100:.0f}%)")
    print(f"\nFeature shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train linear regression model
    This is the core of scikit-learn's power: 3 lines for a complete model
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Model trained successfully! ✓")
    print(f"\nLearned Parameters:")
    print(f"  Coefficient (slope): ${model.coef_[0]:,.2f} per year")
    print(f"  Intercept (base salary): ${model.intercept_:,.2f}")
    print(f"\nEquation: Salary = ${model.intercept_:,.2f} + ${model.coef_[0]:,.2f} × Years")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on both train and test sets
    Production systems track multiple metrics
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Training set predictions
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test set predictions
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTraining Set Performance:")
    print(f"  MSE:  ${train_mse:,.2f}")
    print(f"  RMSE: ${train_rmse:,.2f}")
    print(f"  MAE:  ${train_mae:,.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print("\nTest Set Performance:")
    print(f"  MSE:  ${test_mse:,.2f}")
    print(f"  RMSE: ${test_rmse:,.2f}")
    print(f"  MAE:  ${test_mae:,.2f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    if test_r2 > 0.8:
        print(f"  ✓ Excellent fit! Model explains {test_r2*100:.1f}% of variance")
    elif test_r2 > 0.6:
        print(f"  ✓ Good fit. Model explains {test_r2*100:.1f}% of variance")
    else:
        print(f"  ⚠ Moderate fit. Model explains {test_r2*100:.1f}% of variance")
    
    print(f"  Average prediction error: ±${test_rmse:,.2f}")
    
    # Check for overfitting
    if train_r2 - test_r2 > 0.1:
        print("  ⚠ Warning: Possible overfitting detected (train R² >> test R²)")
    else:
        print("  ✓ No significant overfitting detected")
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'y_test_pred': y_test_pred
    }


def make_predictions(model, experience_years):
    """
    Make predictions for new data
    This is how production systems serve predictions via API
    """
    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)
    
    if not isinstance(experience_years, list):
        experience_years = [experience_years]
    
    X_new = np.array(experience_years).reshape(-1, 1)
    predictions = model.predict(X_new)
    
    for years, salary in zip(experience_years, predictions):
        print(f"  {years:.1f} years experience → ${salary:,.2f} predicted salary")
    
    return predictions


def visualize_results(X_train, X_test, y_train, y_test, model):
    """
    Visualize model fit and predictions
    Production teams always visualize before deployment
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Training and Test Data with Regression Line
    ax1 = axes[0]
    ax1.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data', s=100)
    ax1.scatter(X_test, y_test, color='green', alpha=0.6, label='Test Data', s=100)
    
    # Plot regression line
    X_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_range = model.predict(X_range)
    ax1.plot(X_range, y_range, color='red', linewidth=2, label='Regression Line')
    
    ax1.set_xlabel('Years of Experience', fontsize=12)
    ax1.set_ylabel('Salary ($)', fontsize=12)
    ax1.set_title('Linear Regression: Salary vs Experience', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residual Plot
    ax2 = axes[1]
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    ax2.scatter(y_pred, residuals, color='purple', alpha=0.6, s=100)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Salary ($)', fontsize=12)
    ax2.set_ylabel('Residuals ($)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'regression_analysis.png'")
    
    # Display plot
    plt.show()


def main():
    """
    Main execution pipeline
    This follows production ML workflow: Data → Train → Evaluate → Deploy
    """
    print("\n" + "=" * 60)
    print("DAY 45: LINEAR REGRESSION WITH SCIKIT-LEARN")
    print("Production-Ready Salary Prediction Model")
    print("=" * 60)
    
    # Step 1: Create/Load Data
    df = create_sample_data()
    df.to_csv('salary_data.csv', index=False)
    print("✓ Sample data created and saved")
    
    # Step 2: Explore Data
    explore_data(df)
    
    # Step 3: Prepare Data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Step 4: Train Model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate Model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Step 6: Make New Predictions
    new_predictions = make_predictions(model, [3.5, 7.0, 10.5])
    
    # Step 7: Visualize Results
    visualize_results(X_train, X_test, y_train, y_test, model)
    
    # Step 8: Save Model (for production deployment)
    import joblib
    joblib.dump(model, 'salary_model.pkl')
    print("\n✓ Model saved as 'salary_model.pkl'")
    
    print("\n" + "=" * 60)
    print("LESSON COMPLETE! ✓")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Scikit-learn provides standardized fit/predict interface")
    print("  2. Linear regression learns optimal coefficients automatically")
    print("  3. Train/test split ensures honest model evaluation")
    print(f"  4. Your model achieved R² = {metrics['test_r2']:.4f} on test data")
    print("  5. This exact pattern scales to production systems")
    print("\nNext: Day 46 - Multiple Linear Regression (multiple features)")


if __name__ == "__main__":
    main()
EOF

# Create test_lesson.py
cat > test_lesson.py << 'EOF'
"""
Day 45: Linear Regression with Scikit-learn - Test Suite
Comprehensive tests to verify understanding and implementation
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys


class TestDataPreparation:
    """Test data creation and preparation"""
    
    def test_data_creation(self):
        """Verify sample data can be created"""
        from lesson_code import create_sample_data
        df = create_sample_data()
        
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) > 0, "Should have data"
        assert 'YearsExperience' in df.columns, "Should have YearsExperience column"
        assert 'Salary' in df.columns, "Should have Salary column"
        
    def test_data_shape(self):
        """Verify data has correct shape"""
        from lesson_code import create_sample_data
        df = create_sample_data()
        
        assert df.shape[1] == 2, "Should have 2 columns"
        assert df.shape[0] >= 20, "Should have at least 20 samples"
    
    def test_no_missing_values(self):
        """Verify no missing values in data"""
        from lesson_code import create_sample_data
        df = create_sample_data()
        
        assert df.isnull().sum().sum() == 0, "Should have no missing values"
    
    def test_train_test_split(self):
        """Verify train/test split works correctly"""
        from lesson_code import create_sample_data, prepare_data
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2)
        
        total_samples = len(df)
        expected_test = int(total_samples * 0.2)
        
        assert len(X_test) == expected_test, f"Test set should have {expected_test} samples"
        assert len(X_train) == total_samples - expected_test, "Train set should have remaining samples"


class TestModelTraining:
    """Test model creation and training"""
    
    def test_model_creation(self):
        """Verify LinearRegression model can be created"""
        model = LinearRegression()
        assert model is not None, "Model should be created"
        
    def test_model_training(self):
        """Verify model can be trained"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        assert hasattr(model, 'coef_'), "Model should have coefficients after fitting"
        assert hasattr(model, 'intercept_'), "Model should have intercept after fitting"
        
    def test_learned_parameters(self):
        """Verify learned parameters are reasonable"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        # Coefficient should be positive (more experience → higher salary)
        assert model.coef_[0] > 0, "Coefficient should be positive"
        
        # Intercept should be reasonable base salary
        assert model.intercept_ > 20000, "Intercept should represent reasonable base salary"
        assert model.intercept_ < 50000, "Intercept should be realistic"


class TestModelPredictions:
    """Test model predictions and accuracy"""
    
    def test_predictions_shape(self):
        """Verify predictions have correct shape"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test), "Should predict for all test samples"
        
    def test_predictions_are_numeric(self):
        """Verify predictions are valid numbers"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert np.all(np.isfinite(predictions)), "All predictions should be finite numbers"
        assert np.all(predictions > 0), "Salary predictions should be positive"
        
    def test_r2_score_quality(self):
        """Verify model achieves good R² score"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        assert r2 > 0.6, f"R² score ({r2:.4f}) should be > 0.6 for reasonable fit"
        
    def test_no_overfitting(self):
        """Verify model doesn't overfit badly"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        
        gap = train_r2 - test_r2
        assert gap < 0.15, f"R² gap ({gap:.4f}) should be < 0.15 to avoid overfitting"


class TestNewPredictions:
    """Test predictions on new data"""
    
    def test_single_prediction(self):
        """Verify can predict for single new value"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        new_X = np.array([[5.0]])
        prediction = model.predict(new_X)
        
        assert len(prediction) == 1, "Should return single prediction"
        assert prediction[0] > 0, "Prediction should be positive"
        
    def test_multiple_predictions(self):
        """Verify can predict for multiple new values"""
        from lesson_code import create_sample_data, prepare_data, train_model, make_predictions
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        predictions = make_predictions(model, [3.0, 5.0, 7.0])
        
        assert len(predictions) == 3, "Should return 3 predictions"
        assert all(p > 0 for p in predictions), "All predictions should be positive"
        
    def test_prediction_ordering(self):
        """Verify more experience predicts higher salary"""
        from lesson_code import create_sample_data, prepare_data, train_model, make_predictions
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        predictions = make_predictions(model, [2.0, 5.0, 8.0])
        
        assert predictions[0] < predictions[1] < predictions[2], \
            "Predictions should increase with experience"


class TestMetrics:
    """Test evaluation metrics"""
    
    def test_mse_calculation(self):
        """Verify MSE can be calculated"""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        mse = mean_squared_error(y_true, y_pred)
        expected = ((10**2) + (10**2) + (10**2)) / 3
        
        assert abs(mse - expected) < 0.01, "MSE calculation should be correct"
        
    def test_r2_range(self):
        """Verify R² is in valid range"""
        from lesson_code import create_sample_data, prepare_data, train_model, evaluate_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        assert -1 <= metrics['test_r2'] <= 1, "R² should be between -1 and 1"


class TestFileOperations:
    """Test file creation and saving"""
    
    def test_data_file_created(self):
        """Verify CSV file is created"""
        from lesson_code import create_sample_data
        df = create_sample_data()
        df.to_csv('test_salary_data.csv', index=False)
        
        assert os.path.exists('test_salary_data.csv'), "CSV file should be created"
        
        # Cleanup
        os.remove('test_salary_data.csv')
        
    def test_model_can_be_saved(self):
        """Verify model can be saved with joblib"""
        import joblib
        from lesson_code import create_sample_data, prepare_data, train_model
        
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        joblib.dump(model, 'test_model.pkl')
        assert os.path.exists('test_model.pkl'), "Model file should be created"
        
        # Verify can be loaded
        loaded_model = joblib.load('test_model.pkl')
        assert loaded_model.coef_[0] == model.coef_[0], "Loaded model should match original"
        
        # Cleanup
        os.remove('test_model.pkl')


class TestProductionReadiness:
    """Test production-ready features"""
    
    def test_reproducibility(self):
        """Verify results are reproducible with random_state"""
        from lesson_code import create_sample_data, prepare_data, train_model
        
        # Run 1
        df1 = create_sample_data()
        X_train1, X_test1, y_train1, y_test1 = prepare_data(df1, random_state=42)
        model1 = train_model(X_train1, y_train1)
        
        # Run 2
        df2 = create_sample_data()
        X_train2, X_test2, y_train2, y_test2 = prepare_data(df2, random_state=42)
        model2 = train_model(X_train2, y_train2)
        
        assert np.allclose(model1.coef_, model2.coef_), "Results should be reproducible"
        assert np.allclose(model1.intercept_, model2.intercept_), "Results should be reproducible"
        
    def test_handles_edge_cases(self):
        """Verify model handles edge case inputs"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        # Test with zero experience
        pred_zero = model.predict([[0]])
        assert pred_zero[0] > 0, "Should handle zero experience"
        
        # Test with high experience
        pred_high = model.predict([[20]])
        assert pred_high[0] > pred_zero[0], "Should handle high experience values"


def run_tests():
    """Run all tests with detailed output"""
    print("=" * 60)
    print("Running Day 45 Test Suite")
    print("=" * 60)
    
    pytest_args = [
        __file__,
        '-v',  # Verbose
        '--tb=short',  # Short traceback format
        '--color=yes'  # Colored output
    ]
    
    result = pytest.main(pytest_args)
    
    if result == 0:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nYou've successfully mastered:")
        print("  ✓ Data preparation and train/test splitting")
        print("  ✓ Training scikit-learn linear regression models")
        print("  ✓ Making predictions and evaluating accuracy")
        print("  ✓ Production-ready implementation patterns")
        print("\nReady for Day 46: Multiple Linear Regression!")
    
    return result


if __name__ == "__main__":
    run_tests()
EOF

# Create README.md
cat > README.md << 'EOF'
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
EOF

echo ""
echo "========================================"
echo "✓ All files generated successfully!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  - setup.sh (environment setup)"
echo "  - lesson_code.py (main implementation)"
echo "  - test_lesson.py (test suite)"
echo "  - requirements.txt (dependencies)"
echo "  - README.md (documentation)"
echo ""
echo "Next steps:"
echo "  1. Run: chmod +x setup.sh && ./setup.sh"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: python lesson_code.py"
echo "  4. Run: pytest test_lesson.py -v"
echo ""
echo "Target: R² > 0.80 on test data"
echo "========================================"

