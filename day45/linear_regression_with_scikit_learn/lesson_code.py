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
