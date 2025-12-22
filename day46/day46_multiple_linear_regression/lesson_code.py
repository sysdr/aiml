"""
Day 46: Multiple Linear Regression
Building multi-feature prediction systems like Netflix, Google, and Tesla
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import joblib


class MultipleLinearRegressionModel:
    """
    Multiple Linear Regression Model for House Price Prediction
    
    Demonstrates how production systems like Zillow and Redfin use
    multiple features to predict prices accurately.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = [
            'square_feet',
            'bedrooms',
            'bathrooms',
            'age_years',
            'location_score',
            'has_pool',
            'has_garage',
            'distance_to_city_km'
        ]
        self.is_trained = False
        
    def generate_realistic_dataset(self, n_samples: int = 500) -> pd.DataFrame:
        """
        Generate realistic house price dataset with multiple features
        
        Mimics real-world real estate data with correlations and noise
        similar to production datasets at Zillow, Redfin
        """
        np.random.seed(42)
        
        # Generate base features with realistic distributions
        square_feet = np.random.normal(2000, 500, n_samples).clip(800, 5000)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples)
        age_years = np.random.exponential(15, n_samples).clip(0, 50)
        location_score = np.random.uniform(1, 10, n_samples)
        has_pool = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        has_garage = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        distance_to_city_km = np.random.exponential(10, n_samples).clip(1, 50)
        
        # Generate price with realistic feature impacts
        # Base price formula mimics real-world pricing patterns
        base_price = 100000
        price = (
            base_price +
            square_feet * 150 +                    # $150 per sqft
            bedrooms * 25000 +                     # $25k per bedroom
            bathrooms * 15000 +                    # $15k per bathroom
            age_years * -2000 +                    # -$2k per year age
            location_score * 50000 +               # $50k per location point
            has_pool * 30000 +                     # $30k for pool
            has_garage * 20000 +                   # $20k for garage
            distance_to_city_km * -3000 +          # -$3k per km from city
            np.random.normal(0, 30000, n_samples)  # Realistic noise
        )
        
        # Ensure no negative prices
        price = price.clip(50000, None)
        
        # Create DataFrame
        df = pd.DataFrame({
            'square_feet': square_feet,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age_years': age_years,
            'location_score': location_score,
            'has_pool': has_pool,
            'has_garage': has_garage,
            'distance_to_city_km': distance_to_city_km,
            'price': price
        })
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for modeling
        
        Args:
            df: DataFrame with features and price
            
        Returns:
            X: Feature matrix
            y: Target vector (prices)
        """
        X = df[self.feature_names].values
        y = df['price'].values
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train multiple linear regression model
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Training metrics dictionary
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_train)
        
        metrics = {
            'r2_score': r2_score(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae': mean_absolute_error(y_train, y_pred_train)
        }
        
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Evaluation metrics dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted prices
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (coefficients)
        
        Returns:
            DataFrame with feature names and their coefficients
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get coefficients")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        })
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def visualize_coefficients(self, save_path: str = None):
        """
        Visualize feature coefficients
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to visualize coefficients")
        
        importance_df = self.get_feature_importance()
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in importance_df['coefficient']]
        plt.barh(importance_df['feature'], importance_df['coefficient'], color=colors)
        plt.xlabel('Impact on Price ($)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance for House Price Prediction', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: str = None):
        """
        Visualize actual vs predicted prices
        
        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price ($)', fontsize=12)
        plt.ylabel('Predicted Price ($)', fontsize=12)
        plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                           save_path: str = None):
        """
        Visualize residual plot to check model assumptions
        
        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            save_path: Optional path to save figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=30)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Price ($)', fontsize=12)
        axes[0].set_ylabel('Residuals ($)', fontsize=12)
        axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Residual distribution
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals ($)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"✅ Model loaded from {filepath}")


def demonstrate_multiple_linear_regression():
    """
    Complete demonstration of multiple linear regression
    
    Shows the full workflow from data generation to prediction,
    mimicking production ML pipelines at tech companies
    """
    print("=" * 80)
    print("DAY 46: MULTIPLE LINEAR REGRESSION")
    print("Building Multi-Feature Prediction Systems")
    print("=" * 80)
    print()
    
    # Initialize model
    model = MultipleLinearRegressionModel()
    
    # Generate dataset
    print("📊 Generating realistic house price dataset...")
    df = model.generate_realistic_dataset(n_samples=500)
    print(f"✅ Generated {len(df)} house records")
    print()
    
    # Display sample data
    print("Sample of generated data:")
    print(df.head(10))
    print()
    
    # Display dataset statistics
    print("Dataset Statistics:")
    print(df.describe())
    print()
    
    # Prepare data
    print("🔧 Preparing features and target...")
    X, y = model.prepare_data(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print()
    
    # Split data
    print("✂️  Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Train model
    print("🎓 Training multiple linear regression model...")
    train_metrics = model.train(X_train, y_train)
    print("Training Metrics:")
    print(f"  R² Score: {train_metrics['r2_score']:.4f}")
    print(f"  RMSE: ${train_metrics['rmse']:,.2f}")
    print(f"  MAE: ${train_metrics['mae']:,.2f}")
    print()
    
    # Evaluate model
    print("📈 Evaluating model on test data...")
    test_metrics = model.evaluate(X_test, y_test)
    print("Test Metrics:")
    print(f"  R² Score: {test_metrics['r2_score']:.4f}")
    print(f"  RMSE: ${test_metrics['rmse']:,.2f}")
    print(f"  MAE: ${test_metrics['mae']:,.2f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print()
    
    # Display model equation
    print("📐 Learned Model Equation:")
    print(f"Price = ${model.model.intercept_:,.2f}")
    for feature, coef in zip(model.feature_names, model.model.coef_):
        sign = "+" if coef >= 0 else ""
        print(f"        {sign} ${coef:,.2f} × {feature}")
    print()
    
    # Feature importance
    print("⭐ Feature Importance (ranked by absolute impact):")
    importance_df = model.get_feature_importance()
    print(importance_df.to_string(index=False))
    print()
    
    # Make sample predictions
    print("🎯 Sample Predictions:")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for idx in sample_indices:
        actual = y_test[idx]
        predicted = model.predict(X_test[idx].reshape(1, -1))[0]
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100
        
        print(f"House #{idx}:")
        print(f"  Actual Price: ${actual:,.2f}")
        print(f"  Predicted Price: ${predicted:,.2f}")
        print(f"  Error: ${error:,.2f} ({error_pct:.2f}%)")
        print()
    
    # Visualizations
    print("📊 Generating visualizations...")
    y_pred = model.predict(X_test)
    
    model.visualize_coefficients(save_path='feature_importance.png')
    model.visualize_predictions(y_test, y_pred, save_path='predictions.png')
    model.visualize_residuals(y_test, y_pred, save_path='residuals.png')
    print()
    
    # Save model
    print("💾 Saving trained model...")
    model.save_model('house_price_model.joblib')
    print()
    
    # Production insights
    print("=" * 80)
    print("🏭 PRODUCTION AI INSIGHTS")
    print("=" * 80)
    print()
    print("This model demonstrates core patterns used by:")
    print()
    print("• Zillow/Redfin: Price estimation with multiple property features")
    print("• Netflix: Content recommendation with user behavior features")
    print("• Google: Search ranking with 200+ query/page features")
    print("• Tesla: Speed/steering decisions with sensor array features")
    print()
    print("Key Takeaways:")
    print("1. Multiple features → better predictions than single features")
    print("2. Coefficients reveal feature importance automatically")
    print("3. Same API (fit/predict) scales from 1 to 1000+ features")
    print("4. R² > 0.85 indicates strong predictive power")
    print("=" * 80)


if __name__ == "__main__":
    # Set style for visualizations
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Run demonstration
    demonstrate_multiple_linear_regression()
