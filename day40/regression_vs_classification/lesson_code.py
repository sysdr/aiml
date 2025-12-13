"""
Day 40: Regression vs. Classification
Building house price predictor (regression) and tier classifier (classification)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class HousePriceSystem:
    """
    Dual-purpose ML system demonstrating regression and classification
    
    Real-world parallel: Zillow uses regression to predict exact prices
    and classification to categorize properties into market segments
    """
    
    def __init__(self, n_samples=1000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.regression_model = None
        self.classification_model = None
        self.scaler = StandardScaler()
        
    def generate_data(self):
        """
        Generate synthetic house data
        
        Features mirror real Zillow data:
        - Square footage (800-4000 sq ft)
        - Bedrooms (1-5)
        - Bathrooms (1-4)
        - Age (0-50 years)
        - Location score (1-10)
        """
        np.random.seed(self.random_state)
        
        # Generate features
        sqft = np.random.randint(800, 4001, self.n_samples)
        bedrooms = np.random.randint(1, 6, self.n_samples)
        bathrooms = np.random.randint(1, 5, self.n_samples)
        age = np.random.randint(0, 51, self.n_samples)
        location_score = np.random.uniform(1, 10, self.n_samples)
        
        # Generate price (regression target)
        # Base price formula mirrors real estate pricing
        base_price = (
            sqft * 150 +  # $150 per sq ft
            bedrooms * 25000 +  # $25k per bedroom
            bathrooms * 15000 +  # $15k per bathroom
            location_score * 30000 -  # Location premium
            age * 2000 +  # Depreciation
            np.random.normal(0, 25000, self.n_samples)  # Market variance
        )
        
        # Ensure positive prices
        price = np.maximum(base_price, 50000)
        
        # Generate tier (classification target)
        # Budget: < $200k, Mid: $200k-$400k, Luxury: $400k-$700k, Ultra: > $700k
        tier = pd.cut(
            price,
            bins=[0, 200000, 400000, 700000, np.inf],
            labels=['Budget', 'Mid-Range', 'Luxury', 'Ultra-Luxury']
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'sqft': sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': age,
            'location_score': location_score,
            'price': price,
            'tier': tier
        })
        
        return df
    
    def train_regression_model(self, X_train, y_train):
        """
        Train regression model to predict exact prices
        
        Like Netflix predicting exact star ratings (3.7, 4.2, etc.)
        """
        print("\n" + "="*60)
        print("REGRESSION MODEL: Predicting Exact House Prices")
        print("="*60)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train linear regression
        self.regression_model = LinearRegression()
        self.regression_model.fit(X_train_scaled, y_train)
        
        # Training predictions
        train_pred = self.regression_model.predict(X_train_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_train, train_pred)
        rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        r2 = r2_score(y_train, train_pred)
        
        print(f"\nTraining Metrics:")
        print(f"  Mean Absolute Error: ${mae:,.2f}")
        print(f"  Root Mean Squared Error: ${rmse:,.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"\nInterpretation:")
        print(f"  • On average, predictions are ${mae:,.0f} off")
        print(f"  • Model explains {r2*100:.1f}% of price variations")
        
        return train_pred
    
    def train_classification_model(self, X_train, y_train):
        """
        Train classification model to predict price tiers
        
        Like Gmail classifying emails as spam/not spam
        """
        print("\n" + "="*60)
        print("CLASSIFICATION MODEL: Predicting Price Tiers")
        print("="*60)
        
        # Use already scaled features
        X_train_scaled = self.scaler.transform(X_train)
        
        # Train logistic regression for multi-class
        self.classification_model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=self.random_state
        )
        self.classification_model.fit(X_train_scaled, y_train)
        
        # Training predictions
        train_pred = self.classification_model.predict(X_train_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_train, train_pred)
        
        print(f"\nTraining Accuracy: {accuracy*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_train, train_pred))
        
        return train_pred
    
    def evaluate_models(self, X_test, y_price_test, y_tier_test):
        """
        Evaluate both models on test data
        
        Shows how different metrics apply to different problem types
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION ON TEST DATA")
        print("="*60)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Regression evaluation
        print("\n📊 REGRESSION RESULTS:")
        price_pred = self.regression_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_price_test, price_pred)
        rmse = np.sqrt(mean_squared_error(y_price_test, price_pred))
        r2 = r2_score(y_price_test, price_pred)
        
        print(f"  MAE: ${mae:,.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  R²: {r2:.4f}")
        
        # Classification evaluation
        print("\n🎯 CLASSIFICATION RESULTS:")
        tier_pred = self.classification_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_tier_test, tier_pred)
        print(f"  Accuracy: {accuracy*100:.2f}%")
        
        # Show sample predictions
        print("\n📋 SAMPLE PREDICTIONS:")
        sample_indices = np.random.choice(len(X_test), 5, replace=False)
        
        for idx in sample_indices:
            actual_price = y_price_test.iloc[idx]
            pred_price = price_pred[idx]
            actual_tier = y_tier_test.iloc[idx]
            pred_tier = tier_pred[idx]
            
            print(f"\n  House {idx + 1}:")
            print(f"    Actual Price: ${actual_price:,.0f} | Predicted: ${pred_price:,.0f}")
            print(f"    Actual Tier: {actual_tier} | Predicted: {pred_tier}")
            
        return price_pred, tier_pred
    
    def visualize_comparison(self, y_price_test, price_pred, y_tier_test, tier_pred):
        """
        Create visualization comparing regression and classification
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Regression: Actual vs Predicted
        axes[0].scatter(y_price_test, price_pred, alpha=0.5, s=30)
        axes[0].plot(
            [y_price_test.min(), y_price_test.max()],
            [y_price_test.min(), y_price_test.max()],
            'r--', lw=2, label='Perfect Prediction'
        )
        axes[0].set_xlabel('Actual Price ($)', fontsize=11)
        axes[0].set_ylabel('Predicted Price ($)', fontsize=11)
        axes[0].set_title('Regression: Continuous Value Prediction', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Classification: Confusion Matrix
        cm = confusion_matrix(y_tier_test, tier_pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Budget', 'Mid', 'Luxury', 'Ultra'],
            yticklabels=['Budget', 'Mid', 'Luxury', 'Ultra'],
            ax=axes[1]
        )
        axes[1].set_xlabel('Predicted Tier', fontsize=11)
        axes[1].set_ylabel('Actual Tier', fontsize=11)
        axes[1].set_title('Classification: Category Prediction', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('regression_vs_classification_results.png', dpi=300, bbox_inches='tight')
        print("\n📈 Visualization saved: regression_vs_classification_results.png")


def demonstrate_key_differences():
    """
    Show the fundamental difference between regression and classification
    """
    print("\n" + "="*60)
    print("KEY DIFFERENCES: REGRESSION vs CLASSIFICATION")
    print("="*60)
    
    differences = {
        'Output Type': {
            'Regression': 'Continuous numbers (any value in a range)',
            'Classification': 'Discrete categories (fixed set of options)'
        },
        'Example Output': {
            'Regression': '$347,250.83',
            'Classification': 'Mid-Range'
        },
        'Real-World Use': {
            'Regression': 'Tesla predicting steering angle (0-360°)',
            'Classification': 'Tesla detecting objects (car, pedestrian, bike)'
        },
        'Evaluation': {
            'Regression': 'MAE, RMSE, R² (how far off)',
            'Classification': 'Accuracy, Precision, Recall (how often right)'
        },
        'Output Layer': {
            'Regression': 'Linear activation or none',
            'Classification': 'Softmax activation (probabilities)'
        }
    }
    
    for aspect, comparison in differences.items():
        print(f"\n{aspect}:")
        print(f"  Regression: {comparison['Regression']}")
        print(f"  Classification: {comparison['Classification']}")


def main():
    """
    Main execution: Build and compare regression vs classification
    """
    print("="*60)
    print("DAY 40: REGRESSION VS. CLASSIFICATION")
    print("Building: House Price Predictor (Regression) + Tier Classifier")
    print("="*60)
    
    # Initialize system
    system = HousePriceSystem(n_samples=1000, random_state=42)
    
    # Generate data
    print("\n📊 Generating synthetic house data...")
    df = system.generate_data()
    print(f"Generated {len(df)} houses")
    print("\nSample data:")
    print(df.head())
    
    # Prepare features and targets
    feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
    X = df[feature_cols]
    y_price = df['price']  # Regression target
    y_tier = df['tier']     # Classification target
    
    # Split data
    X_train, X_test, y_price_train, y_price_test, y_tier_train, y_tier_test = train_test_split(
        X, y_price, y_tier, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train)} houses")
    print(f"Test set: {len(X_test)} houses")
    
    # Train regression model
    system.train_regression_model(X_train, y_price_train)
    
    # Train classification model
    system.train_classification_model(X_train, y_tier_train)
    
    # Evaluate both models
    price_pred, tier_pred = system.evaluate_models(
        X_test, y_price_test, y_tier_test
    )
    
    # Visualize comparison
    system.visualize_comparison(y_price_test, price_pred, y_tier_test, tier_pred)
    
    # Show key differences
    demonstrate_key_differences()
    
    print("\n" + "="*60)
    print("✅ LESSON COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  1. Regression predicts continuous values (exact numbers)")
    print("  2. Classification predicts discrete categories")
    print("  3. Same dataset, different questions, different models")
    print("  4. Evaluation metrics match the problem type")
    print("\nNext: Day 41 - Overfitting and Underfitting")
    print("="*60)


if __name__ == "__main__":
    main()
