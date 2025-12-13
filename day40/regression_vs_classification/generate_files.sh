#!/bin/bash

# Day 40: Regression vs. Classification - Implementation Package Generator
# This script creates all necessary files for the lesson

echo "🚀 Generating Day 40: Regression vs. Classification lesson files..."

# Create setup.sh
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 40: Regression vs. Classification environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the lesson:"
echo "  python lesson_code.py"
echo ""
echo "To run tests:"
echo "  pytest test_lesson.py -v"
EOF

chmod +x setup.sh

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1.post1
matplotlib==3.8.3
seaborn==0.13.2
pytest==8.1.1
EOF

# Create lesson_code.py
cat > lesson_code.py << 'EOF'
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
EOF

# Create test_lesson.py
cat > test_lesson.py << 'EOF'
"""
Tests for Day 40: Regression vs. Classification
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lesson_code import HousePriceSystem


@pytest.fixture
def house_system():
    """Create HousePriceSystem instance for testing"""
    return HousePriceSystem(n_samples=100, random_state=42)


@pytest.fixture
def sample_data(house_system):
    """Generate sample data for testing"""
    df = house_system.generate_data()
    feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
    X = df[feature_cols]
    y_price = df['price']
    y_tier = df['tier']
    
    X_train, X_test, y_price_train, y_price_test, y_tier_train, y_tier_test = train_test_split(
        X, y_price, y_tier, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_price_train, y_price_test, y_tier_train, y_tier_test


class TestDataGeneration:
    """Test data generation functionality"""
    
    def test_data_shape(self, house_system):
        """Test that generated data has correct shape"""
        df = house_system.generate_data()
        assert len(df) == 100
        assert len(df.columns) == 7  # 5 features + price + tier
    
    def test_feature_ranges(self, house_system):
        """Test that features are in expected ranges"""
        df = house_system.generate_data()
        
        assert df['sqft'].min() >= 800
        assert df['sqft'].max() <= 4000
        assert df['bedrooms'].min() >= 1
        assert df['bedrooms'].max() <= 5
        assert df['bathrooms'].min() >= 1
        assert df['bathrooms'].max() <= 4
        assert df['age'].min() >= 0
        assert df['age'].max() <= 50
        assert df['location_score'].min() >= 1
        assert df['location_score'].max() <= 10
    
    def test_price_positive(self, house_system):
        """Test that all prices are positive"""
        df = house_system.generate_data()
        assert (df['price'] > 0).all()
    
    def test_tier_categories(self, house_system):
        """Test that tiers have correct categories"""
        df = house_system.generate_data()
        expected_tiers = ['Budget', 'Mid-Range', 'Luxury', 'Ultra-Luxury']
        assert set(df['tier'].unique()).issubset(set(expected_tiers))


class TestRegressionModel:
    """Test regression model functionality"""
    
    def test_model_training(self, house_system, sample_data):
        """Test that regression model trains successfully"""
        X_train, _, y_price_train, _, _, _ = sample_data
        
        predictions = house_system.train_regression_model(X_train, y_price_train)
        
        assert house_system.regression_model is not None
        assert len(predictions) == len(X_train)
        assert predictions.shape[0] == X_train.shape[0]
    
    def test_prediction_output_type(self, house_system, sample_data):
        """Test that predictions are continuous values"""
        X_train, X_test, y_price_train, _, _, _ = sample_data
        
        house_system.train_regression_model(X_train, y_price_train)
        X_test_scaled = house_system.scaler.transform(X_test)
        predictions = house_system.regression_model.predict(X_test_scaled)
        
        # Regression outputs continuous values
        assert predictions.dtype in [np.float64, np.float32]
        assert len(np.unique(predictions)) > 10  # Many unique values
    
    def test_reasonable_predictions(self, house_system, sample_data):
        """Test that price predictions are reasonable"""
        X_train, X_test, y_price_train, _, _, _ = sample_data
        
        house_system.train_regression_model(X_train, y_price_train)
        X_test_scaled = house_system.scaler.transform(X_test)
        predictions = house_system.regression_model.predict(X_test_scaled)
        
        # Prices should be positive and in reasonable range
        assert (predictions > 0).all()
        assert predictions.min() > 10000  # At least $10k
        assert predictions.max() < 2000000  # Less than $2M


class TestClassificationModel:
    """Test classification model functionality"""
    
    def test_model_training(self, house_system, sample_data):
        """Test that classification model trains successfully"""
        X_train, _, _, _, y_tier_train, _ = sample_data
        
        # Need to train regression first to scale features
        house_system.scaler.fit(X_train)
        predictions = house_system.train_classification_model(X_train, y_tier_train)
        
        assert house_system.classification_model is not None
        assert len(predictions) == len(X_train)
    
    def test_prediction_output_type(self, house_system, sample_data):
        """Test that predictions are discrete categories"""
        X_train, X_test, _, _, y_tier_train, _ = sample_data
        
        house_system.scaler.fit(X_train)
        house_system.train_classification_model(X_train, y_tier_train)
        
        X_test_scaled = house_system.scaler.transform(X_test)
        predictions = house_system.classification_model.predict(X_test_scaled)
        
        # Classification outputs discrete categories
        expected_categories = ['Budget', 'Mid-Range', 'Luxury', 'Ultra-Luxury']
        assert all(pred in expected_categories for pred in predictions)
        assert len(np.unique(predictions)) <= 4  # Limited categories
    
    def test_probability_output(self, house_system, sample_data):
        """Test that model can output probabilities"""
        X_train, X_test, _, _, y_tier_train, _ = sample_data
        
        house_system.scaler.fit(X_train)
        house_system.train_classification_model(X_train, y_tier_train)
        
        X_test_scaled = house_system.scaler.transform(X_test)
        probabilities = house_system.classification_model.predict_proba(X_test_scaled)
        
        # Probabilities should sum to 1 and be valid
        # Number of classes depends on what's in training data (can be 3-4)
        assert probabilities.shape[1] >= 3  # At least 3 classes should be present
        assert probabilities.shape[1] <= 4  # Maximum 4 classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()


class TestModelEvaluation:
    """Test model evaluation functionality"""
    
    def test_evaluation_runs(self, house_system, sample_data):
        """Test that evaluation completes without errors"""
        X_train, X_test, y_price_train, y_price_test, y_tier_train, y_tier_test = sample_data
        
        house_system.train_regression_model(X_train, y_price_train)
        house_system.train_classification_model(X_train, y_tier_train)
        
        price_pred, tier_pred = house_system.evaluate_models(
            X_test, y_price_test, y_tier_test
        )
        
        assert price_pred is not None
        assert tier_pred is not None
        assert len(price_pred) == len(X_test)
        assert len(tier_pred) == len(X_test)
    
    def test_different_output_types(self, house_system, sample_data):
        """Test that regression and classification produce different output types"""
        X_train, X_test, y_price_train, y_price_test, y_tier_train, y_tier_test = sample_data
        
        house_system.train_regression_model(X_train, y_price_train)
        house_system.train_classification_model(X_train, y_tier_train)
        
        price_pred, tier_pred = house_system.evaluate_models(
            X_test, y_price_test, y_tier_test
        )
        
        # Regression: continuous numerical
        assert price_pred.dtype in [np.float64, np.float32]
        
        # Classification: discrete categories
        assert tier_pred.dtype == object or isinstance(tier_pred[0], str)


class TestKeyDifferences:
    """Test understanding of key differences"""
    
    def test_regression_continuous_output(self):
        """Verify regression produces continuous output"""
        # Two very similar houses should have very similar prices
        from lesson_code import HousePriceSystem
        system = HousePriceSystem(n_samples=10, random_state=42)
        
        df = system.generate_data()
        feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
        X = df[feature_cols]
        y_price = df['price']
        
        from sklearn.linear_model import LinearRegression
        system.scaler.fit(X)
        system.regression_model = LinearRegression()
        X_scaled = system.scaler.transform(X)
        system.regression_model.fit(X_scaled, y_price)
        
        predictions = system.regression_model.predict(X_scaled)
        
        # Should have many unique values (continuous)
        unique_ratio = len(np.unique(predictions)) / len(predictions)
        assert unique_ratio > 0.8  # Most predictions are unique
    
    def test_classification_discrete_output(self):
        """Verify classification produces discrete output"""
        from lesson_code import HousePriceSystem
        system = HousePriceSystem(n_samples=10, random_state=42)
        
        df = system.generate_data()
        feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
        X = df[feature_cols]
        y_tier = df['tier']
        
        from sklearn.linear_model import LogisticRegression
        system.scaler.fit(X)
        system.classification_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        X_scaled = system.scaler.transform(X)
        system.classification_model.fit(X_scaled, y_tier)
        
        predictions = system.classification_model.predict(X_scaled)
        
        # Should have limited unique values (discrete)
        unique_count = len(np.unique(predictions))
        assert unique_count <= 4  # Maximum 4 categories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

# Create README.md
cat > README.md << 'EOF'
# Day 40: Regression vs. Classification

## Overview
Learn the fundamental difference between predicting continuous values (regression) and discrete categories (classification) by building a dual-purpose house price system.

## What You'll Build
1. **Regression Model**: Predicts exact house prices ($347,500)
2. **Classification Model**: Predicts price tiers (Budget, Mid-Range, Luxury, Ultra-Luxury)
3. **Comparison Dashboard**: Side-by-side evaluation of both approaches

## Real-World Applications
- **Netflix**: Predicting star ratings (regression) + content categorization (classification)
- **Tesla**: Steering angle prediction (regression) + object detection (classification)
- **Google**: Click-through rate (regression) + query type classification

## Quick Start

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run the Lesson
```bash
python lesson_code.py
```

Expected output:
- Training metrics for both models
- Test set evaluation
- Sample predictions
- Comparison visualization

### 3. Run Tests
```bash
pytest test_lesson.py -v
```

## Key Concepts

### Regression
- **Output**: Continuous numbers (any value in a range)
- **Example**: $347,250.83
- **Metrics**: MAE, RMSE, R²
- **Use case**: When you need the exact number

### Classification
- **Output**: Discrete categories (fixed set of options)
- **Example**: "Mid-Range"
- **Metrics**: Accuracy, Precision, Recall
- **Use case**: When you need the category

### The Fundamental Rule
> Look at the output layer to identify the problem type:
> - Linear/no activation = Regression
> - Softmax activation = Classification

## Files Created
- `lesson_code.py` - Main implementation
- `test_lesson.py` - Comprehensive tests
- `requirements.txt` - Dependencies
- `regression_vs_classification_results.png` - Visualization

## Learning Objectives
After completing this lesson, you'll:
1. ✅ Understand when to use regression vs classification
2. ✅ Know which metrics to use for each problem type
3. ✅ Build and evaluate both model types
4. ✅ Recognize problem types in real AI systems

## Next Steps
**Day 41**: Overfitting and Underfitting
- Learn why models fail on new data
- Detect overfitting in regression vs classification
- Apply production-grade regularization techniques

## Troubleshooting

### Import Error
```bash
pip install -r requirements.txt
```

### Virtual Environment Not Activated
```bash
source venv/bin/activate  # Unix/Mac
venv\Scripts\activate     # Windows
```

### Tests Failing
Ensure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

## Time Estimate
- Setup: 5 minutes
- Code execution: 3 minutes
- Understanding output: 10 minutes
- Experimentation: 15 minutes
- **Total: ~35 minutes**

## Success Criteria
- ✅ Both models train without errors
- ✅ Regression MAE < $50,000
- ✅ Classification accuracy > 70%
- ✅ All tests pass
- ✅ Understand output differences

---

**Course**: 180-Day AI and Machine Learning from Scratch  
**Module**: Core Concepts (Week 7)  
**Day**: 40 of 180
EOF

echo "✅ All files generated successfully!"
echo ""
echo "📁 Generated files:"
echo "  - setup.sh"
echo "  - lesson_code.py"
echo "  - test_lesson.py"
echo "  - requirements.txt"
echo "  - README.md"
echo ""
echo "🚀 To get started:"
echo "  1. chmod +x setup.sh && ./setup.sh"
echo "  2. source venv/bin/activate"
echo "  3. python lesson_code.py"
echo ""
echo "📚 Lesson: Day 40 - Regression vs. Classification"

