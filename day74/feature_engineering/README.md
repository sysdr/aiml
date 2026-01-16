# Day 74: Feature Engineering

## Overview
Learn feature engineering techniques used in production ML systems at Netflix, Tesla, Uber, and Stripe.

## Quick Start

### Setup
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Run Main Implementation
```bash
python lesson_code.py
```

Expected output:
- Feature analysis and transformation
- Polynomial feature creation
- Feature binning demonstration
- Model performance comparison
- Feature selection results

### Run Tests
```bash
pytest test_lesson.py -v
```

Should see 20+ tests passing, covering:
- Scaling strategies (Standard, MinMax, Robust)
- Encoding methods (OneHot, Label)
- Polynomial feature creation
- Feature binning
- Feature selection
- Integration workflows

## Key Concepts

### 1. Feature Scaling
- **StandardScaler**: Mean=0, Std=1 (Spotify's audio features)
- **MinMaxScaler**: Values in [0,1] (Amazon's rating systems)
- **RobustScaler**: Handles outliers (Stripe's fraud detection)

### 2. Categorical Encoding
- **OneHotEncoding**: Binary columns per category (Netflix genres)
- **LabelEncoding**: Integer mapping (Uber's trip priority)
- **TargetEncoding**: Mean target value (PayPal's fraud rates)

### 3. Polynomial Features
- Create interaction terms: x₁ × x₂
- Higher-degree terms: x², x³
- Used by Tesla for trajectory prediction

### 4. Feature Binning
- Discretize continuous variables
- Insurance companies: age brackets
- E-commerce: price ranges

### 5. Feature Selection
- Reduce dimensionality
- Prevent overfitting
- Faster model training

## Production Patterns

### Data Leakage Prevention
```python
# WRONG - leaks test information
scaler.fit(X_all)

# RIGHT - fit only on training
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Pipeline Integration
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('selector', SelectKBest(k=10)),
    ('model', LogisticRegression())
])
```

## Real-World Examples

**Netflix**: 1000+ engineered features per user-video pair
- Watch completion rate
- Genre preference scores
- Time-of-day patterns

**Tesla**: Sensor features for Autopilot
- Lane curvature rate of change
- Relative velocity closing rate
- Time-to-collision estimates

**Stripe**: Fraud detection features
- Transaction amount / 30-day average
- Device fingerprint similarity
- Shipping-billing address distance

## Exercises

1. **Feature Creation**: Add time-based features (day of week, is_weekend)
2. **Experiment**: Compare different scaling strategies on your data
3. **Optimization**: Find optimal polynomial degree (2 vs 3 vs 4)
4. **Selection**: Try different feature selection methods

## Common Pitfalls

1. **Feature Explosion**: Degree-3 polynomials with 10 features = 1000+ features
2. **Target Leakage**: Don't use full dataset for target encoding
3. **Forgetting Test Transform**: Feature engineering must match training
4. **Scaling Order**: Always split first, then fit transformers

## Performance Impact

Typical improvements with feature engineering:
- Baseline model: 70% accuracy
- With scaling/encoding: 75% (+5%)
- With polynomial features: 78% (+8%)
- With feature selection: 76% (+6%, faster inference)

## Next: Day 75 - Model Persistence

Learn how to save and load engineered features and models for production deployment.

## Resources
- Scikit-learn documentation: https://scikit-learn.org
- Feature Engineering for ML by Alice Zheng
- Google's ML Crash Course: Feature Engineering

## Dependencies
- Python 3.11+
- scikit-learn 1.4.0
- pandas 2.1.4
- numpy 1.26.3
- pytest 7.4.4
