# Day 75: Model Persistence - Saving and Loading Models

## Overview

Learn production-grade model persistence patterns used by companies like Netflix, Uber, and Stripe. This lesson covers serialization, version management, and hot-swapping for zero-downtime deployments.

## What You'll Build

- **Model Serialization**: Save/load models with compression and validation
- **Version Control**: Track model lineage with metadata and metrics
- **Hot-Swapping Server**: Update models without service restarts

## Quick Start

### 1. Setup Environment (2 minutes)

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run Demo (5 minutes)

```bash
python lesson_code.py
```

Expected output:
```
🎯 Training fraud detection models...
Training Logistic Regression...
  Accuracy: 0.9400
  F1 Score: 0.7234

Training Random Forest...
  Accuracy: 0.9550
  F1 Score: 0.7895

💾 Saving models...
✅ Model saved: models/logistic_regression_v1.pkl (0.12 MB)
✅ Model saved: models/random_forest_v1.pkl (2.34 MB)

📦 Loaded model: random_forest_v1
   Type: RandomForestClassifier
   Saved: 2024-01-15T10:30:45.123456
   ✓ Validation passed
```

### 3. Run Tests (2 minutes)

```bash
python -m pytest test_lesson.py -v
```

Expected: 15 tests passing, demonstrating:
- Serialization integrity
- Compression effectiveness
- Version management
- Hot-swap functionality

## Key Concepts

### 1. Model Serialization

```python
from lesson_code import ModelPersistence

persistence = ModelPersistence(models_dir="models")

# Save with compression and metadata
persistence.save_model(
    model=trained_model,
    model_name='fraud_v1',
    metadata={
        'version': 'v1.0.0',
        'accuracy': 0.94,
        'features': feature_names
    },
    compress=3  # Level 3: balanced size/speed
)

# Load with validation
model, metadata = persistence.load_model('fraud_v1')
```

**Why joblib over pickle?**
- 3-5x smaller files for NumPy arrays
- 2x faster serialization
- Better version compatibility

### 2. Version Management

```python
from lesson_code import ModelVersionManager

version_manager = ModelVersionManager(persistence)

# Register versions
version_manager.register_version(
    model_name='fraud_detector',
    version='v1.0.0',
    metrics={'accuracy': 0.94, 'f1': 0.89},
    notes='Initial production model'
)

# Compare versions
comparison = version_manager.compare_versions(
    'v1.0.0', 'v2.0.0', 
    metric='f1'
)
print(f"Improvement: {comparison['improvement']:.3f}")
```

### 3. Hot-Swapping

```python
from lesson_code import ModelServer

# Server automatically reloads on file change
server = ModelServer(model_path='models/fraud_v1.pkl')

# Predictions use latest model version
predictions = server.predict(X)

# Check status
status = server.get_status()
print(f"Requests served: {status['requests_served']}")
```

## Production Patterns

### Pattern 1: Metadata Bundling

Always save models with metadata:
```python
metadata = {
    'version': 'v2.1.0',
    'training_date': '2024-01-15',
    'accuracy': 0.943,
    'features': ['amount', 'age', 'device'],
    'hyperparameters': {'n_estimators': 100}
}
```

**Why?** Debug issues months later, compare versions, audit model performance.

### Pattern 2: Compression Levels

- Level 0: No compression (fastest, largest)
- Level 3: Balanced (70% size reduction, minimal CPU)
- Level 9: Maximum (80% reduction, slower)

**Use Level 3** for production—best tradeoff.

### Pattern 3: Validation on Load

```python
def _validate_model(model, metadata):
    # Check feature count
    assert model.n_features_in_ == metadata['n_features']
    
    # Verify predict method
    assert hasattr(model, 'predict')
    
    # Test prediction shape
    test_pred = model.predict(X_test[:1])
    assert test_pred.shape[0] == 1
```

Catches corrupted files, version mismatches, incompatible models.

## File Structure

```
day-75-model-persistence/
├── setup.sh                 # Environment setup
├── lesson_code.py          # Implementation
├── test_lesson.py          # Test suite
├── requirements.txt        # Dependencies
├── README.md              # This file
└── models/                # Saved models (created on run)
    ├── logistic_regression_v1.pkl
    ├── random_forest_v1.pkl
    └── version_history.json
```

## Real-World Applications

### Netflix: 15,000+ Models Daily
- One model per content category per region
- Metadata includes A/B test results
- Hot-swapping for zero-downtime updates

### Uber: Dynamic Pricing
- Models retrain every 15 minutes
- Serve predictions every millisecond
- Background training + atomic swaps

### Stripe: Fraud Detection
- Version every model with confusion matrix
- Compare across time periods
- Rollback on performance degradation

## Common Pitfalls

❌ **Using pickle directly**
- Not version-safe
- Security risks
- Larger files

✅ **Use joblib with compression**
```python
joblib.dump(model, 'model.pkl', compress=3)
```

❌ **Saving models without metadata**
- Can't debug issues later
- Can't compare versions

✅ **Bundle metadata**
```python
joblib.dump({
    'model': model,
    'metadata': {...}
}, 'model.pkl')
```

❌ **No validation on load**
- Corrupt files reach production
- Feature mismatches cause errors

✅ **Always validate**
```python
model, metadata = persistence.load_model('model', validate=True)
```

## Next Steps

### Tomorrow: Day 85 - Introduction to Unsupervised Learning
- Clustering algorithms
- Dimensionality reduction
- Anomaly detection
- Apply persistence to unsupervised models

### Practice Exercise

Take your Day 74 feature engineering pipeline and:
1. Train three different models
2. Save each with comprehensive metadata
3. Compare versions and select the best
4. Build a model server with hot-swapping
5. Write tests validating persistence

## Resources

- **Joblib docs**: https://joblib.readthedocs.io/
- **Scikit-learn model persistence**: https://scikit-learn.org/stable/model_persistence.html
- **ONNX for cross-platform**: https://onnx.ai/

## Success Criteria

✅ Models save/load with <2% file size overhead
✅ Metadata includes all key metrics
✅ Hot-swapping updates without service restart
✅ Tests verify serialization integrity
✅ Compression reduces file size by >60%

---

**Built by**: Day 75 - 180-Day AI/ML Course
**Time to complete**: 2-3 hours
**Prerequisites**: Days 71-74 (Scikit-learn fundamentals)
