# Day 76-84: End-to-End ML Pipeline

Build a complete production-ready machine learning system using the Titanic dataset.

## Quick Start

```bash
# 1. Setup environment
./setup.sh
source venv/bin/activate

# 2. Run the complete pipeline
python lesson_code.py

# 3. Run tests
pytest test_lesson.py -v
```

## What You'll Build

A production-grade ML pipeline with:
- Data validation and quality checks
- Feature engineering and preprocessing
- Model training with cross-validation
- Prediction service simulation
- Model persistence and loading
- Comprehensive testing (20+ tests)

## Project Structure

```
.
├── setup.sh                 # Environment setup
├── requirements.txt         # Python dependencies
├── lesson_code.py          # Complete ML pipeline implementation
├── test_lesson.py          # Comprehensive test suite
├── data/                   # Dataset directory
│   └── titanic_train.csv   # Training data
└── models/                 # Saved models
    └── titanic_model_v1.pkl # Trained model artifact
```

## Pipeline Components

### 1. DataValidator
- Validates required columns
- Checks data types
- Verifies value ranges
- Generates quality reports

### 2. FeatureTransformer
- Imputes missing values
- Scales numerical features
- Encodes categorical variables
- Maintains consistency between training and prediction

### 3. MLPipeline
- Orchestrates complete workflow
- Performs cross-validation
- Tracks comprehensive metrics
- Handles model persistence

## Usage Examples

### Training a Model

```python
from lesson_code import MLPipeline

config = {
    'required_columns': ['Pclass', 'Sex', 'Age', 'Fare', 'Survived'],
    'feature_columns': ['Pclass', 'Sex', 'Age', 'Fare'],
    'target_column': 'Survived',
    'model_params': {'n_estimators': 100, 'max_depth': 10}
}

pipeline = MLPipeline(config)
df, _ = pipeline.load_data('data/titanic_train.csv')
metrics = pipeline.train(df)
pipeline.save_model('models/my_model.pkl')
```

### Making Predictions

```python
# Single prediction
passenger = {
    'Pclass': 1,
    'Sex': 'female',
    'Age': 30.0,
    'Fare': 50.0
}

result = pipeline.predict_single(passenger)
print(f"Survival probability: {result['probability_survived']:.2%}")
```

### Loading Saved Models

```python
loaded_pipeline = MLPipeline.load_model('models/my_model.pkl')
predictions, probabilities = loaded_pipeline.predict(new_data)
```

## Testing

The test suite includes 20+ tests covering:
- Data validation logic
- Feature transformation correctness
- Pipeline integration
- Edge case handling
- Model persistence

```bash
# Run all tests
pytest test_lesson.py -v

# Run specific test class
pytest test_lesson.py::TestDataValidator -v

# Run with coverage
pytest test_lesson.py --cov=lesson_code
```

## Expected Output

```
=== Training ML Pipeline ===
Loaded 891 samples from data/titanic_train.csv
Feature matrix shape: (891, 7)
Train set: 712 samples, Test set: 179 samples

Cross-validation scores (5 folds):
  Fold 1: 0.8239
  Fold 2: 0.8310
  Fold 3: 0.8169
  Fold 4: 0.8239
  Fold 5: 0.8169
  Mean CV accuracy: 0.8225 (+/- 0.0055)

=== Test Set Performance ===
Accuracy:  0.8268
Precision: 0.8182
Recall:    0.7500
F1 Score:  0.7826

Top 5 Important Features:
  Sex: 0.2845
  Fare: 0.2134
  Age: 0.1987
  Pclass: 0.1654
  Parch: 0.0691

Model saved to: models/titanic_model_v1.pkl
```

## Key Concepts Demonstrated

1. **Separation of Concerns**: Distinct components for validation, transformation, training
2. **Configuration Management**: Parameters separated from code
3. **Automated Testing**: Comprehensive test coverage for reliability
4. **Model Artifacts**: Complete packaging of model + transformers + metadata
5. **Production Patterns**: Error handling, logging, reproducibility

## Real-World Applications

This pipeline architecture mirrors systems at:
- **Booking.com**: Hotel recommendation models
- **DoorDash**: Delivery time estimation
- **Robinhood**: Risk assessment models
- **LinkedIn**: Job recommendation system

## Next Steps

- Experiment with different model types (LogisticRegression, GradientBoosting)
- Add feature selection and engineering
- Implement A/B testing framework
- Deploy as REST API using FastAPI
- Add monitoring and alerting

## Troubleshooting

**Issue**: Model accuracy is low
- Check for data leakage in features
- Verify train/test split stratification
- Experiment with hyperparameters

**Issue**: Predictions fail on new data
- Ensure new data has all required columns
- Check for unseen categorical values
- Verify feature distributions match training data

**Issue**: Tests failing
- Ensure virtual environment is activated
- Check Python version (3.11+ required)
- Reinstall dependencies: `pip install -r requirements.txt`

## Resources

- [Scikit-learn Pipeline Documentation](https://scikit-learn.org/stable/modules/compose.html)
- [Model Persistence Best Practices](https://scikit-learn.org/stable/model_persistence.html)
- [Cross-Validation Strategies](https://scikit-learn.org/stable/modules/cross_validation.html)
