# Day 72: Data Preprocessing and Feature Scaling

Production-grade data preprocessing pipeline demonstrating techniques used by Netflix, Spotify, and major tech companies for handling real-world messy data.

## Overview

This lesson covers the critical preprocessing steps that separate production AI systems from prototype code:
- Missing data handling with multiple strategies
- Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Categorical encoding (one-hot, label, frequency, target)
- Complete preprocessing pipelines

## Quick Start

```bash
# Make generation script executable
chmod +x generate_lesson_files.sh

# Generate all lesson files
./generate_lesson_files.sh

# Navigate to lesson directory
cd day_72_preprocessing

# Run setup
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Run main demonstration
python lesson_code.py

# Run tests
pytest test_lesson.py -v
```

## What You'll Learn

1. **Missing Data Strategies**
   - Simple imputation (mean, median, mode)
   - KNN imputation
   - Missing value indicators

2. **Feature Scaling**
   - StandardScaler: Z-score normalization
   - MinMaxScaler: [0,1] range normalization
   - RobustScaler: Outlier-resistant scaling

3. **Categorical Encoding**
   - One-hot encoding for low cardinality
   - Frequency encoding for high cardinality
   - Target encoding with smoothing
   - Label encoding (when appropriate)

4. **Production Patterns**
   - Fit on training, transform on all data
   - Handle unseen categories
   - Save/load preprocessing pipelines
   - Consistent train/test transformations

## Key Files

- `lesson_code.py` - Complete implementation with production patterns
- `test_lesson.py` - 25 comprehensive tests covering edge cases
- `requirements.txt` - All dependencies (scikit-learn 1.4+)
- `setup.sh` - Environment setup automation

## Real-World Applications

**Netflix**: Preprocesses 7M+ listings nightly with consistent scaling strategies
**Spotify**: Handles categorical features with millions of distinct values  
**Stripe**: Processes 1M+ transactions/hour with robust missing data handling
**LinkedIn**: Uses frequency-based bucketing for high-cardinality features

## Common Pitfalls Avoided

✓ Never fit scalers on test data (data leakage)  
✓ Handle unseen categories in production  
✓ Choose scaler based on data distribution  
✓ Save preprocessing artifacts with models  
✓ Test edge cases (all missing, all same value, outliers)

## Testing

Run comprehensive test suite:
```bash
pytest test_lesson.py -v
```

Tests cover:
- Edge cases (all missing, unseen categories)
- Consistent train/test transformations
- All encoding strategies
- Pipeline serialization
- Missing data handling

## Next Steps

Day 73 will formalize these concepts into scikit-learn Pipelines—the production standard for composing preprocessing, feature engineering, and modeling into deployable units.

## Requirements

- Python 3.11+
- scikit-learn 1.4.0
- pandas 2.1.4
- numpy 1.26.3
- pytest 7.4.4

## License

Part of the 180-Day AI/ML Course from Scratch
