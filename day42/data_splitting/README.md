# Day 42: Data Splitting (Train/Test/Validation)

## Overview

Learn production-grade data splitting strategies used by Netflix, Tesla, Google, and Meta. Understand why proper data splitting is critical for preventing model failures in production.

## What You'll Learn

- **Basic Splitting**: Standard 70-15-15 train/validation/test splits
- **Stratified Splitting**: Preserving class distributions in imbalanced datasets
- **Time-Series Splitting**: Temporal data splitting without leakage
- **Cross-Validation**: K-fold strategies for robust evaluation
- **Data Leakage Prevention**: Why splitting order matters

## Quick Start

### Option 1: Local Setup (Recommended)

```bash
# Run setup
./setup.sh

# Activate environment
source venv/bin/activate

# Run lesson
python lesson_code.py

# Run tests
pytest test_lesson.py -v
```

### Option 2: Docker Setup

```bash
# Build image
docker build -t day42-data-splitting .

# Run lesson
docker run --rm day42-data-splitting python lesson_code.py

# Run tests
docker run --rm day42-data-splitting pytest test_lesson.py -v
```

## Expected Output

The lesson will demonstrate:

1. **Basic splitting** with size verification
2. **Stratified splitting** showing class distribution preservation
3. **Time-series splitting** with temporal ordering
4. **K-fold cross-validation** coverage
5. **Data leakage** comparison (wrong vs. correct approach)
6. **Full ML pipeline** with hyperparameter tuning
7. **Visualizations** saved as PNG files

## Key Files

- `lesson_code.py` - Complete implementation with all splitting strategies
- `test_lesson.py` - 15+ tests verifying correctness
- `requirements.txt` - Python dependencies
- `setup.sh` - Environment setup script
- `data_splitting_visualization.png` - Generated visual (after running)
- `Dockerfile` - Container image definition
- `.gitignore` - Git ignore rules

## Production Patterns Demonstrated

### Netflix Pattern
- Separate data lakes for train/val/test
- A/B testing on 1-5% of users (test set)
- Historical data for training, recent interactions for validation

### Tesla Autopilot Pattern
- Stratified splits ensuring all scenarios represented
- Geographic and temporal splits for robust evaluation
- Edge case validation before production deployment

### Google Search Pattern
- 1% traffic for validation before wider rollout
- Time-series splits respecting temporal ordering
- Multiple metrics evaluation on held-out data

### Meta Low-Resource Pattern
- K-fold cross-validation for limited data
- Stratified sampling for rare languages
- Robust performance estimation across folds

## Common Pitfalls Avoided

1. **Data Leakage**: Demonstrated wrong vs. right preprocessing order
2. **Temporal Leakage**: Time-series splits maintain proper ordering
3. **Test Set Contamination**: Touch test set only once
4. **Class Imbalance**: Stratified splits preserve distributions

## Testing

Run the comprehensive test suite:

```bash
pytest test_lesson.py -v
```

Tests cover:
- Split size verification
- No data overlap between sets
- Stratification accuracy
- Temporal ordering in time-series
- K-fold coverage
- Reproducibility
- Edge cases

## Next Steps

Tomorrow (Day 43): **Model Evaluation Metrics**
- Accuracy, Precision, Recall, F1-Score
- When to use which metric
- Production monitoring strategies

## Troubleshooting

**Import errors**: Ensure virtual environment is activated
```bash
source venv/bin/activate
```

**Permission denied**: Make setup script executable
```bash
chmod +x setup.sh
```

**Tests failing**: Verify all dependencies installed
```bash
pip install -r requirements.txt
```

## Time Estimate

- Setup: 5 minutes
- Running lesson: 10 minutes
- Understanding output: 15 minutes
- Running tests: 5 minutes
- **Total: ~35 minutes**

## Additional Resources

- Scikit-learn splitting docs: https://scikit-learn.org/stable/modules/cross_validation.html
- Production ML best practices: https://developers.google.com/machine-learning/guides
- Time-series validation: https://otexts.com/fpp3/

---

**Course**: 180-Day AI/ML from Scratch
**Module**: Foundational Skills
**Week**: 7 - Core Concepts
**Day**: 42 of 180
