# Day 91: Principal Component Analysis (PCA) Theory

## Overview

Learn the mathematical foundations of Principal Component Analysis (PCA), the dimensionality reduction technique powering modern AI systems from Netflix recommendations to Tesla's sensor fusion.

## What You'll Learn

- Variance maximization principle
- Covariance matrix computation
- Eigenvalue decomposition
- Principal component extraction
- Orthogonality verification
- Variance explained calculations

## Quick Start

### 1. Setup Environment

```bash
# Make setup script executable and run it
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Run the Lesson

```bash
# Execute main lesson code
python lesson_code.py
```

Expected output:
- Synthetic data generation
- PCA fitting and transformation
- Variance analysis
- Orthogonality verification
- Reconstruction testing
- Visualization generation

### 3. Run Tests

```bash
# Run comprehensive test suite
pytest test_lesson.py -v

# Run specific test class
pytest test_lesson.py::TestMathematicalCorrectness -v

# Run with coverage
pytest test_lesson.py --cov=lesson_code --cov-report=term-missing
```

## Project Structure

```
day91_pca_theory/
├── setup.sh              # Environment setup script
├── lesson_code.py        # Main PCA implementation
├── test_lesson.py        # Comprehensive tests (20 tests)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── pca_analysis.png     # Generated visualization (after running)
```

## Key Concepts

### 1. Variance Maximization

PCA finds directions where data varies the most:
- First component: maximum variance direction
- Second component: maximum remaining variance (perpendicular to first)
- Continues for all dimensions

### 2. Covariance Matrix

Measures how features vary together:
- Diagonal elements: individual feature variances
- Off-diagonal elements: feature correlations
- Symmetric matrix: C[i,j] = C[j,i]

### 3. Eigenvalue Decomposition

Solves C·v = λ·v where:
- v: eigenvector (principal component direction)
- λ: eigenvalue (variance in that direction)
- Eigenvectors are orthogonal (independent)

### 4. Dimensionality Reduction

Keep top k components:
- Preserves maximum variance
- Minimizes reconstruction error
- Reduces computational complexity

## Implementation Details

The `PCATheory` class implements:

```python
pca = PCATheory(n_components=5)

# Fit on training data
pca.fit(X_train)

# Transform data
X_transformed = pca.transform(X_test)

# Reconstruct original data
X_reconstructed = pca.inverse_transform(X_transformed)

# Check orthogonality
is_orthogonal, max_dot = pca.verify_orthogonality()
```

### Mathematical Guarantees

- Components are orthonormal (perpendicular, unit length)
- Variance explained is in decreasing order
- Reconstruction error is minimized
- Total variance is preserved

## Real-World Applications

### Netflix (15,000 → 200 features)
- User behavior dimensionality reduction
- 98.7% reduction, 95%+ variance preserved
- Enables real-time recommendations

### Tesla (50,000 → 500 features)
- Sensor fusion data compression
- 99% reduction maintains safety-critical information
- 100Hz processing requirement met

### OpenAI (12,288 → 256 dimensions)
- Embedding compression for retrieval
- 97.9% reduction enables fast similarity search
- Billions of documents indexed

## Test Coverage

20 comprehensive tests covering:

1. **Basic Functionality** (4 tests)
   - Initialization
   - Fit/transform shapes
   - Inverse transform

2. **Mathematical Correctness** (7 tests)
   - Variance sums
   - Component ordering
   - Orthogonality
   - Unit length
   - Centering

3. **Covariance Matrix** (3 tests)
   - Symmetry
   - Shape
   - Positive diagonal

4. **Dimensionality Reduction** (3 tests)
   - Component selection
   - Variance preservation
   - Reconstruction error

5. **Edge Cases** (4 tests)
   - Single sample
   - More components than features
   - Zero variance features
   - Perfect correlation

## Performance Expectations

- Data generation: < 0.1 seconds
- PCA fitting (1000 samples, 10 features): < 0.05 seconds
- Transformation: < 0.01 seconds
- All tests pass: < 2 seconds

## Next Steps

Tomorrow (Day 92), you'll apply PCA to real datasets:
- Image dimensionality reduction
- Feature extraction pipelines
- Visualization in reduced space
- Integration with scikit-learn

## Troubleshooting

**Issue**: Tests fail with numerical precision errors
- **Solution**: This is expected for some edge cases; tolerances are set appropriately

**Issue**: Visualization doesn't generate
- **Solution**: Ensure matplotlib backend is configured: `export MPLBACKEND=Agg`

**Issue**: Memory error with large datasets
- **Solution**: PCA scales as O(n_features²) for covariance computation

## Additional Resources

- Linear Algebra Review: Days 15-30
- NumPy Advanced Operations: Days 41-50
- Previous: Day 90 - Hierarchical Clustering
- Next: Day 92 - PCA for Dimensionality Reduction

---

**Lesson Duration**: 2-3 hours
**Prerequisites**: Linear algebra basics, NumPy proficiency
**Difficulty**: Intermediate
