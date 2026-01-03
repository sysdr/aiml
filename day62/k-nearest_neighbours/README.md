# Day 62: K-Nearest Neighbors (KNN) Theory

## Overview
Learn the foundational algorithm behind Netflix recommendations, Spotify playlists, and Amazon product suggestions. This lesson implements KNN from scratch to understand how it works before using scikit-learn tomorrow.

## What You'll Learn
- How KNN makes predictions by finding similar examples
- Euclidean vs Manhattan distance metrics
- Choosing the optimal K parameter
- Real-world applications in production AI systems

## Quick Start

### Setup
```bash
chmod +x setup_env.sh
./setup_env.sh
source venv/bin/activate
```

### Run Lesson
```bash
python lesson_code.py
```

Expected output:
- Distance metrics comparison
- K parameter impact analysis
- Iris flower classification example
- Two visualization plots saved

### Run Tests
```bash
pytest test_lesson.py -v
```

Expected: All 15 tests pass

## Key Concepts

### The Algorithm
1. **Store** all training data (lazy learning)
2. **Calculate** distances to new point
3. **Find** K nearest neighbors
4. **Vote** on prediction (majority class)

### Distance Metrics
- **Euclidean**: Straight-line distance √[(x₂-x₁)² + (y₂-y₁)²]
- **Manhattan**: Grid distance |x₂-x₁| + |y₂-y₁|

### Choosing K
- K too small (K=1): Sensitive to noise
- K too large (K=100): Overly generic
- Sweet spot: K=3 to 15 (odd numbers preferred)

## Real-World Applications

**Netflix**: K=50 users with similar taste → recommend their favorites
**Tesla Autopilot**: K=7 similar sensor readings → classify object
**Amazon Fraud**: K=10 similar transactions → detect anomalies
**Spotify**: K=100 similar listeners → discover new music

## Project Structure
```
.
├── setup.sh              # File generator script
├── setup_env.sh          # Environment setup
├── lesson_code.py        # KNN implementation
├── test_lesson.py        # Test suite
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Implementation Highlights
```python
class KNNClassifier:
    def fit(self, X, y):
        # Store training data
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        # Find K nearest neighbors
        # Take majority vote
        return predictions
```

## Visualizations Generated
1. `knn_k_comparison.png` - Decision boundaries for K=1, 5, 20
2. `knn_iris_classification.png` - Iris flower classification

## Next Steps
Tomorrow (Day 63): KNN with scikit-learn
- Optimized implementations (KD-trees, Ball trees)
- Hyperparameter tuning with cross-validation
- Building a song recommendation system

## Success Criteria
✓ Understand KNN algorithm steps
✓ Explain distance metrics
✓ Choose appropriate K values
✓ Connect theory to production systems

## Common Issues

**Issue**: Slow predictions
**Solution**: Tomorrow we'll learn KD-trees for O(log n) lookups

**Issue**: Ties in voting (even K)
**Solution**: Use odd K values or implement weighted voting

**Issue**: High-dimensional data
**Solution**: Use approximate nearest neighbors or dimensionality reduction

## Resources
- Lesson article: Deep dive into KNN theory
- Visualizations: See algorithm behavior
- Tests: Verify understanding with 15 test cases

Time to complete: 2-3 hours
