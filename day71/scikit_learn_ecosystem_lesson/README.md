# Day 71: The Scikit-learn Ecosystem

A comprehensive lesson on understanding and implementing production-grade ML pipelines using scikit-learn's complete ecosystem.

## What You'll Learn

- The architecture of scikit-learn and its six core modules
- How to build production-grade ML pipelines with preprocessing, feature engineering, and modeling
- Creating custom transformers that integrate seamlessly with sklearn's API
- Deploying complete ML systems as single serializable objects
- Real-world patterns used by Netflix, Spotify, and Uber

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

```bash
# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### Run the Lesson

```bash
# Execute complete demonstration
python lesson_code.py

# Expected output: Full pipeline demo with metrics and serialization
```

### Run Tests

```bash
# Run test suite
pytest test_lesson.py -v

# Expected: All tests pass, verifying dataset, transformers, and pipeline
```

## Project Structure

```
scikit_learn_ecosystem_lesson/
├── lesson_code.py          # Main implementation
├── test_lesson.py          # Test suite
├── requirements.txt        # Dependencies
├── setup.sh                # Environment setup
├── README.md               # This file
└── venv/                   # Virtual environment (created by setup)
```

## Key Components

### 1. MovieRecommendationDataset

Generates synthetic rating data similar to MovieLens:
- 1000 users, 500 movies, 50,000 ratings
- Realistic patterns with latent factors
- User and movie metadata

### 2. UserMovieFeatureTransformer

Custom sklearn transformer demonstrating ecosystem extensibility:
- Fits sklearn's BaseEstimator and TransformerMixin
- Creates interaction features (user × movie patterns)
- Extracts temporal features from timestamps
- Integrates seamlessly into pipelines

### 3. SklearnEcosystemPipeline

Production-grade ML pipeline showcasing:
- Preprocessing: StandardScaler for feature normalization
- Modeling: RandomForestRegressor for predictions
- Cross-validation: 5-fold CV for generalization estimates
- Hyperparameter tuning: GridSearchCV (optional)
- Serialization: joblib for production deployment

## Usage Examples

### Basic Pipeline

```python
from lesson_code import SklearnEcosystemPipeline, MovieRecommendationDataset

# Generate data
dataset = MovieRecommendationDataset()
df = dataset.generate()

# Build and train pipeline
ml_pipeline = SklearnEcosystemPipeline()
X, y, _, _, _ = ml_pipeline.create_features(df)
ml_pipeline.train(X, y)

# Save for production
ml_pipeline.save_pipeline('my_recommender.pkl')
```

### Custom Transformer

```python
from lesson_code import UserMovieFeatureTransformer

# Create and fit transformer
transformer = UserMovieFeatureTransformer()
transformed_data = transformer.fit_transform(df)

# New features include:
# - user_avg_rating: User's average rating tendency
# - movie_avg_rating: Movie's average quality
# - user_movie_interaction: Combined preference signal
# - hour_of_day, day_of_week: Temporal patterns
```

### Production Deployment

```python
import joblib

# Load serialized pipeline
pipeline = joblib.load('production_pipeline.pkl')

# Make predictions (preprocessing happens automatically)
new_ratings = pipeline.predict(new_user_data)
```

## Performance Benchmarks

Expected performance on test set:
- **RMSE**: ~0.84 (within 1 star accuracy)
- **MAE**: ~0.65
- **R²**: ~0.35 (explains 35% of variance)

Training time: ~10-15 seconds on modern CPU

## Real-World Connections

This lesson demonstrates patterns used in production at:

- **Netflix**: Pipeline-based recommendation systems that version control entire models
- **Spotify**: Custom transformers for audio feature extraction integrated with sklearn
- **Uber**: Hyperparameter tuning infrastructure running GridSearchCV at scale
- **Airbnb**: Feature engineering transformers for pricing models
- **Stripe**: Real-time fraud detection using serialized sklearn pipelines

## Next Steps

Tomorrow (Day 72): Deep dive into `sklearn.preprocessing` - the foundation of clean ML pipelines.

## Troubleshooting

**Import errors**: Ensure virtual environment is activated
**Slow training**: Reduce dataset size in MovieRecommendationDataset parameters
**Test failures**: Run `pip install -r requirements.txt` again

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pipeline User Guide](https://scikit-learn.org/stable/modules/compose.html)
- [Custom Transformers](https://scikit-learn.org/stable/developers/develop.html)

---

**Course**: 180-Day AI and Machine Learning from Scratch  
**Module**: Foundational Skills  
**Week**: 12 - Scikit-learn Hands-on  
**Day**: 71
