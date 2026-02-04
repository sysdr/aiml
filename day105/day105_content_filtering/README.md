# Day 105: Content-Based Filtering

Production-grade content-based recommendation system using TF-IDF and cosine similarity.

## Overview

This lesson implements a scalable content-based filtering engine that:
- Extracts features using TF-IDF with configurable parameters
- Computes item similarities using cosine distance
- Applies business logic overlays (popularity boosting, diversity filtering)
- Supports incremental updates for new items

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

1. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

### Running the Demo

Execute the main lesson code:
```bash
python lesson_code.py
```

Expected output:
- Sample dataset loaded (8 movies)
- TF-IDF feature extraction
- Similarity matrix computation
- Top 5 recommendations for "The Matrix"
- Feature importance analysis
- Incremental item addition demo
- System performance metrics

### Running Tests

Execute the test suite:
```bash
pytest test_lesson.py -v
```

Expected results:
- 15+ tests covering core functionality
- All tests should pass
- Coverage report showing >90% code coverage

## Architecture

### Component Flow

```
Item Catalog → Feature Extraction → TF-IDF Matrix
                                           ↓
                                    Similarity Matrix
                                           ↓
                Query Item → Similarity Lookup → Business Logic
                                                      ↓
                                               Top-K Recommendations
```

### Key Components

**ContentBasedRecommender**: Main recommendation engine
- `fit()`: Build TF-IDF matrix and similarity index
- `get_recommendations()`: Generate recommendations with scoring
- `add_new_item()`: Incremental catalog updates
- `get_feature_importance()`: Feature analysis

## Usage Examples

### Basic Recommendations

```python
from lesson_code import ContentBasedRecommender, create_sample_dataset

# Load data
movies_df = create_sample_dataset()

# Initialize and fit
recommender = ContentBasedRecommender()
recommender.fit(movies_df, text_column='combined_features')

# Get recommendations
recs = recommender.get_recommendations(
    item_id='movie_001',
    n_recommendations=5
)

for rec in recs:
    print(f"{rec['item_data']['title']}: {rec['similarity_score']:.3f}")
```

### With Business Logic

```python
# Apply popularity boosting and diversity filtering
recs = recommender.get_recommendations(
    item_id='movie_001',
    n_recommendations=10,
    apply_boost=True,           # Boost popular items
    diversity_threshold=0.85    # Filter overly similar items
)
```

### Feature Analysis

```python
# Get most important features
features = recommender.get_feature_importance('movie_001', top_n=10)
for feature, weight in features:
    print(f"{feature}: {weight:.4f}")
```

### Incremental Updates

```python
# Add new item without refitting
new_movie = {
    'item_id': 'movie_new',
    'title': 'New Release',
    'genres': 'Action Thriller',
    # ... other fields
}

new_features = f"{new_movie['title']} {new_movie['genres']} ..."
recommender.add_new_item(new_movie, new_features)
```

## Testing

### Test Coverage

- Initialization and configuration
- Model fitting and data validation
- Similarity matrix properties
- Recommendation generation
- Diversity filtering
- Popularity boosting
- Incremental updates
- Feature importance
- Error handling

### Running Specific Tests

```bash
# Run specific test class
pytest test_lesson.py::TestContentBasedRecommender -v

# Run specific test
pytest test_lesson.py::TestContentBasedRecommender::test_get_recommendations -v

# With coverage
pytest test_lesson.py --cov=lesson_code --cov-report=html
```

## Performance Considerations

### Scalability

- **TF-IDF Matrix**: Sparse matrix storage (memory efficient)
- **Similarity Computation**: O(n²) for n items, computed offline
- **Incremental Updates**: O(n) per new item
- **Query Time**: O(1) lookup + O(k log k) sorting for top-k

### Optimization Strategies

1. **Approximate Nearest Neighbors**: Use FAISS/Annoy for large catalogs
2. **Precompute Top-K**: Store top-100 neighbors per item
3. **Caching**: Cache frequently requested recommendations
4. **Batch Processing**: Process multiple queries in parallel

## Production Deployment

### Recommended Architecture

```
Offline Pipeline:
- Feature extraction (batch, nightly)
- Similarity matrix computation
- Index building

Online Serving:
- Low-latency similarity lookup (<10ms)
- Business logic application
- Result caching (Redis/Memcached)
```

### Monitoring Metrics

- Average similarity scores
- Recommendation diversity
- Query latency (p50, p95, p99)
- Cache hit rate
- Feature drift detection

## Next Steps

Tomorrow (Days 106-112): Building a complete movie recommender system
- Combine collaborative and content-based filtering
- Build REST API with Flask
- Add persistence layer
- Implement A/B testing framework
- Deploy with monitoring

## Resources

- [TF-IDF Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
- [Recommendation Systems Best Practices](https://developers.google.com/machine-learning/recommendation)

## Troubleshooting

### Common Issues

**Import Errors**: Ensure virtual environment is activated
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**NLTK Data Missing**: Download required data
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**Memory Issues**: Reduce `max_features` or use approximate methods
```python
recommender = ContentBasedRecommender(max_features=1000)
```

## License

This code is part of the 180-Day AI/ML Course educational materials.
