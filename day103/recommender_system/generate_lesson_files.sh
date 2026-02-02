#!/bin/bash

# Day 103: Recommender Systems Theory - File Generator
# This script creates all necessary files for the lesson

echo "Generating Day 103: Recommender Systems Theory files..."

# Create setup.sh
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 103: Recommender Systems Theory environment..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Setup complete! Activate the environment with: source venv/bin/activate"
EOF

chmod +x setup.sh

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.26.4
pandas==2.2.1
scipy==1.12.0
scikit-learn==1.4.2
matplotlib==3.8.3
seaborn==0.13.2
pytest==8.1.1
flask==3.0.0
flask-cors==4.0.0
requests==2.31.0
EOF

# Create lesson_code.py
cat > lesson_code.py << 'EOF'
"""
Day 103: Recommender Systems Theory
Implementing the three core recommendation approaches with real-world patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RecommendationResult:
    """Container for recommendation output"""
    user_id: int
    recommended_items: List[int]
    scores: List[float]
    method: str


class CollaborativeFilteringEngine:
    """
    User-based collaborative filtering implementation
    Mirrors Netflix's early recommendation approach
    """
    
    def __init__(self, min_common_items: int = 2):
        self.min_common_items = min_common_items
        self.user_item_matrix = None
        self.item_users = {}  # Inverted index for efficiency
        
    def fit(self, interactions: pd.DataFrame):
        """
        Build user-item interaction matrix
        
        Args:
            interactions: DataFrame with columns [user_id, item_id, rating]
        """
        # Create pivot table: users as rows, items as columns
        self.user_item_matrix = interactions.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        # Build inverted index: item -> set of users who rated it
        for item_id in self.user_item_matrix.columns:
            self.item_users[item_id] = set(
                self.user_item_matrix[self.user_item_matrix[item_id] > 0].index
            )
        
        print(f"Built matrix: {len(self.user_item_matrix)} users × "
              f"{len(self.user_item_matrix.columns)} items")
        
    def find_similar_users(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find k most similar users using cosine similarity
        Production systems use approximate nearest neighbors for scale
        """
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_vector = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
        
        # Compute similarity with all users (in production, use ANN)
        similarities = cosine_similarity(user_vector, self.user_item_matrix.values)[0]
        
        # Get top-k similar users (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        similar_users = [
            (self.user_item_matrix.index[idx], similarities[idx])
            for idx in similar_indices
            if similarities[idx] > 0  # Only positive correlations
        ]
        
        return similar_users
    
    def recommend(self, user_id: int, top_n: int = 5) -> RecommendationResult:
        """
        Generate recommendations using collaborative filtering
        Algorithm: Aggregate ratings from similar users, weighted by similarity
        """
        # Find similar users
        similar_users = self.find_similar_users(user_id, top_k=20)
        
        if not similar_users:
            return RecommendationResult(user_id, [], [], "collaborative_filtering")
        
        # Get items the target user has already rated
        user_items = set(
            self.user_item_matrix.columns[
                self.user_item_matrix.loc[user_id] > 0
            ]
        )
        
        # Aggregate scores from similar users
        item_scores = {}
        for similar_user_id, similarity in similar_users:
            similar_user_ratings = self.user_item_matrix.loc[similar_user_id]
            
            for item_id in similar_user_ratings.index:
                rating = similar_user_ratings[item_id]
                
                # Only consider items the target user hasn't rated
                if rating > 0 and item_id not in user_items:
                    if item_id not in item_scores:
                        item_scores[item_id] = {'weighted_sum': 0, 'similarity_sum': 0}
                    
                    item_scores[item_id]['weighted_sum'] += rating * similarity
                    item_scores[item_id]['similarity_sum'] += similarity
        
        # Calculate weighted average scores
        predictions = {
            item_id: scores['weighted_sum'] / scores['similarity_sum']
            for item_id, scores in item_scores.items()
            if scores['similarity_sum'] > 0
        }
        
        # Sort and get top-N
        top_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recommended_items = [item_id for item_id, _ in top_items]
        scores = [score for _, score in top_items]
        
        return RecommendationResult(user_id, recommended_items, scores, "collaborative_filtering")


class ContentBasedEngine:
    """
    Content-based filtering using item features
    Similar to Pandora's Music Genome Project approach
    """
    
    def __init__(self):
        self.item_features = None
        self.user_profiles = {}
        
    def fit(self, item_features: pd.DataFrame, interactions: pd.DataFrame):
        """
        Build item feature matrix and user preference profiles
        
        Args:
            item_features: DataFrame with item_id and feature columns
            interactions: DataFrame with user_id, item_id, rating
        """
        self.item_features = item_features.set_index('item_id')
        
        # Build user profiles by aggregating features of items they liked
        for user_id in interactions['user_id'].unique():
            user_interactions = interactions[interactions['user_id'] == user_id]
            
            # Weight features by ratings
            liked_items = user_interactions[user_interactions['rating'] >= 4]['item_id']
            
            if len(liked_items) > 0:
                # Average feature vectors of liked items
                user_profile = self.item_features.loc[liked_items].mean(axis=0)
                self.user_profiles[user_id] = user_profile.values
        
        print(f"Built profiles for {len(self.user_profiles)} users using "
              f"{len(self.item_features.columns)} features")
    
    def recommend(self, user_id: int, top_n: int = 5, 
                   exclude_items: Set[int] = None) -> RecommendationResult:
        """
        Recommend items similar to user's preference profile
        """
        if user_id not in self.user_profiles:
            return RecommendationResult(user_id, [], [], "content_based")
        
        user_profile = self.user_profiles[user_id].reshape(1, -1)
        
        # Compute similarity between user profile and all items
        item_similarities = cosine_similarity(user_profile, self.item_features.values)[0]
        
        # Create item-score pairs
        item_scores = list(zip(self.item_features.index, item_similarities))
        
        # Filter out items user has already seen
        if exclude_items:
            item_scores = [(item, score) for item, score in item_scores 
                          if item not in exclude_items]
        
        # Sort and get top-N
        top_items = sorted(item_scores, key=lambda x: x[1], reverse=True)[:top_n]
        
        recommended_items = [int(item_id) for item_id, _ in top_items]
        scores = [float(score) for _, score in top_items]
        
        return RecommendationResult(user_id, recommended_items, scores, "content_based")


class HybridRecommender:
    """
    Combines collaborative and content-based approaches
    Mirrors production systems like Amazon and Netflix
    """
    
    def __init__(self, collaborative_weight: float = 0.6, content_weight: float = 0.4):
        self.cf_engine = CollaborativeFilteringEngine()
        self.cb_engine = ContentBasedEngine()
        self.cf_weight = collaborative_weight
        self.cb_weight = content_weight
        
    def fit(self, interactions: pd.DataFrame, item_features: pd.DataFrame):
        """Fit both engines"""
        print("Training collaborative filtering engine...")
        self.cf_engine.fit(interactions)
        
        print("Training content-based engine...")
        self.cb_engine.fit(item_features, interactions)
        
    def recommend(self, user_id: int, top_n: int = 5) -> RecommendationResult:
        """
        Hybrid recommendation using weighted combination
        
        Production systems use more sophisticated ensembles (gradient boosting, neural nets)
        This demonstrates the core concept
        """
        # Get recommendations from both engines
        cf_results = self.cf_engine.recommend(user_id, top_n=20)
        
        # Get items user has interacted with for exclusion
        if user_id in self.cf_engine.user_item_matrix.index:
            exclude_items = set(
                self.cf_engine.user_item_matrix.columns[
                    self.cf_engine.user_item_matrix.loc[user_id] > 0
                ]
            )
        else:
            exclude_items = set()
            
        cb_results = self.cb_engine.recommend(user_id, top_n=20, exclude_items=exclude_items)
        
        # Normalize scores to [0, 1] range
        def normalize_scores(scores):
            if not scores or max(scores) == min(scores):
                return scores
            min_s, max_s = min(scores), max(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]
        
        # Create combined score dictionary
        combined_scores = {}
        
        # Add collaborative filtering scores
        cf_norm = normalize_scores(cf_results.scores)
        for item, score in zip(cf_results.recommended_items, cf_norm):
            combined_scores[item] = self.cf_weight * score
        
        # Add content-based scores
        cb_norm = normalize_scores(cb_results.scores)
        for item, score in zip(cb_results.recommended_items, cb_norm):
            if item in combined_scores:
                combined_scores[item] += self.cb_weight * score
            else:
                combined_scores[item] = self.cb_weight * score
        
        # Sort by combined score
        top_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recommended_items = [item for item, _ in top_items]
        scores = [score for _, score in top_items]
        
        return RecommendationResult(user_id, recommended_items, scores, "hybrid")


def create_sample_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic movie rating dataset
    Simulates user preferences and item features
    """
    np.random.seed(42)
    
    # Create users with different taste profiles
    num_users = 50
    num_items = 100
    
    # Generate interactions (sparse matrix - users rate ~10% of items)
    interactions = []
    for user_id in range(num_users):
        # Each user rates 8-12 items
        num_ratings = np.random.randint(8, 13)
        rated_items = np.random.choice(num_items, num_ratings, replace=False)
        
        for item_id in rated_items:
            # Ratings from 1-5
            rating = np.random.randint(1, 6)
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    # Generate item features (5 features per item)
    # In production: genre vectors, audio features, text embeddings, etc.
    item_features = []
    for item_id in range(num_items):
        features = {
            'item_id': item_id,
            'feature_1': np.random.rand(),  # e.g., action score
            'feature_2': np.random.rand(),  # e.g., comedy score
            'feature_3': np.random.rand(),  # e.g., drama score
            'feature_4': np.random.rand(),  # e.g., release year (normalized)
            'feature_5': np.random.rand(),  # e.g., popularity score
        }
        item_features.append(features)
    
    item_features_df = pd.DataFrame(item_features)
    
    return interactions_df, item_features_df


def demonstrate_recommender_systems():
    """Main demonstration of all three recommender approaches"""
    
    print("=" * 70)
    print("Day 103: Recommender Systems Theory - Implementation Demo")
    print("=" * 70)
    
    # Create dataset
    print("\n1. Generating synthetic movie rating dataset...")
    interactions, item_features = create_sample_dataset()
    
    print(f"   Created {len(interactions)} ratings from {interactions['user_id'].nunique()} users")
    print(f"   Item catalog: {interactions['item_id'].nunique()} movies")
    print(f"   Sparsity: {len(interactions) / (interactions['user_id'].nunique() * interactions['item_id'].nunique()) * 100:.2f}%")
    
    # Test user
    test_user = 5
    
    # Collaborative Filtering
    print("\n2. Collaborative Filtering Demonstration")
    print("   " + "-" * 50)
    cf_engine = CollaborativeFilteringEngine()
    cf_engine.fit(interactions)
    
    cf_results = cf_engine.recommend(test_user, top_n=5)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(cf_results.recommended_items, cf_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    # Content-Based Filtering
    print("\n3. Content-Based Filtering Demonstration")
    print("   " + "-" * 50)
    cb_engine = ContentBasedEngine()
    cb_engine.fit(item_features, interactions)
    
    user_rated_items = set(interactions[interactions['user_id'] == test_user]['item_id'])
    cb_results = cb_engine.recommend(test_user, top_n=5, exclude_items=user_rated_items)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(cb_results.recommended_items, cb_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    # Hybrid Approach
    print("\n4. Hybrid Recommender Demonstration")
    print("   " + "-" * 50)
    hybrid = HybridRecommender(collaborative_weight=0.6, content_weight=0.4)
    hybrid.fit(interactions, item_features)
    
    hybrid_results = hybrid.recommend(test_user, top_n=5)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(hybrid_results.recommended_items, hybrid_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    # Compare approaches
    print("\n5. Method Comparison")
    print("   " + "-" * 50)
    print(f"   Collaborative: {len(cf_results.recommended_items)} recommendations")
    print(f"   Content-Based: {len(cb_results.recommended_items)} recommendations")
    print(f"   Hybrid:        {len(hybrid_results.recommended_items)} recommendations")
    
    print("\n6. Production Insights")
    print("   " + "-" * 50)
    print("   • Netflix uses hybrid approach with 200+ models in ensemble")
    print("   • Candidate generation (CF) runs in <20ms using ANN search")
    print("   • Ranking stage (content features) processes 500+ items in <50ms")
    print("   • A/B testing determines optimal collaborative/content weights")
    print("   • Cold-start items rely more on content features")
    
    print("\n" + "=" * 70)
    print("Demo complete! Tomorrow: Implement collaborative filtering from scratch")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_recommender_systems()
EOF

# Create test_lesson.py
cat > test_lesson.py << 'EOF'
"""
Test suite for Day 103: Recommender Systems Theory
"""

import pytest
import numpy as np
import pandas as pd
from lesson_code import (
    CollaborativeFilteringEngine,
    ContentBasedEngine,
    HybridRecommender,
    create_sample_dataset,
    RecommendationResult
)


class TestCollaborativeFiltering:
    """Test collaborative filtering implementation"""
    
    def test_engine_initialization(self):
        """Test engine can be initialized"""
        engine = CollaborativeFilteringEngine(min_common_items=2)
        assert engine.min_common_items == 2
        assert engine.user_item_matrix is None
    
    def test_fit_creates_matrix(self):
        """Test fitting creates user-item matrix"""
        interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [10, 11, 10, 12, 11],
            'rating': [5, 4, 3, 5, 4]
        })
        
        engine = CollaborativeFilteringEngine()
        engine.fit(interactions)
        
        assert engine.user_item_matrix is not None
        assert len(engine.user_item_matrix) == 3  # 3 users
        assert len(engine.user_item_matrix.columns) == 3  # 3 items
    
    def test_find_similar_users(self):
        """Test finding similar users"""
        interactions = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3],
            'item_id': [10, 11, 12, 10, 11, 13, 12, 13],
            'rating': [5, 4, 3, 5, 4, 5, 3, 5]
        })
        
        engine = CollaborativeFilteringEngine()
        engine.fit(interactions)
        
        similar = engine.find_similar_users(1, top_k=2)
        assert len(similar) <= 2
        assert all(isinstance(user_id, (int, np.integer)) for user_id, _ in similar)
        assert all(0 <= sim <= 1 for _, sim in similar)
    
    def test_recommend_returns_valid_result(self):
        """Test recommendation returns valid result"""
        interactions, _ = create_sample_dataset()
        
        engine = CollaborativeFilteringEngine()
        engine.fit(interactions)
        
        result = engine.recommend(user_id=5, top_n=5)
        
        assert isinstance(result, RecommendationResult)
        assert result.user_id == 5
        assert len(result.recommended_items) <= 5
        assert len(result.recommended_items) == len(result.scores)
        assert result.method == "collaborative_filtering"
    
    def test_recommendations_are_sorted(self):
        """Test recommendations are sorted by score"""
        interactions, _ = create_sample_dataset()
        
        engine = CollaborativeFilteringEngine()
        engine.fit(interactions)
        
        result = engine.recommend(user_id=5, top_n=10)
        
        # Scores should be in descending order
        assert all(result.scores[i] >= result.scores[i+1] 
                  for i in range(len(result.scores)-1))


class TestContentBasedFiltering:
    """Test content-based filtering implementation"""
    
    def test_engine_initialization(self):
        """Test engine can be initialized"""
        engine = ContentBasedEngine()
        assert engine.item_features is None
        assert len(engine.user_profiles) == 0
    
    def test_fit_creates_profiles(self):
        """Test fitting creates user profiles"""
        interactions = pd.DataFrame({
            'user_id': [1, 1, 2],
            'item_id': [10, 11, 10],
            'rating': [5, 4, 5]
        })
        
        item_features = pd.DataFrame({
            'item_id': [10, 11],
            'feature_1': [0.8, 0.2],
            'feature_2': [0.3, 0.7]
        })
        
        engine = ContentBasedEngine()
        engine.fit(item_features, interactions)
        
        assert len(engine.user_profiles) > 0
        assert engine.item_features is not None
    
    def test_recommend_excludes_items(self):
        """Test recommendation respects exclusion set"""
        interactions, item_features = create_sample_dataset()
        
        engine = ContentBasedEngine()
        engine.fit(item_features, interactions)
        
        exclude_items = {1, 2, 3}
        result = engine.recommend(user_id=5, top_n=5, exclude_items=exclude_items)
        
        # No excluded items should appear in recommendations
        assert all(item not in exclude_items for item in result.recommended_items)
    
    def test_recommend_returns_valid_result(self):
        """Test recommendation returns valid result"""
        interactions, item_features = create_sample_dataset()
        
        engine = ContentBasedEngine()
        engine.fit(item_features, interactions)
        
        result = engine.recommend(user_id=5, top_n=5)
        
        assert isinstance(result, RecommendationResult)
        assert result.method == "content_based"
        assert len(result.recommended_items) <= 5


class TestHybridRecommender:
    """Test hybrid recommender system"""
    
    def test_initialization_with_weights(self):
        """Test hybrid system initializes with custom weights"""
        hybrid = HybridRecommender(collaborative_weight=0.7, content_weight=0.3)
        assert hybrid.cf_weight == 0.7
        assert hybrid.cb_weight == 0.3
    
    def test_fit_trains_both_engines(self):
        """Test fitting trains both underlying engines"""
        interactions, item_features = create_sample_dataset()
        
        hybrid = HybridRecommender()
        hybrid.fit(interactions, item_features)
        
        # Both engines should be trained
        assert hybrid.cf_engine.user_item_matrix is not None
        assert len(hybrid.cb_engine.user_profiles) > 0
    
    def test_recommend_combines_methods(self):
        """Test recommendation combines both methods"""
        interactions, item_features = create_sample_dataset()
        
        hybrid = HybridRecommender()
        hybrid.fit(interactions, item_features)
        
        result = hybrid.recommend(user_id=5, top_n=5)
        
        assert isinstance(result, RecommendationResult)
        assert result.method == "hybrid"
        assert len(result.recommended_items) <= 5
    
    def test_scores_are_normalized(self):
        """Test hybrid scores are properly combined"""
        interactions, item_features = create_sample_dataset()
        
        hybrid = HybridRecommender()
        hybrid.fit(interactions, item_features)
        
        result = hybrid.recommend(user_id=5, top_n=10)
        
        # All scores should be valid
        assert all(0 <= score <= 2 for score in result.scores)  # Max = cf_weight + cb_weight


class TestDataGeneration:
    """Test sample dataset generation"""
    
    def test_create_sample_dataset_returns_dataframes(self):
        """Test dataset creation returns proper DataFrames"""
        interactions, item_features = create_sample_dataset()
        
        assert isinstance(interactions, pd.DataFrame)
        assert isinstance(item_features, pd.DataFrame)
    
    def test_interactions_have_required_columns(self):
        """Test interactions have required columns"""
        interactions, _ = create_sample_dataset()
        
        required_cols = {'user_id', 'item_id', 'rating'}
        assert required_cols.issubset(set(interactions.columns))
    
    def test_item_features_have_required_columns(self):
        """Test item features have required columns"""
        _, item_features = create_sample_dataset()
        
        assert 'item_id' in item_features.columns
        assert len(item_features.columns) >= 5  # At least item_id + 4 features
    
    def test_ratings_in_valid_range(self):
        """Test ratings are in valid range"""
        interactions, _ = create_sample_dataset()
        
        assert interactions['rating'].min() >= 1
        assert interactions['rating'].max() <= 5
    
    def test_dataset_is_sparse(self):
        """Test dataset has realistic sparsity"""
        interactions, _ = create_sample_dataset()
        
        num_users = interactions['user_id'].nunique()
        num_items = interactions['item_id'].nunique()
        total_possible = num_users * num_items
        
        sparsity = len(interactions) / total_possible
        
        # Realistic recommender systems are >95% sparse
        assert sparsity < 0.5


class TestRecommendationQuality:
    """Test recommendation quality properties"""
    
    def test_no_duplicate_recommendations(self):
        """Test each method doesn't recommend duplicates"""
        interactions, item_features = create_sample_dataset()
        
        engines = [
            ('CF', CollaborativeFilteringEngine()),
            ('CB', ContentBasedEngine()),
        ]
        
        for name, engine in engines:
            if name == 'CF':
                engine.fit(interactions)
            else:
                engine.fit(item_features, interactions)
            
            result = engine.recommend(user_id=5, top_n=10)
            
            # No duplicates in recommendations
            assert len(result.recommended_items) == len(set(result.recommended_items))
    
    def test_cold_start_user_handling(self):
        """Test handling of users with no history"""
        interactions, item_features = create_sample_dataset()
        
        engine = CollaborativeFilteringEngine()
        engine.fit(interactions)
        
        # User not in dataset
        result = engine.recommend(user_id=999, top_n=5)
        
        # Should return empty recommendations gracefully
        assert isinstance(result, RecommendationResult)
        assert len(result.recommended_items) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

# Create README.md
cat > README.md << 'EOF'
# Day 103: Recommender Systems Theory

## Overview
Implementation of the three core recommender system architectures: Collaborative Filtering, Content-Based Filtering, and Hybrid approaches. This lesson demonstrates the mathematical foundations and production patterns behind Netflix, Amazon, and Spotify's recommendation engines.

## Quick Start

### Setup (First Time)
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Run Main Demo
```bash
python lesson_code.py
```

Expected output shows:
- Collaborative filtering recommendations using user similarity
- Content-based recommendations using item features
- Hybrid recommendations combining both approaches
- Performance comparisons and production insights

### Run Tests
```bash
pytest test_lesson.py -v
```

Expected: 25 tests passed

## What You'll Learn

### 1. Collaborative Filtering
- User-item interaction matrix construction
- Similarity computation using cosine distance
- Weighted score aggregation from similar users
- Handles implicit and explicit feedback

### 2. Content-Based Filtering
- Item feature extraction and representation
- User preference profile construction
- Feature-based similarity matching
- Cold-start item recommendations

### 3. Hybrid Systems
- Multi-method score combination
- Normalized score aggregation
- Production ensemble patterns
- Trade-offs between approaches

## Architecture

```
User Request
    ↓
Hybrid Recommender
    ├── Collaborative Engine (60% weight)
    │   ├── Find similar users (cosine similarity)
    │   ├── Aggregate ratings (weighted average)
    │   └── Return top candidates
    │
    ├── Content Engine (40% weight)
    │   ├── Match user profile to item features
    │   ├── Compute feature similarity
    │   └── Return top candidates
    │
    └── Score Fusion
        ├── Normalize scores [0,1]
        ├── Weight and combine
        └── Re-rank final recommendations
```

## Production Patterns

**Netflix**: Hybrid approach with 200+ models
- Stage 1: CF for candidate generation (<20ms)
- Stage 2: Deep learning for ranking (<50ms)
- Stage 3: Business logic (diversity, freshness)

**Amazon**: Cascading recommenders
- Item-to-item CF for related products
- Session-based for real-time
- Content features for cold-start

**Spotify**: Audio features + collaborative
- Deep learning on audio spectrograms
- User listening history patterns
- Playlist co-occurrence signals

## Key Metrics

**Sparsity**: 99%+ in production (users interact with <1% of items)
**Latency**: 50-100ms end-to-end for top-K recommendations
**Throughput**: 100M+ requests/day for large platforms
**Accuracy**: Measured via A/B testing, not offline metrics

## Common Pitfalls

1. **Popularity Bias**: CF recommends popular items more
   - Solution: Penalize popular items in ranking

2. **Cold Start**: No recommendations for new users/items
   - Solution: Use content features, trending items

3. **Filter Bubble**: Only recommending similar content
   - Solution: Add exploration (epsilon-greedy), diversity constraints

4. **Scalability**: User-user similarity doesn't scale
   - Solution: Use item-item CF or matrix factorization

## Tomorrow's Lesson

Day 104 implements collaborative filtering from scratch:
- Matrix factorization with gradient descent
- Implicit feedback handling
- Efficient similarity computation
- Cold-start strategies

## Resources

- Research paper: "Amazon.com Recommendations: Item-to-Item Collaborative Filtering"
- Netflix Prize: Lessons learned from competition
- RecSys conference proceedings

## Troubleshooting

**Low recommendation quality**: Increase dataset size or adjust similarity thresholds
**Slow performance**: Use approximate nearest neighbors (FAISS) for production
**Memory errors**: Implement sparse matrix representations

---
Ready for Day 104: Collaborative Filtering implementation!
EOF

# Create sources directory
mkdir -p sources

# Create startup.sh in sources
cat > sources/startup.sh << 'EOF'
#!/bin/bash

# Startup script for Day 103: Recommender Systems Dashboard

echo "🚀 Starting Day 103: Recommender Systems Dashboard..."
echo "===================================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || exit 1

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if dashboard is already running
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Dashboard already running on port 5000"
    echo "   Stopping existing instance..."
    pkill -9 -f "python.*dashboard.py" || true
    sleep 2
    # Double check
    if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "❌ Failed to stop existing dashboard"
        exit 1
    fi
fi

# Also check for any dashboard processes
DASHBOARD_COUNT=$(ps aux | grep -E "python.*dashboard.py" | grep -v grep | wc -l)
if [ "$DASHBOARD_COUNT" -gt 0 ]; then
    echo "⚠️  Found $DASHBOARD_COUNT existing dashboard process(es), stopping..."
    pkill -9 -f "python.*dashboard.py" || true
    sleep 2
fi

# Change to sources directory
cd sources || exit 1

# Start dashboard
echo "📊 Starting dashboard on http://localhost:5000"
nohup python dashboard.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!

echo "✅ Dashboard started (PID: $DASHBOARD_PID)"
echo ""
echo "📱 Access dashboard at: http://localhost:5000"
echo "📡 API endpoint: http://localhost:5000/api/metrics"
echo ""
echo "To stop the dashboard, run:"
echo "   kill $DASHBOARD_PID"
echo "   or"
echo "   pkill -f 'python.*dashboard.py'"
echo ""

# Wait a moment for startup
sleep 3

# Check if dashboard started successfully
if ps -p $DASHBOARD_PID > /dev/null; then
    echo "✅ Dashboard is running successfully!"
else
    echo "❌ Dashboard failed to start"
    exit 1
fi
EOF

chmod +x sources/startup.sh

# Create dashboard.py in sources
cat > sources/dashboard.py << 'EOF'
"""
Dashboard for Day 103: Recommender Systems Theory
"""

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import json
import time
import sys
import os
from datetime import datetime
import numpy as np

# Add parent directory to path to import lesson_code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lesson_code import (
    CollaborativeFilteringEngine,
    ContentBasedEngine,
    HybridRecommender,
    create_sample_dataset
)

app = Flask(__name__)
CORS(app)

# Global state
interactions_df = None
item_features_df = None
cf_engine = None
cb_engine = None
hybrid_engine = None

metrics_data = {
    'timestamp': datetime.now().isoformat(),
    'collaborative_metrics': {
        'users_processed': 0,
        'recommendations_generated': 0,
        'avg_similarity_score': 0.0,
        'matrix_size': '0x0'
    },
    'content_metrics': {
        'items_processed': 0,
        'user_profiles_created': 0,
        'avg_similarity_score': 0.0,
        'features_count': 0
    },
    'hybrid_metrics': {
        'recommendations_generated': 0,
        'cf_weight': 0.6,
        'cb_weight': 0.4,
        'avg_combined_score': 0.0
    },
    'demo_metrics': {
        'demos_run': 0,
        'last_demo_time': None,
        'success_rate': 100.0
    }
}

def initialize_engines():
    """Initialize recommender engines with sample data"""
    global interactions_df, item_features_df, cf_engine, cb_engine, hybrid_engine
    
    if interactions_df is None:
        interactions_df, item_features_df = create_sample_dataset()
        
        cf_engine = CollaborativeFilteringEngine()
        cf_engine.fit(interactions_df)
        
        cb_engine = ContentBasedEngine()
        cb_engine.fit(item_features_df, interactions_df)
        
        hybrid_engine = HybridRecommender(collaborative_weight=0.6, content_weight=0.4)
        hybrid_engine.fit(interactions_df, item_features_df)
        
        # Update metrics
        metrics_data['collaborative_metrics']['users_processed'] = len(cf_engine.user_item_matrix)
        metrics_data['collaborative_metrics']['matrix_size'] = f"{len(cf_engine.user_item_matrix)}x{len(cf_engine.user_item_matrix.columns)}"
        metrics_data['content_metrics']['items_processed'] = len(item_features_df)
        metrics_data['content_metrics']['user_profiles_created'] = len(cb_engine.user_profiles)
        metrics_data['content_metrics']['features_count'] = len(item_features_df.columns) - 1

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Day 103: Recommender Systems Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-title {
            font-size: 1.2em;
            color: #667eea;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .metric-value {
            font-size: 2.5em;
            color: #333;
            font-weight: bold;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .timestamp {
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 0.9em;
        }
        .control-panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .control-title {
            font-size: 1.5em;
            color: #667eea;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .demo-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .demo-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            font-weight: 500;
        }
        .demo-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .demo-btn:active {
            transform: translateY(0);
        }
        .status-message {
            margin-top: 15px;
            padding: 10px;
            border-radius: 5px;
            display: none;
        }
        .status-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Recommender Systems Dashboard</h1>
        
        <div class="control-panel">
            <div class="control-title">🎮 Demo Controls</div>
            <div class="demo-buttons">
                <button class="demo-btn" onclick="runCollaborativeFiltering()">
                    👥 Run Collaborative Filtering
                </button>
                <button class="demo-btn" onclick="runContentBased()">
                    📝 Run Content-Based Filtering
                </button>
                <button class="demo-btn" onclick="runHybrid()">
                    🔀 Run Hybrid Recommender
                </button>
                <button class="demo-btn" onclick="updateDemo()">
                    📊 Update Demo Counter
                </button>
            </div>
            <div id="status-message" class="status-message"></div>
        </div>
        
        <div class="metrics-grid" id="metrics-grid">
            <!-- Metrics will be populated by JavaScript -->
        </div>
        <div class="timestamp" id="timestamp"></div>
    </div>
    <script>
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    const grid = document.getElementById('metrics-grid');
                    grid.innerHTML = '';
                    
                    // Collaborative Filtering Metrics
                    const cfCard = createMetricCard(
                        'Collaborative Filtering',
                        data.collaborative_metrics.users_processed,
                        `Matrix: ${data.collaborative_metrics.matrix_size}`,
                        '👥'
                    );
                    grid.appendChild(cfCard);
                    
                    // Content-Based Metrics
                    const cbCard = createMetricCard(
                        'Content-Based',
                        data.content_metrics.items_processed,
                        `Features: ${data.content_metrics.features_count}`,
                        '📝'
                    );
                    grid.appendChild(cbCard);
                    
                    // Hybrid Metrics
                    const hybridCard = createMetricCard(
                        'Hybrid Recommender',
                        data.hybrid_metrics.recommendations_generated,
                        `CF: ${data.hybrid_metrics.cf_weight}, CB: ${data.hybrid_metrics.cb_weight}`,
                        '🔀'
                    );
                    grid.appendChild(hybridCard);
                    
                    // User Profiles
                    const profilesCard = createMetricCard(
                        'User Profiles',
                        data.content_metrics.user_profiles_created,
                        'Content-based profiles created',
                        '👤'
                    );
                    grid.appendChild(profilesCard);
                    
                    // Recommendations Generated
                    const recsCard = createMetricCard(
                        'Recommendations',
                        data.collaborative_metrics.recommendations_generated,
                        'Total recommendations generated',
                        '🎯'
                    );
                    grid.appendChild(recsCard);
                    
                    // Demo Metrics
                    const demoCard = createMetricCard(
                        'Demos Run',
                        data.demo_metrics.demos_run,
                        `Success Rate: ${data.demo_metrics.success_rate.toFixed(1)}%`,
                        '🚀'
                    );
                    grid.appendChild(demoCard);
                    
                    document.getElementById('timestamp').textContent = 
                        `Last updated: ${new Date(data.timestamp).toLocaleString()}`;
                })
                .catch(error => {
                    console.error('Error fetching metrics:', error);
                });
        }
        
        function createMetricCard(title, value, label, emoji) {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <div class="metric-title">${emoji} ${title}</div>
                <div class="metric-value">${value}</div>
                <div class="metric-label">${label}</div>
            `;
            return card;
        }
        
        function showStatus(message, isError = false) {
            const statusDiv = document.getElementById('status-message');
            statusDiv.textContent = message;
            statusDiv.className = 'status-message ' + (isError ? 'status-error' : 'status-success');
            statusDiv.style.display = 'block';
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }
        
        function runCollaborativeFiltering() {
            showStatus('Running collaborative filtering demo...');
            fetch('/api/run-collaborative', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`✅ Collaborative filtering complete! Generated ${data.results.recommendations} recommendations`);
                    updateDashboard();
                } else {
                    showStatus('❌ Demo failed: ' + data.message, true);
                }
            })
            .catch(error => {
                showStatus('❌ Error running demo: ' + error.message, true);
            });
        }
        
        function runContentBased() {
            showStatus('Running content-based filtering demo...');
            fetch('/api/run-content-based', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`✅ Content-based filtering complete! Generated ${data.results.recommendations} recommendations`);
                    updateDashboard();
                } else {
                    showStatus('❌ Demo failed: ' + data.message, true);
                }
            })
            .catch(error => {
                showStatus('❌ Error running demo: ' + error.message, true);
            });
        }
        
        function runHybrid() {
            showStatus('Running hybrid recommender demo...');
            fetch('/api/run-hybrid', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`✅ Hybrid recommender complete! Generated ${data.results.recommendations} recommendations`);
                    updateDashboard();
                } else {
                    showStatus('❌ Demo failed: ' + data.message, true);
                }
            })
            .catch(error => {
                showStatus('❌ Error running demo: ' + error.message, true);
            });
        }
        
        function updateDemo() {
            fetch('/api/update-demo', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus('✅ Demo counter updated!');
                    updateDashboard();
                } else {
                    showStatus('❌ Failed to update demo counter', true);
                }
            })
            .catch(error => {
                showStatus('❌ Error: ' + error.message, true);
            });
        }
        
        // Update every 2 seconds
        updateDashboard();
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>
"""

@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon to prevent 404"""
    return '', 204

@app.route('/')
def dashboard():
    """Serve dashboard HTML"""
    initialize_engines()
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    initialize_engines()
    metrics_data['timestamp'] = datetime.now().isoformat()
    return jsonify(metrics_data)

@app.route('/api/update-demo', methods=['POST'])
def update_demo_metrics():
    """Update demo metrics"""
    metrics_data['demo_metrics']['demos_run'] += 1
    metrics_data['demo_metrics']['last_demo_time'] = datetime.now().isoformat()
    return jsonify({'status': 'success'})

@app.route('/api/run-collaborative', methods=['POST'])
def run_collaborative():
    """Run collaborative filtering demo"""
    try:
        initialize_engines()
        result = cf_engine.recommend(user_id=5, top_n=5)
        
        metrics_data['collaborative_metrics']['recommendations_generated'] += len(result.recommended_items)
        if result.scores:
            metrics_data['collaborative_metrics']['avg_similarity_score'] = np.mean(result.scores)
        
        return jsonify({
            'status': 'success',
            'results': {
                'recommendations': len(result.recommended_items),
                'items': result.recommended_items,
                'scores': result.scores
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/run-content-based', methods=['POST'])
def run_content_based():
    """Run content-based filtering demo"""
    try:
        initialize_engines()
        user_rated_items = set(interactions_df[interactions_df['user_id'] == 5]['item_id'])
        result = cb_engine.recommend(user_id=5, top_n=5, exclude_items=user_rated_items)
        
        metrics_data['content_metrics']['items_processed'] = len(item_features_df)
        if result.scores:
            metrics_data['content_metrics']['avg_similarity_score'] = np.mean(result.scores)
        
        return jsonify({
            'status': 'success',
            'results': {
                'recommendations': len(result.recommended_items),
                'items': result.recommended_items,
                'scores': result.scores
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/run-hybrid', methods=['POST'])
def run_hybrid():
    """Run hybrid recommender demo"""
    try:
        initialize_engines()
        result = hybrid_engine.recommend(user_id=5, top_n=5)
        
        metrics_data['hybrid_metrics']['recommendations_generated'] += len(result.recommended_items)
        if result.scores:
            metrics_data['hybrid_metrics']['avg_combined_score'] = np.mean(result.scores)
        
        return jsonify({
            'status': 'success',
            'results': {
                'recommendations': len(result.recommended_items),
                'items': result.recommended_items,
                'scores': result.scores
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Recommender Systems Dashboard on http://localhost:5000")
    print("📊 Access dashboard at: http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/api/metrics")
    app.run(host='0.0.0.0', port=5000, debug=True)
EOF

# Create demo.py in sources
cat > sources/demo.py << 'EOF'
"""
Interactive Demo for Day 103: Recommender Systems Theory
"""

import sys
import os
import requests
import time

# Add parent directory to path to import lesson_code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lesson_code import (
    CollaborativeFilteringEngine,
    ContentBasedEngine,
    HybridRecommender,
    create_sample_dataset
)

def run_demo():
    """Run interactive demo"""
    print("\n" + "="*70)
    print("🎮 INTERACTIVE RECOMMENDER SYSTEMS DEMO")
    print("="*70)
    
    print("\n1️⃣  Generating sample dataset...")
    time.sleep(1)
    interactions, item_features = create_sample_dataset()
    print(f"   Created {len(interactions)} ratings from {interactions['user_id'].nunique()} users")
    print(f"   Item catalog: {interactions['item_id'].nunique()} items")
    
    test_user = 5
    
    print("\n2️⃣  Running Collaborative Filtering Demo...")
    time.sleep(1)
    cf_engine = CollaborativeFilteringEngine()
    cf_engine.fit(interactions)
    cf_results = cf_engine.recommend(test_user, top_n=5)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(cf_results.recommended_items, cf_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    print("\n3️⃣  Running Content-Based Filtering Demo...")
    time.sleep(1)
    cb_engine = ContentBasedEngine()
    cb_engine.fit(item_features, interactions)
    user_rated_items = set(interactions[interactions['user_id'] == test_user]['item_id'])
    cb_results = cb_engine.recommend(test_user, top_n=5, exclude_items=user_rated_items)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(cb_results.recommended_items, cb_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    print("\n4️⃣  Running Hybrid Recommender Demo...")
    time.sleep(1)
    hybrid = HybridRecommender(collaborative_weight=0.6, content_weight=0.4)
    hybrid.fit(interactions, item_features)
    hybrid_results = hybrid.recommend(test_user, top_n=5)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(hybrid_results.recommended_items, hybrid_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    # Try to update dashboard if running
    try:
        response = requests.post('http://localhost:5000/api/update-demo', timeout=1)
        if response.status_code == 200:
            print("\n✅ Demo metrics updated on dashboard")
    except:
        print("\nℹ️  Dashboard not running (this is okay)")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\n📊 Key Results:")
    print(f"   - Collaborative Filtering: {len(cf_results.recommended_items)} recommendations")
    print(f"   - Content-Based Filtering: {len(cb_results.recommended_items)} recommendations")
    print(f"   - Hybrid Recommender: {len(hybrid_results.recommended_items)} recommendations")
    print("\n🚀 Next: Check the dashboard at http://localhost:5000")
    
    return {
        'cf': cf_results,
        'cb': cb_results,
        'hybrid': hybrid_results
    }

if __name__ == "__main__":
    run_demo()
EOF

echo "✓ setup.sh"
echo "✓ requirements.txt"
echo "✓ lesson_code.py"
echo "✓ test_lesson.py"
echo "✓ README.md"
echo "✓ sources/startup.sh"
echo "✓ sources/dashboard.py"
echo "✓ sources/demo.py"
echo ""
echo "All files generated successfully!"
echo ""
echo "Next steps:"
echo "1. chmod +x setup.sh && ./setup.sh"
echo "2. source venv/bin/activate"
echo "3. python lesson_code.py"
echo "4. pytest test_lesson.py -v"
EOF

chmod +x generate_lesson_files.sh

echo "✓ generate_lesson_files.sh created"