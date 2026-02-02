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
