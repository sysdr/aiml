"""
Test suite for Day 104: Collaborative Filtering
"""

import pytest
import numpy as np
import pandas as pd
from lesson_code import (
    CollaborativeFiltering,
    create_sample_dataset,
    RecommendationResult
)


class TestCollaborativeFiltering:
    """Test collaborative filtering implementation"""
    
    def test_initialization(self):
        """Test engine can be initialized"""
        cf = CollaborativeFiltering(method='user-based', min_common_items=2)
        assert cf.method == 'user-based'
        assert cf.min_common_items == 2
        assert cf.user_item_matrix is None
    
    def test_fit_creates_matrix(self):
        """Test fitting creates user-item matrix"""
        interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [10, 11, 10, 12, 11],
            'rating': [5, 4, 3, 5, 4]
        })
        
        cf = CollaborativeFiltering()
        cf.fit(interactions)
        
        assert cf.user_item_matrix is not None
        assert len(cf.user_item_matrix) == 3
        assert len(cf.user_item_matrix.columns) == 3
    
    def test_compute_user_similarity(self):
        """Test computing user similarity"""
        interactions = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3],
            'item_id': [10, 11, 12, 10, 11, 13, 12, 13],
            'rating': [5, 4, 3, 5, 4, 5, 3, 5]
        })
        
        cf = CollaborativeFiltering(method='user-based')
        cf.fit(interactions)
        
        similar = cf.compute_user_similarity(1, top_k=2)
        assert len(similar) <= 2
        assert all(isinstance(user_id, int) for user_id, _ in similar)
        assert all(0 <= sim <= 1 for _, sim in similar)
    
    def test_recommend_user_based(self):
        """Test user-based recommendations"""
        interactions, _ = create_sample_dataset(), None
        if isinstance(interactions, tuple):
            interactions = interactions[0]
        
        cf = CollaborativeFiltering(method='user-based')
        cf.fit(interactions)
        
        result = cf.recommend(user_id=5, top_n=5)
        
        assert isinstance(result, RecommendationResult)
        assert result.user_id == 5
        assert len(result.recommended_items) <= 5
        assert result.method == "user-based"
    
    def test_recommend_item_based(self):
        """Test item-based recommendations"""
        interactions = create_sample_dataset()
        if isinstance(interactions, tuple):
            interactions = interactions[0]
        
        cf = CollaborativeFiltering(method='item-based')
        cf.fit(interactions)
        
        result = cf.recommend(user_id=5, top_n=5)
        
        assert isinstance(result, RecommendationResult)
        assert result.method == "item-based"
        assert len(result.recommended_items) <= 5
    
    def test_no_duplicate_recommendations(self):
        """Test no duplicate recommendations"""
        interactions = create_sample_dataset()
        if isinstance(interactions, tuple):
            interactions = interactions[0]
        
        cf = CollaborativeFiltering()
        cf.fit(interactions)
        
        result = cf.recommend(user_id=5, top_n=10)
        
        assert len(result.recommended_items) == len(set(result.recommended_items))


class TestDataGeneration:
    """Test sample dataset generation"""
    
    def test_create_sample_dataset(self):
        """Test dataset creation"""
        interactions = create_sample_dataset()
        if isinstance(interactions, tuple):
            interactions = interactions[0]
        
        assert isinstance(interactions, pd.DataFrame)
        assert 'user_id' in interactions.columns
        assert 'item_id' in interactions.columns
        assert 'rating' in interactions.columns
    
    def test_ratings_in_range(self):
        """Test ratings are in valid range"""
        interactions = create_sample_dataset()
        if isinstance(interactions, tuple):
            interactions = interactions[0]
        
        assert interactions['rating'].min() >= 1
        assert interactions['rating'].max() <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
