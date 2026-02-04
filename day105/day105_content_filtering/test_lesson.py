"""
Day 105: Content-Based Filtering - Test Suite
Comprehensive tests for the recommendation system
"""

import pytest
import numpy as np
import pandas as pd
from lesson_code import ContentBasedRecommender, create_sample_dataset


class TestContentBasedRecommender:
    """Test suite for ContentBasedRecommender class."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample movie data."""
        return create_sample_dataset()
    
    @pytest.fixture
    def fitted_recommender(self, sample_data):
        """Fixture providing a fitted recommender."""
        recommender = ContentBasedRecommender(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1
        )
        recommender.fit(sample_data, text_column='combined_features')
        return recommender
    
    def test_initialization(self):
        """Test recommender initialization."""
        recommender = ContentBasedRecommender(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2
        )
        
        assert recommender.fitted is False
        assert recommender.tfidf_matrix is None
        assert recommender.similarity_matrix is None
        assert recommender.vectorizer.max_features == 5000
        assert recommender.vectorizer.ngram_range == (1, 3)
    
    def test_fit(self, sample_data):
        """Test model fitting."""
        recommender = ContentBasedRecommender()
        recommender.fit(sample_data, text_column='combined_features')
        
        assert recommender.fitted is True
        assert recommender.tfidf_matrix is not None
        assert recommender.similarity_matrix is not None
        assert recommender.tfidf_matrix.shape[0] == len(sample_data)
        assert recommender.similarity_matrix.shape == (len(sample_data), len(sample_data))
        assert recommender.metrics['total_items'] == len(sample_data)
        assert recommender.metrics['vocabulary_size'] > 0
    
    def test_similarity_matrix_properties(self, fitted_recommender):
        """Test similarity matrix has correct properties."""
        sim_matrix = fitted_recommender.similarity_matrix
        
        # Diagonal should be 1 (self-similarity)
        np.testing.assert_array_almost_equal(
            np.diag(sim_matrix),
            np.ones(sim_matrix.shape[0])
        )
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T)
        
        # All values should be between 0 and 1 (with tolerance for floating point)
        assert np.all(sim_matrix >= -1e-10)  # Allow small negative due to floating point
        assert np.all(sim_matrix <= 1 + 1e-10)  # Allow small positive due to floating point
    
    def test_get_recommendations(self, fitted_recommender):
        """Test recommendation generation."""
        recommendations = fitted_recommender.get_recommendations(
            'movie_001',
            n_recommendations=5
        )
        
        assert len(recommendations) <= 5
        assert all('item_id' in rec for rec in recommendations)
        assert all('similarity_score' in rec for rec in recommendations)
        assert all('final_score' in rec for rec in recommendations)
        
        # Scores should be in descending order
        scores = [rec['final_score'] for rec in recommendations]
        assert scores == sorted(scores, reverse=True)
    
    def test_recommendation_excludes_self(self, fitted_recommender):
        """Test that recommendations don't include the target item."""
        target_id = 'movie_001'
        recommendations = fitted_recommender.get_recommendations(
            target_id,
            n_recommendations=10
        )
        
        rec_ids = [rec['item_id'] for rec in recommendations]
        assert target_id not in rec_ids
    
    def test_diversity_filtering(self, fitted_recommender):
        """Test diversity filtering reduces similar items."""
        # Get recommendations with and without diversity
        recs_no_diversity = fitted_recommender.get_recommendations(
            'movie_001',
            n_recommendations=5,
            diversity_threshold=1.0  # No filtering
        )
        
        recs_with_diversity = fitted_recommender.get_recommendations(
            'movie_001',
            n_recommendations=5,
            diversity_threshold=0.7  # Strict filtering
        )
        
        # With diversity filtering, we might get fewer or different results
        assert len(recs_with_diversity) <= len(recs_no_diversity)
    
    def test_popularity_boosting(self, fitted_recommender, sample_data):
        """Test that popularity boosting affects scores."""
        # Ensure we have items with different popularity
        assert sample_data['popularity'].nunique() > 1
        
        recs_no_boost = fitted_recommender.get_recommendations(
            'movie_001',
            n_recommendations=5,
            apply_boost=False
        )
        
        recs_with_boost = fitted_recommender.get_recommendations(
            'movie_001',
            n_recommendations=5,
            apply_boost=True
        )
        
        # Final scores should differ when boost is applied
        scores_no_boost = [r['final_score'] for r in recs_no_boost]
        scores_with_boost = [r['final_score'] for r in recs_with_boost]
        
        # At least some scores should be different
        assert not np.allclose(scores_no_boost[:3], scores_with_boost[:3])
    
    def test_add_new_item(self, fitted_recommender):
        """Test incremental item addition."""
        initial_count = fitted_recommender.metrics['total_items']
        
        new_item = {
            'item_id': 'movie_new',
            'title': 'Test Movie',
            'genres': 'Science Fiction',
            'description': 'A test movie for verification',
            'director': 'Test Director',
            'actors': 'Test Actor',
            'year': 2024,
            'popularity': 5000
        }
        
        new_features = ' '.join([
            new_item['title'],
            new_item['genres'],
            new_item['description']
        ])
        
        fitted_recommender.add_new_item(new_item, new_features)
        
        assert fitted_recommender.metrics['total_items'] == initial_count + 1
        assert 'movie_new' in fitted_recommender.item_indices
        
        # Should be able to get recommendations for the new item
        recs = fitted_recommender.get_recommendations('movie_new', n_recommendations=3)
        assert len(recs) > 0
    
    def test_get_feature_importance(self, fitted_recommender):
        """Test feature importance extraction."""
        features = fitted_recommender.get_feature_importance('movie_001', top_n=5)
        
        assert len(features) <= 5
        assert all(isinstance(f, tuple) for f in features)
        assert all(len(f) == 2 for f in features)
        assert all(isinstance(f[0], str) for f in features)
        assert all(isinstance(f[1], (int, float)) for f in features)
        
        # Weights should be in descending order
        weights = [f[1] for f in features]
        assert weights == sorted(weights, reverse=True)
    
    def test_invalid_item_id(self, fitted_recommender):
        """Test error handling for invalid item IDs."""
        with pytest.raises(ValueError, match="not found"):
            fitted_recommender.get_recommendations('invalid_id')
        
        with pytest.raises(ValueError, match="not found"):
            fitted_recommender.get_feature_importance('invalid_id')
    
    def test_unfitted_recommender(self):
        """Test that unfitted recommender raises errors."""
        recommender = ContentBasedRecommender()
        
        with pytest.raises(ValueError, match="must be fitted"):
            recommender.get_recommendations('movie_001')
        
        with pytest.raises(ValueError, match="must be fitted"):
            recommender.add_new_item({}, "test")
    
    def test_metrics_tracking(self, fitted_recommender):
        """Test that metrics are tracked correctly."""
        initial_recs = fitted_recommender.metrics['recommendations_served']
        
        fitted_recommender.get_recommendations('movie_001', n_recommendations=3)
        fitted_recommender.get_recommendations('movie_002', n_recommendations=3)
        
        assert fitted_recommender.metrics['recommendations_served'] == initial_recs + 2
    
    def test_content_similarity_sanity(self, fitted_recommender):
        """Test that similar content items have high similarity."""
        # The Matrix (sci-fi action) should be more similar to
        # Inception (sci-fi thriller) than to The Notebook (romance)
        
        matrix_to_inception = fitted_recommender.similarity_matrix[
            fitted_recommender.item_indices['movie_001'],
            fitted_recommender.item_indices['movie_002']
        ]
        
        matrix_to_notebook = fitted_recommender.similarity_matrix[
            fitted_recommender.item_indices['movie_001'],
            fitted_recommender.item_indices['movie_003']
        ]
        
        assert matrix_to_inception > matrix_to_notebook


def test_create_sample_dataset():
    """Test sample dataset creation."""
    df = create_sample_dataset()
    
    assert len(df) > 0
    assert 'item_id' in df.columns
    assert 'title' in df.columns
    assert 'genres' in df.columns
    assert 'combined_features' in df.columns
    assert df['item_id'].is_unique


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
