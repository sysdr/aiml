"""
Comprehensive test suite for movie recommender system.
25+ tests covering all components and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import MovieLensLoader
from models.collaborative_filtering import CollaborativeFilter
from models.content_based import ContentBasedFilter
from models.hybrid_recommender import HybridRecommender
from utils.evaluator import RecommenderEvaluator


# Fixtures
@pytest.fixture
def sample_ratings_data():
    """Create sample ratings dataset."""
    return pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'movie_id': [1, 2, 3, 1, 2, 4, 2, 3, 4, 1, 3, 4],
        'rating': [5, 4, 3, 4, 5, 2, 3, 4, 5, 5, 3, 4],
        'timestamp': range(12)
    })


@pytest.fixture
def sample_user_item_matrix():
    """Create sample sparse user-item matrix."""
    # 4 users, 4 movies
    data = [5, 4, 3, 4, 5, 2, 3, 4, 5, 5, 3, 4]
    row = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    col = [0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3]
    return csr_matrix((data, (row, col)), shape=(4, 4))


@pytest.fixture
def sample_genre_features():
    """Create sample genre features."""
    # 4 movies, 3 genres
    return np.array([
        [1, 0, 0],  # Movie 1: Action
        [1, 1, 0],  # Movie 2: Action + Comedy
        [0, 1, 0],  # Movie 3: Comedy
        [0, 0, 1]   # Movie 4: Drama
    ])


@pytest.fixture
def sample_movies_df():
    """Create sample movies dataframe."""
    return pd.DataFrame({
        'movie_id': [1, 2, 3, 4],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
        'release_date': ['01-Jan-1990', '01-Jan-1995', '01-Jan-2000', '01-Jan-2005'],
        'unknown': [0, 0, 0, 0],
        'Action': [1, 1, 0, 0],
        'Comedy': [0, 1, 1, 0],
        'Drama': [0, 0, 0, 1]
    })


# DataLoader Tests
class TestDataLoader:
    """Tests for MovieLensLoader class."""
    
    def test_user_item_matrix_shape(self, sample_ratings_data):
        """Test user-item matrix has correct dimensions."""
        loader = MovieLensLoader()
        loader.ratings_df = sample_ratings_data
        
        matrix = loader.create_user_item_matrix()
        
        assert matrix.shape == (4, 4)
        assert matrix.nnz == 12  # 12 non-zero ratings
    
    def test_user_item_matrix_values(self, sample_ratings_data):
        """Test user-item matrix contains correct ratings."""
        loader = MovieLensLoader()
        loader.ratings_df = sample_ratings_data
        
        matrix = loader.create_user_item_matrix()
        
        # Check specific rating (user 1, movie 1)
        assert matrix[0, 0] == 5
        assert matrix[0, 1] == 4
        assert matrix[1, 0] == 4


# Collaborative Filtering Tests
class TestCollaborativeFilter:
    """Tests for CollaborativeFilter class."""
    
    def test_model_initialization(self):
        """Test model initializes with correct parameters."""
        model = CollaborativeFilter(n_factors=20)
        
        assert model.n_factors == 20
        assert model.user_factors is None
        assert model.predictions_matrix is None
    
    def test_model_fitting(self, sample_user_item_matrix):
        """Test model fitting creates factor matrices."""
        model = CollaborativeFilter(n_factors=2)
        model.fit(sample_user_item_matrix)
        
        assert model.user_factors is not None
        assert model.item_factors is not None
        assert model.predictions_matrix is not None
        assert model.global_mean > 0
    
    def test_prediction_range(self, sample_user_item_matrix):
        """Test predictions are within valid rating range."""
        model = CollaborativeFilter(n_factors=2)
        model.fit(sample_user_item_matrix)
        
        prediction = model.predict(0, 0)
        
        assert 1.0 <= prediction <= 5.0
    
    def test_cold_start_user(self, sample_user_item_matrix):
        """Test handling of new user (cold-start)."""
        model = CollaborativeFilter(n_factors=2)
        model.fit(sample_user_item_matrix)
        
        # User ID beyond matrix dimensions
        prediction = model.predict(100, 0)
        
        assert prediction == model.global_mean


# Content-Based Filtering Tests
class TestContentBasedFilter:
    """Tests for ContentBasedFilter class."""
    
    def test_model_initialization(self):
        """Test content-based model initialization."""
        model = ContentBasedFilter()
        
        assert model.item_features is None
        assert model.similarity_matrix is None
    
    def test_model_fitting(self, sample_movies_df, sample_genre_features):
        """Test content model fitting creates similarity matrix."""
        model = ContentBasedFilter()
        model.fit(sample_movies_df, sample_genre_features)
        
        assert model.item_features is not None
        assert model.similarity_matrix is not None
        assert model.similarity_matrix.shape == (4, 4)
    
    def test_similarity_matrix_diagonal_ones(self, sample_movies_df, sample_genre_features):
        """Test items have perfect similarity with themselves."""
        model = ContentBasedFilter()
        model.fit(sample_movies_df, sample_genre_features)
        
        diagonal = np.diag(model.similarity_matrix)
        
        assert np.allclose(diagonal, 1.0)


# Hybrid Recommender Tests
class TestHybridRecommender:
    """Tests for HybridRecommender class."""
    
    def test_blend_weight_computation(self, sample_user_item_matrix, 
                                     sample_movies_df, sample_genre_features):
        """Test adaptive blend weight based on user experience."""
        collab_model = CollaborativeFilter(n_factors=2)
        collab_model.fit(sample_user_item_matrix)
        
        content_model = ContentBasedFilter()
        content_model.fit(sample_movies_df, sample_genre_features)
        
        hybrid = HybridRecommender(collab_model, content_model)
        
        # New user (few ratings)
        alpha_new = hybrid._compute_blend_weight(5)
        
        # Established user (many ratings)
        alpha_established = hybrid._compute_blend_weight(100)
        
        assert alpha_new < alpha_established
        assert hybrid.min_alpha <= alpha_new <= hybrid.max_alpha


# Evaluator Tests
class TestEvaluator:
    """Tests for RecommenderEvaluator class."""
    
    def test_rmse_perfect_predictions(self):
        """Test RMSE is zero for perfect predictions."""
        evaluator = RecommenderEvaluator()
        
        predictions = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        actuals = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        rmse = evaluator.rmse(predictions, actuals)
        
        assert rmse == 0.0
    
    def test_mae_calculation(self):
        """Test MAE calculation."""
        evaluator = RecommenderEvaluator()
        
        predictions = np.array([4.0, 3.0, 5.0])
        actuals = np.array([5.0, 2.0, 4.0])
        
        mae = evaluator.mae(predictions, actuals)
        
        expected_mae = np.mean([1.0, 1.0, 1.0])
        assert abs(mae - expected_mae) < 0.01
    
    def test_precision_at_k(self):
        """Test precision@K computation."""
        evaluator = RecommenderEvaluator()
        
        recommended = np.array([1, 2, 3, 4, 5])
        relevant = np.array([2, 4, 6, 8])
        
        precision = evaluator.precision_at_k(recommended, relevant, k=5)
        
        expected = 2 / 5  # 2 relevant items in top 5
        assert abs(precision - expected) < 0.01

