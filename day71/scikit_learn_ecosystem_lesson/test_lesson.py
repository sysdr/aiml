"""
Test suite for Day 71: The Scikit-learn Ecosystem
Validates all components of the ML pipeline
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
import os

from lesson_code import (
    MovieRecommendationDataset,
    UserMovieFeatureTransformer,
    SklearnEcosystemPipeline
)


class TestMovieRecommendationDataset:
    """Test dataset generation"""
    
    def test_dataset_creation(self):
        """Verify dataset has correct structure"""
        dataset = MovieRecommendationDataset(
            n_users=100, n_movies=50, n_ratings=500
        )
        df = dataset.generate()
        
        assert len(df) == 500
        assert 'user_id' in df.columns
        assert 'movie_id' in df.columns
        assert 'rating' in df.columns
        assert df['rating'].between(1, 5).all()
    
    def test_dataset_metadata(self):
        """Verify metadata columns exist"""
        dataset = MovieRecommendationDataset(n_ratings=100)
        df = dataset.generate()
        
        required_cols = [
            'genre', 'movie_year', 'user_age', 'user_country', 'timestamp'
        ]
        for col in required_cols:
            assert col in df.columns


class TestUserMovieFeatureTransformer:
    """Test custom transformer"""
    
    def test_transformer_fit_transform(self):
        """Verify transformer creates expected features"""
        dataset = MovieRecommendationDataset(n_ratings=1000)
        df = dataset.generate()
        
        transformer = UserMovieFeatureTransformer()
        transformed = transformer.fit_transform(df)
        
        # Check new features were added
        assert 'user_avg_rating' in transformed.columns
        assert 'movie_avg_rating' in transformed.columns
        assert 'user_movie_interaction' in transformed.columns
        assert 'hour_of_day' in transformed.columns
        assert 'user_rating_count' in transformed.columns
    
    def test_transformer_sklearn_compatibility(self):
        """Verify transformer follows sklearn interface"""
        transformer = UserMovieFeatureTransformer()
        
        # Check required methods exist
        assert hasattr(transformer, 'fit')
        assert hasattr(transformer, 'transform')
        assert hasattr(transformer, 'fit_transform')


class TestSklearnEcosystemPipeline:
    """Test complete ML pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample dataset for testing"""
        dataset = MovieRecommendationDataset(
            n_users=200, n_movies=100, n_ratings=2000
        )
        return dataset.generate()
    
    def test_pipeline_creation(self, sample_data):
        """Verify pipeline is constructed correctly"""
        ml_pipeline = SklearnEcosystemPipeline()
        X, y, _, _, _ = ml_pipeline.create_features(sample_data)
        
        pipeline = ml_pipeline.build_pipeline()
        
        assert isinstance(pipeline, Pipeline)
        assert 'scaler' in pipeline.named_steps
        assert 'model' in pipeline.named_steps
    
    def test_pipeline_training(self, sample_data):
        """Verify pipeline trains successfully"""
        ml_pipeline = SklearnEcosystemPipeline()
        X, y, _, _, _ = ml_pipeline.create_features(sample_data)
        
        # Train without CV for speed
        pipeline = ml_pipeline.train(X, y, perform_cv=False, tune_hyperparameters=False)
        
        assert pipeline is not None
        assert hasattr(pipeline, 'predict')
    
    def test_pipeline_prediction(self, sample_data):
        """Verify pipeline makes predictions with correct shape"""
        ml_pipeline = SklearnEcosystemPipeline()
        X, y, _, _, _ = ml_pipeline.create_features(sample_data)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train and predict
        ml_pipeline.train(X_train, y_train, perform_cv=False)
        predictions = ml_pipeline.pipeline.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(1 <= p <= 5 for p in predictions)
    
    def test_pipeline_evaluation(self, sample_data):
        """Verify evaluation returns correct metrics"""
        ml_pipeline = SklearnEcosystemPipeline()
        X, y, _, _, _ = ml_pipeline.create_features(sample_data)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        ml_pipeline.train(X_train, y_train, perform_cv=False)
        metrics, predictions = ml_pipeline.evaluate(X_test, y_test)
        
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R2' in metrics
        assert metrics['RMSE'] > 0
        assert -1 <= metrics['R2'] <= 1
    
    def test_pipeline_serialization(self, sample_data, tmp_path):
        """Verify pipeline can be saved and loaded"""
        ml_pipeline = SklearnEcosystemPipeline()
        X, y, _, _, _ = ml_pipeline.create_features(sample_data)
        
        ml_pipeline.train(X, y, perform_cv=False)
        
        # Save to temp file
        temp_file = tmp_path / "test_pipeline.pkl"
        ml_pipeline.save_pipeline(str(temp_file))
        
        assert temp_file.exists()
        
        # Load and verify
        loaded_pipeline = SklearnEcosystemPipeline.load_pipeline(str(temp_file))
        assert hasattr(loaded_pipeline, 'predict')
        
        # Verify predictions match
        original_pred = ml_pipeline.pipeline.predict(X[:10])
        loaded_pred = loaded_pipeline.predict(X[:10])
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_end_to_end_workflow(self):
        """Test complete pipeline from data to deployment"""
        # Generate data
        dataset = MovieRecommendationDataset(
            n_users=300, n_movies=150, n_ratings=3000
        )
        df = dataset.generate()
        
        # Create features
        ml_pipeline = SklearnEcosystemPipeline()
        X, y, _, _, _ = ml_pipeline.create_features(df)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train
        ml_pipeline.train(X_train, y_train, perform_cv=False)
        
        # Evaluate
        metrics, predictions = ml_pipeline.evaluate(X_test, y_test)
        
        # Verify reasonable performance
        assert metrics['RMSE'] < 2.0  # Should predict within 2 stars
        assert metrics['R2'] > 0.0  # Should explain some variance
        
        # Save and load
        ml_pipeline.save_pipeline('test_integration_pipeline.pkl')
        loaded = SklearnEcosystemPipeline.load_pipeline('test_integration_pipeline.pkl')
        
        # Verify loaded pipeline works
        loaded_predictions = loaded.predict(X_test[:5])
        assert len(loaded_predictions) == 5
        
        # Cleanup
        os.remove('test_integration_pipeline.pkl')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
