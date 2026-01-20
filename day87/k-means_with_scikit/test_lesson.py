"""
Day 87: K-Means with Scikit-learn - Comprehensive Test Suite

Tests cover:
- Model initialization and configuration
- Data preprocessing and scaling
- Training and prediction
- Edge cases and error handling
- Model persistence
- Production scenarios
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from lesson_code import (
    CustomerSegmentation,
    generate_synthetic_customers,
    analyze_segments
)


class TestCustomerSegmentation:
    """Test suite for CustomerSegmentation class."""
    
    def test_model_initialization(self):
        """Test model initializes with correct parameters."""
        model = CustomerSegmentation(n_clusters=3, random_state=42)
        
        assert model.n_clusters == 3
        assert model.kmeans.random_state == 42
        assert model.kmeans.init == 'k-means++'
        assert not model.is_fitted
    
    def test_fit_basic(self):
        """Test basic model fitting."""
        X, _, feature_names = generate_synthetic_customers(
            n_samples=100,
            n_features=3,
            n_clusters=3
        )
        
        model = CustomerSegmentation(n_clusters=3)
        result = model.fit(X, feature_names)
        
        # Check fit returns self for chaining
        assert result is model
        assert model.is_fitted
        assert model.feature_names == feature_names
    
    def test_fit_scales_features(self):
        """Test that fit properly scales features."""
        # Create data with different scales
        X = np.array([
            [1, 1000],
            [2, 2000],
            [3, 3000],
            [100, 10],
            [200, 20],
            [300, 30]
        ])
        
        model = CustomerSegmentation(n_clusters=2)
        model.fit(X)
        
        # Check scaler was fitted
        assert hasattr(model.scaler, 'mean_')
        assert hasattr(model.scaler, 'scale_')
        
        # Verify scaling is reasonable
        assert not np.allclose(model.scaler.mean_, 0)
        assert not np.allclose(model.scaler.scale_, 1)
    
    def test_predict_basic(self):
        """Test basic prediction functionality."""
        X, _, _ = generate_synthetic_customers(n_samples=100, n_clusters=3)
        
        model = CustomerSegmentation(n_clusters=3)
        model.fit(X)
        
        # Predict on same data
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert predictions.min() >= 0
        assert predictions.max() < 3
    
    def test_predict_without_fit_raises_error(self):
        """Test that predict raises error if model not fitted."""
        model = CustomerSegmentation(n_clusters=3)
        X = np.random.randn(10, 3)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)
    
    def test_fit_predict(self):
        """Test fit_predict convenience method."""
        X, y_true, feature_names = generate_synthetic_customers(
            n_samples=100,
            n_clusters=3
        )
        
        model = CustomerSegmentation(n_clusters=3)
        labels = model.fit_predict(X, feature_names)
        
        assert model.is_fitted
        assert len(labels) == len(X)
        assert model.feature_names == feature_names
    
    def test_get_cluster_centers(self):
        """Test cluster center retrieval in original scale."""
        X, _, _ = generate_synthetic_customers(n_samples=100, n_features=4, n_clusters=3)
        
        model = CustomerSegmentation(n_clusters=3)
        model.fit(X)
        
        centers = model.get_cluster_centers()
        
        # Check shape
        assert centers.shape == (3, 4)  # n_clusters × n_features
        
        # Centers should be in similar range as original data
        assert centers.min() >= X.min() - X.std()
        assert centers.max() <= X.max() + X.std()
    
    def test_get_cluster_centers_without_fit_raises_error(self):
        """Test that get_cluster_centers requires fitted model."""
        model = CustomerSegmentation(n_clusters=3)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_cluster_centers()
    
    def test_get_metrics(self):
        """Test clustering metrics retrieval."""
        X, _, _ = generate_synthetic_customers(n_samples=100, n_clusters=3)
        
        model = CustomerSegmentation(n_clusters=3)
        model.fit(X)
        
        metrics = model.get_metrics()
        
        assert 'inertia' in metrics
        assert 'n_iter' in metrics
        assert 'n_clusters' in metrics
        
        assert metrics['inertia'] > 0
        assert metrics['n_iter'] > 0
        assert metrics['n_clusters'] == 3
    
    def test_model_persistence(self):
        """Test saving and loading model."""
        X, _, feature_names = generate_synthetic_customers(n_samples=100, n_clusters=3)
        
        # Train model
        model = CustomerSegmentation(n_clusters=3)
        model.fit(X, feature_names)
        original_predictions = model.predict(X[:10])
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.pkl')
            model.save(filepath)
            
            # Load model
            loaded_model = CustomerSegmentation.load(filepath)
            loaded_predictions = loaded_model.predict(X[:10])
            
            # Predictions should be identical
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            assert loaded_model.n_clusters == model.n_clusters
            assert loaded_model.is_fitted
    
    def test_save_creates_directory(self):
        """Test that save creates parent directories if needed."""
        X, _, _ = generate_synthetic_customers(n_samples=100, n_clusters=3)
        
        model = CustomerSegmentation(n_clusters=3)
        model.fit(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'nested', 'dir', 'model.pkl')
            model.save(filepath)
            
            assert os.path.exists(filepath)
    
    def test_fit_on_empty_data_raises_error(self):
        """Test that fitting on empty data raises error."""
        model = CustomerSegmentation(n_clusters=3)
        X = np.array([]).reshape(0, 3)
        
        with pytest.raises(ValueError, match="Cannot fit on empty dataset"):
            model.fit(X)
    
    def test_fit_with_too_few_samples_raises_error(self):
        """Test that fitting with fewer samples than clusters raises error."""
        model = CustomerSegmentation(n_clusters=5)
        X = np.random.randn(3, 4)  # Only 3 samples, 5 clusters
        
        with pytest.raises(ValueError, match="Number of samples.*must be >= n_clusters"):
            model.fit(X)
    
    def test_high_dimensional_clustering(self):
        """Test clustering with many features (production scenario)."""
        # Generate high-dimensional data (e.g., user embeddings)
        X, _, _ = generate_synthetic_customers(
            n_samples=200,
            n_features=50,  # High-dimensional
            n_clusters=5
        )
        
        model = CustomerSegmentation(n_clusters=5)
        labels = model.fit_predict(X)
        
        assert len(labels) == 200
        assert len(np.unique(labels)) == 5  # All clusters used
    
    def test_single_cluster(self):
        """Test edge case of single cluster."""
        X, _, _ = generate_synthetic_customers(n_samples=100, n_clusters=1)
        
        model = CustomerSegmentation(n_clusters=1)
        labels = model.fit_predict(X)
        
        # All points should be in cluster 0
        assert np.all(labels == 0)
        assert len(model.get_cluster_centers()) == 1


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_generate_synthetic_customers(self):
        """Test synthetic data generation."""
        X, y, feature_names = generate_synthetic_customers(
            n_samples=500,
            n_features=3,
            n_clusters=4,
            random_state=42
        )
        
        assert X.shape == (500, 3)
        assert len(y) == 500
        assert len(feature_names) == 3
        assert len(np.unique(y)) == 4  # 4 true clusters
    
    def test_generate_synthetic_customers_reproducibility(self):
        """Test that same random_state produces same data."""
        X1, y1, _ = generate_synthetic_customers(random_state=42)
        X2, y2, _ = generate_synthetic_customers(random_state=42)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_analyze_segments(self):
        """Test segment analysis function."""
        X, _, feature_names = generate_synthetic_customers(
            n_samples=100,
            n_features=4,
            n_clusters=3
        )
        
        model = CustomerSegmentation(n_clusters=3)
        labels = model.fit_predict(X, feature_names)
        
        df = analyze_segments(model, X, labels, feature_names)
        
        # Check output structure
        assert len(df) == 3  # 3 segments
        assert 'Segment' in df.columns
        assert 'Size' in df.columns
        assert 'Percentage' in df.columns
        
        # Check all feature names present
        for feature in feature_names:
            assert feature in df.columns
        
        # Check percentages sum to ~100%
        percentages = df['Percentage'].str.rstrip('%').astype(float)
        assert 99.0 <= percentages.sum() <= 101.0


class TestProductionScenarios:
    """Test production-level scenarios and edge cases."""
    
    def test_batch_prediction(self):
        """Test predicting on multiple batches (production pattern)."""
        # Train on historical data
        X_train, _, _ = generate_synthetic_customers(n_samples=1000, random_state=42)
        
        model = CustomerSegmentation(n_clusters=5)
        model.fit(X_train)
        
        # Predict on multiple batches of new data
        for i in range(5):
            X_batch, _, _ = generate_synthetic_customers(
                n_samples=100,
                random_state=100 + i
            )
            predictions = model.predict(X_batch)
            
            assert len(predictions) == 100
            assert predictions.min() >= 0
            assert predictions.max() < 5
    
    def test_numerical_stability(self):
        """Test model handles extreme values gracefully."""
        # Create data with extreme values
        X = np.array([
            [1e-10, 1e10],
            [2e-10, 2e10],
            [1000, 0.001],
            [2000, 0.002]
        ] * 10)  # Repeat to have enough samples
        
        model = CustomerSegmentation(n_clusters=2)
        
        # Should not raise errors
        labels = model.fit_predict(X)
        assert len(labels) == len(X)
    
    def test_different_initialization_methods(self):
        """Test different initialization strategies."""
        X, _, _ = generate_synthetic_customers(n_samples=100, n_clusters=3)
        
        for init_method in ['k-means++', 'random']:
            model = CustomerSegmentation(n_clusters=3, init=init_method)
            model.fit(X)
            
            assert model.is_fitted
            metrics = model.get_metrics()
            assert metrics['inertia'] > 0


def test_full_pipeline():
    """Integration test: full production pipeline."""
    # Generate data
    X, y_true, feature_names = generate_synthetic_customers(
        n_samples=500,
        n_features=4,
        n_clusters=5,
        random_state=42
    )
    
    # Train model
    model = CustomerSegmentation(n_clusters=5, random_state=42)
    labels = model.fit_predict(X, feature_names)
    
    # Get metrics
    metrics = model.get_metrics()
    assert metrics['inertia'] > 0
    
    # Analyze segments
    segment_df = analyze_segments(model, X, labels, feature_names)
    assert len(segment_df) == 5
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'model.pkl')
        model.save(filepath)
        loaded_model = CustomerSegmentation.load(filepath)
        
        # Predict on new data
        X_new, _, _ = generate_synthetic_customers(
            n_samples=50,
            random_state=999
        )
        predictions = loaded_model.predict(X_new)
        
        assert len(predictions) == 50
        assert predictions.min() >= 0
        assert predictions.max() < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
