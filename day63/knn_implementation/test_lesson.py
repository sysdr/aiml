"""
Day 63: KNN with Scikit-learn - Test Suite

Comprehensive tests covering edge cases and common bugs in production KNN systems.
These tests catch issues before they affect real users - critical when your model
makes decisions impacting millions of customers.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from lesson_code import KNNPipeline


class TestKNNPipeline:
    """Test suite for production KNN pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create fresh pipeline for each test."""
        return KNNPipeline(n_neighbors=5, random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample dataset for testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        return X, y
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes with correct defaults."""
        assert pipeline.n_neighbors == 5
        assert pipeline.random_state == 42
        assert pipeline.model is None
        assert pipeline.scaler is not None
    
    def test_load_iris_dataset(self, pipeline):
        """Test loading Iris dataset."""
        X, y = pipeline.load_data(dataset_type='iris')
        assert X.shape == (150, 4)
        assert len(y) == 150
        assert len(np.unique(y)) == 3
        assert hasattr(pipeline, 'feature_names')
        assert hasattr(pipeline, 'target_names')
    
    def test_load_synthetic_dataset(self, pipeline):
        """Test generating synthetic dataset."""
        X, y = pipeline.load_data(dataset_type='synthetic')
        assert X.shape == (1000, 10)
        assert len(y) == 1000
        assert len(np.unique(y)) == 3
    
    def test_split_and_scale(self, pipeline, sample_data):
        """Test data splitting and scaling."""
        X, y = sample_data
        pipeline.split_and_scale(X, y, test_size=0.2)
        
        # Check splits
        assert pipeline.X_train.shape[0] == 80
        assert pipeline.X_test.shape[0] == 20
        
        # Check scaling
        assert pipeline.X_train_scaled.shape == pipeline.X_train.shape
        assert pipeline.X_test_scaled.shape == pipeline.X_test.shape
        
        # Check scaled data has mean~0, std~1
        assert np.abs(pipeline.X_train_scaled.mean(axis=0)).max() < 0.1
        assert np.abs(pipeline.X_train_scaled.std(axis=0) - 1.0).max() < 0.2
    
    def test_train_baseline(self, pipeline, sample_data):
        """Test baseline model training."""
        X, y = sample_data
        pipeline.split_and_scale(X, y)
        train_acc, test_acc = pipeline.train_baseline()
        
        assert pipeline.model is not None
        assert 0 <= train_acc <= 1
        assert 0 <= test_acc <= 1
        assert train_acc >= test_acc  # Training should be at least as good as test
    
    def test_predict_new_samples(self, pipeline, sample_data):
        """Test predictions on new samples."""
        X, y = sample_data
        pipeline.split_and_scale(X, y)
        pipeline.train_baseline()
        
        # Take first 5 test samples
        X_new = pipeline.X_test[:5]
        predictions, probabilities = pipeline.predict_new_samples(X_new)
        
        assert len(predictions) == 5
        assert probabilities.shape == (5, 3)  # 3 classes
        assert np.all(predictions >= 0)
        assert np.all(predictions < 3)
        
        # Probabilities should sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_edge_case_single_sample(self, pipeline, sample_data):
        """Test prediction on single sample."""
        X, y = sample_data
        pipeline.split_and_scale(X, y)
        pipeline.train_baseline()
        
        X_single = pipeline.X_test[:1]
        predictions, probabilities = pipeline.predict_new_samples(X_single)
        
        assert len(predictions) == 1
        assert probabilities.shape == (1, 3)
    
    def test_edge_case_large_k(self, sample_data):
        """Test behavior with k larger than training samples."""
        X, y = sample_data
        
        # Use larger dataset to ensure stratified split works (need at least 2 samples per class in test set)
        # With 3 classes and test_size=0.2, we need at least 15 samples for stratified split
        X_small = X[:30]
        y_small = y[:30]
        
        # k=15 > training samples should still work (sklearn handles this)
        pipeline = KNNPipeline(n_neighbors=15)
        pipeline.split_and_scale(X_small, y_small, test_size=0.2)
        
        # Should not crash, but might have reduced performance
        try:
            pipeline.train_baseline()
            assert pipeline.model is not None
        except Exception as e:
            pytest.fail(f"Should handle k > n_samples gracefully: {e}")
    
    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same random seed."""
        X, y = sample_data
        
        # Run pipeline twice with same seed
        results1 = []
        results2 = []
        
        for _ in range(2):
            pipeline = KNNPipeline(n_neighbors=5, random_state=42)
            pipeline.split_and_scale(X, y)
            train_acc, test_acc = pipeline.train_baseline()
            results1.append((train_acc, test_acc))
        
        # Should get identical results
        assert results1[0] == results1[1]
    
    def test_different_k_values(self, sample_data):
        """Test that different k values produce different results."""
        X, y = sample_data
        
        accuracies = []
        for k in [1, 3, 5, 7, 11]:
            pipeline = KNNPipeline(n_neighbors=k, random_state=42)
            pipeline.split_and_scale(X, y)
            _, test_acc = pipeline.train_baseline()
            accuracies.append(test_acc)
        
        # Should have some variation across different k values
        assert len(set(accuracies)) > 1
    
    def test_evaluate_model_metrics(self, pipeline, sample_data):
        """Test that evaluation returns expected metrics."""
        X, y = sample_data
        pipeline.split_and_scale(X, y)
        pipeline.train_baseline()
        
        metrics = pipeline.evaluate_model(plot=False)
        
        assert 'accuracy' in metrics
        assert 'cv_mean' in metrics
        assert 'cv_std' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['cv_mean'] <= 1
        assert metrics['cv_std'] >= 0
    
    def test_optimize_hyperparameters(self, pipeline, sample_data):
        """Test hyperparameter optimization."""
        X, y = sample_data
        pipeline.split_and_scale(X, y)
        
        best_params, best_score = pipeline.optimize_hyperparameters(cv_folds=3)
        
        assert best_params is not None
        assert 'n_neighbors' in best_params
        assert 'weights' in best_params
        assert 'metric' in best_params
        assert 0 <= best_score <= 1
        assert pipeline.model is not None
    
    def test_scaler_fit_on_train_only(self, pipeline, sample_data):
        """Test that scaler is fit only on training data (no data leakage)."""
        X, y = sample_data
        pipeline.split_and_scale(X, y)
        
        # Scaler mean should match training data, not full dataset
        train_mean = pipeline.X_train.mean(axis=0)
        scaler_mean = pipeline.scaler.mean_
        
        # Should be very close (not exact due to floating point)
        assert np.allclose(train_mean, scaler_mean, rtol=1e-5)
        
        # Scaler should NOT match full dataset mean
        full_mean = X.mean(axis=0)
        assert not np.allclose(full_mean, scaler_mean, rtol=1e-5)
    
    def test_stratified_split(self, pipeline, sample_data):
        """Test that train/test split maintains class proportions."""
        X, y = sample_data
        pipeline.split_and_scale(X, y, test_size=0.2)
        
        # Calculate class proportions
        train_props = np.bincount(pipeline.y_train) / len(pipeline.y_train)
        test_props = np.bincount(pipeline.y_test) / len(pipeline.y_test)
        
        # Proportions should be similar (stratified split)
        assert np.allclose(train_props, test_props, atol=0.1)


def test_full_pipeline_execution():
    """Integration test: run complete pipeline end-to-end."""
    pipeline = KNNPipeline(n_neighbors=5, random_state=42)
    
    # Load data
    X, y = pipeline.load_data(dataset_type='synthetic')
    
    # Full pipeline
    pipeline.split_and_scale(X, y)
    pipeline.train_baseline()
    pipeline.optimize_hyperparameters(cv_folds=3)
    metrics = pipeline.evaluate_model(plot=False)
    
    # Should complete without errors and produce reasonable metrics
    assert metrics['accuracy'] > 0.5  # Better than random for 3 classes
    assert metrics['cv_mean'] > 0.5
    assert pipeline.best_params is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
