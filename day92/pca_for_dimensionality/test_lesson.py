"""
Comprehensive tests for Day 92: PCA for Dimensionality Reduction
Tests cover production scenarios, edge cases, and numerical correctness
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from lesson_code import ProductionPCA, run_pca_dimensionality_reduction
import os


class TestProductionPCA:
    """Test suite for ProductionPCA class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        X, y = make_classification(
            n_samples=200,
            n_features=50,
            n_informative=20,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    def test_initialization(self):
        """Test PCA pipeline initialization."""
        pca = ProductionPCA(variance_threshold=0.95)
        assert pca.variance_threshold == 0.95
        assert pca.random_state == 42
        assert pca.pca is None
        assert pca.n_components_optimal is None
    
    def test_fit_basic(self, sample_data):
        """Test basic fitting functionality."""
        X_train, X_test, y_train, y_test = sample_data
        pca = ProductionPCA(variance_threshold=0.95)
        pca.fit(X_train)
        
        assert pca.pca is not None
        assert pca.n_components_optimal is not None
        assert pca.n_components_optimal <= X_train.shape[1]
        assert pca.fit_time is not None
        assert pca.fit_time > 0
    
    def test_transform_basic(self, sample_data):
        """Test basic transformation."""
        X_train, X_test, y_train, y_test = sample_data
        pca = ProductionPCA(variance_threshold=0.95)
        pca.fit(X_train)
        
        X_reduced = pca.transform(X_test)
        assert X_reduced.shape[0] == X_test.shape[0]
        assert X_reduced.shape[1] == pca.n_components_optimal
        assert pca.transform_time is not None
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X_train, X_test, y_train, y_test = sample_data
        pca = ProductionPCA(variance_threshold=0.95)
        
        X_reduced = pca.fit_transform(X_train)
        assert X_reduced.shape[0] == X_train.shape[0]
        assert X_reduced.shape[1] == pca.n_components_optimal
    
    def test_transform_before_fit_raises_error(self, sample_data):
        """Test that transforming before fitting raises error."""
        X_train, X_test, y_train, y_test = sample_data
        pca = ProductionPCA()
        
        with pytest.raises(ValueError, match="Must fit PCA before transforming"):
            pca.transform(X_train)
    
    def test_variance_threshold_respected(self, sample_data):
        """Test that variance threshold is respected."""
        X_train, X_test, y_train, y_test = sample_data
        threshold = 0.90
        pca = ProductionPCA(variance_threshold=threshold)
        pca.fit(X_train)
        
        cumulative_variance = pca.get_cumulative_variance()
        assert cumulative_variance[-1] >= threshold
    
    def test_different_variance_thresholds(self, sample_data):
        """Test that different thresholds produce different components."""
        X_train, X_test, y_train, y_test = sample_data
        
        pca_90 = ProductionPCA(variance_threshold=0.90).fit(X_train)
        pca_95 = ProductionPCA(variance_threshold=0.95).fit(X_train)
        pca_99 = ProductionPCA(variance_threshold=0.99).fit(X_train)
        
        # Higher threshold should need more components
        assert pca_90.n_components_optimal <= pca_95.n_components_optimal
        assert pca_95.n_components_optimal <= pca_99.n_components_optimal
    
    def test_inverse_transform(self, sample_data):
        """Test inverse transformation."""
        X_train, X_test, y_train, y_test = sample_data
        pca = ProductionPCA(variance_threshold=0.95)
        pca.fit(X_train)
        
        X_reduced = pca.transform(X_test)
        X_reconstructed = pca.inverse_transform(X_reduced)
        
        assert X_reconstructed.shape == X_test.shape
    
    def test_reconstruction_error(self, sample_data):
        """Test reconstruction error calculation."""
        X_train, X_test, y_train, y_test = sample_data
        pca = ProductionPCA(variance_threshold=0.95)
        pca.fit(X_train)
        
        error = pca.get_reconstruction_error(X_test)
        assert error >= 0
        assert isinstance(error, float)
    
    def test_reconstruction_error_increases_with_compression(self, sample_data):
        """Test that higher compression increases reconstruction error."""
        X_train, X_test, y_train, y_test = sample_data
        
        pca_high = ProductionPCA(variance_threshold=0.99).fit(X_train)
        pca_low = ProductionPCA(variance_threshold=0.80).fit(X_train)
        
        error_high = pca_high.get_reconstruction_error(X_test)
        error_low = pca_low.get_reconstruction_error(X_test)
        
        # Lower variance threshold (more compression) should have higher error
        assert error_low >= error_high
    
    def test_explained_variance_ratio(self, sample_data):
        """Test explained variance ratio retrieval."""
        X_train, X_test, y_train, y_test = sample_data
        pca = ProductionPCA(variance_threshold=0.95)
        pca.fit(X_train)
        
        var_ratio = pca.get_explained_variance_ratio()
        assert len(var_ratio) == pca.n_components_optimal
        assert np.all(var_ratio >= 0)
        assert np.all(var_ratio <= 1)
        # Should be sorted descending
        assert np.all(var_ratio[:-1] >= var_ratio[1:])
    
    def test_cumulative_variance(self, sample_data):
        """Test cumulative variance calculation."""
        X_train, X_test, y_train, y_test = sample_data
        pca = ProductionPCA(variance_threshold=0.95)
        pca.fit(X_train)
        
        cum_var = pca.get_cumulative_variance()
        assert len(cum_var) == pca.n_components_optimal
        # Should be monotonically increasing
        assert np.all(cum_var[1:] >= cum_var[:-1])
        # Last value should be close to or exceed threshold
        assert cum_var[-1] >= 0.95
    
    def test_component_loadings(self, sample_data):
        """Test component loadings retrieval."""
        X_train, X_test, y_train, y_test = sample_data
        pca = ProductionPCA(variance_threshold=0.95)
        pca.fit(X_train)
        
        loadings = pca.get_component_loadings()
        assert len(loadings) == pca.n_components_optimal
        assert 'PC1' in loadings
        assert len(loadings['PC1']) == 5  # Top 5 features
    
    def test_save_and_load(self, sample_data, tmp_path):
        """Test model persistence."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Fit and save
        pca_original = ProductionPCA(variance_threshold=0.95)
        pca_original.fit(X_train)
        
        filepath = tmp_path / "test_pca.pkl"
        pca_original.save(str(filepath))
        
        # Load and verify
        pca_loaded = ProductionPCA.load(str(filepath))
        
        assert pca_loaded.n_components_optimal == pca_original.n_components_optimal
        assert pca_loaded.variance_threshold == pca_original.variance_threshold
        
        # Verify transformations are identical
        X_reduced_original = pca_original.transform(X_test)
        X_reduced_loaded = pca_loaded.transform(X_test)
        
        np.testing.assert_array_almost_equal(X_reduced_original, X_reduced_loaded)
    
    def test_single_component(self, sample_data):
        """Test PCA with extremely low variance threshold (edge case)."""
        X_train, X_test, y_train, y_test = sample_data
        pca = ProductionPCA(variance_threshold=0.01)
        pca.fit(X_train)
        
        # Should select at least 1 component
        assert pca.n_components_optimal >= 1
        
        X_reduced = pca.transform(X_test)
        assert X_reduced.shape[1] >= 1
    
    def test_all_components(self, sample_data):
        """Test PCA with very high variance threshold."""
        X_train, X_test, y_train, y_test = sample_data
        pca = ProductionPCA(variance_threshold=0.9999)
        pca.fit(X_train)
        
        # Might need all or most components
        assert pca.n_components_optimal <= X_train.shape[1]
    
    def test_consistency_across_runs(self, sample_data):
        """Test that results are consistent with fixed random state."""
        X_train, X_test, y_train, y_test = sample_data
        
        pca1 = ProductionPCA(variance_threshold=0.95, random_state=42)
        pca2 = ProductionPCA(variance_threshold=0.95, random_state=42)
        
        pca1.fit(X_train)
        pca2.fit(X_train)
        
        X_reduced1 = pca1.transform(X_test)
        X_reduced2 = pca2.transform(X_test)
        
        np.testing.assert_array_almost_equal(X_reduced1, X_reduced2)


class TestProductionScenarios:
    """Test production deployment scenarios."""
    
    def test_mnist_compression(self):
        """Test PCA on MNIST digits (real-world scenario)."""
        digits = load_digits()
        X, y = digits.data, digits.target
        
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        
        pca = ProductionPCA(variance_threshold=0.95)
        pca.fit(X_train)
        
        # Should achieve significant compression
        compression_ratio = X.shape[1] / pca.n_components_optimal
        assert compression_ratio > 1.5
        
        # Should preserve most variance
        assert pca.get_cumulative_variance()[-1] >= 0.95
    
    def test_batch_processing(self):
        """Test processing multiple batches (production pattern)."""
        pca = ProductionPCA(variance_threshold=0.95)
        
        # Fit on initial batch
        X_batch1, _ = make_classification(n_samples=100, n_features=50, random_state=42)
        pca.fit(X_batch1)
        
        # Transform subsequent batches
        results = []
        for i in range(5):
            X_batch, _ = make_classification(n_samples=100, n_features=50, random_state=100+i)
            X_reduced = pca.transform(X_batch)
            results.append(X_reduced)
        
        # All batches should have same reduced dimensionality
        dims = [r.shape[1] for r in results]
        assert len(set(dims)) == 1  # All same
        assert dims[0] == pca.n_components_optimal
    
    def test_performance_benchmark(self):
        """Test that PCA performs within acceptable time bounds."""
        X, y = make_classification(n_samples=1000, n_features=100, random_state=42)
        
        pca = ProductionPCA(variance_threshold=0.95)
        pca.fit(X)
        
        # Fit should be reasonably fast
        assert pca.fit_time < 5.0  # seconds
        
        # Transform should be very fast
        X_reduced = pca.transform(X)
        assert pca.transform_time < 1.0  # seconds


class TestMainFunction:
    """Test main execution function."""
    
    def test_run_pca_dimensionality_reduction(self):
        """Test main function returns correct metrics."""
        metrics = run_pca_dimensionality_reduction(n_samples=200, n_features=50)
        
        required_keys = [
            'original_dims', 'reduced_dims', 'compression_ratio',
            'variance_preserved', 'reconstruction_error_train',
            'reconstruction_error_test', 'fit_time', 'transform_time'
        ]
        
        for key in required_keys:
            assert key in metrics
        
        assert metrics['original_dims'] == 50
        assert metrics['reduced_dims'] < 50
        assert metrics['compression_ratio'] > 1
        assert metrics['variance_preserved'] >= 0.95
        assert metrics['reconstruction_error_train'] >= 0
        assert metrics['reconstruction_error_test'] >= 0
    
    def test_run_with_different_sizes(self):
        """Test main function with different data sizes."""
        sizes = [(100, 20), (500, 100), (1000, 200)]
        
        for n_samples, n_features in sizes:
            metrics = run_pca_dimensionality_reduction(
                n_samples=n_samples,
                n_features=n_features
            )
            
            assert metrics['original_dims'] == n_features
            assert metrics['reduced_dims'] <= n_features
            assert 0 < metrics['variance_preserved'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
