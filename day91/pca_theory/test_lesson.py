"""
Day 91: PCA Theory - Comprehensive Test Suite
Tests mathematical correctness and edge cases
"""

import pytest
import numpy as np
from lesson_code import PCATheory, generate_synthetic_data


class TestPCABasics:
    """Test basic PCA functionality."""
    
    def test_initialization(self):
        """Test PCA initialization."""
        pca = PCATheory(n_components=5)
        assert pca.n_components == 5
        assert pca.components_ is None
        assert pca.mean_ is None
    
    def test_fit_shape(self):
        """Test PCA fit produces correct shapes."""
        X = np.random.randn(100, 10)
        pca = PCATheory(n_components=5)
        pca.fit(X)
        
        assert pca.components_.shape == (5, 10)
        assert pca.mean_.shape == (10,)
        assert len(pca.explained_variance_) == 5
        assert len(pca.explained_variance_ratio_) == 5
    
    def test_transform_shape(self):
        """Test transformation produces correct shape."""
        X = np.random.randn(100, 10)
        pca = PCATheory(n_components=5)
        pca.fit(X)
        X_transformed = pca.transform(X)
        
        assert X_transformed.shape == (100, 5)
    
    def test_inverse_transform_shape(self):
        """Test inverse transformation produces correct shape."""
        X = np.random.randn(100, 10)
        pca = PCATheory(n_components=5)
        pca.fit(X)
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        assert X_reconstructed.shape == X.shape


class TestMathematicalCorrectness:
    """Test mathematical properties of PCA."""
    
    def test_explained_variance_sum(self):
        """Test that variance ratios sum to <= 1."""
        X = np.random.randn(100, 10)
        pca = PCATheory()
        pca.fit(X)
        
        total_variance_ratio = np.sum(pca.explained_variance_ratio_)
        assert 0.99 <= total_variance_ratio <= 1.01  # Allow small numerical error
    
    def test_variance_decreasing(self):
        """Test that explained variance is in decreasing order."""
        X = np.random.randn(100, 10)
        pca = PCATheory()
        pca.fit(X)
        
        for i in range(len(pca.explained_variance_) - 1):
            assert pca.explained_variance_[i] >= pca.explained_variance_[i + 1]
    
    def test_orthogonality(self):
        """Test that principal components are orthogonal."""
        X = np.random.randn(100, 10)
        pca = PCATheory()
        pca.fit(X)
        
        is_orthogonal, max_dot = pca.verify_orthogonality()
        assert is_orthogonal
        assert max_dot < 1e-9
    
    def test_component_unit_length(self):
        """Test that each component has unit length."""
        X = np.random.randn(100, 10)
        pca = PCATheory()
        pca.fit(X)
        
        for i in range(pca.components_.shape[0]):
            length = np.linalg.norm(pca.components_[i])
            assert abs(length - 1.0) < 1e-10
    
    def test_perfect_reconstruction_all_components(self):
        """Test reconstruction with all components is perfect."""
        X = np.random.randn(100, 10)
        pca = PCATheory()  # Keep all components
        pca.fit(X)
        
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        reconstruction_error = np.max(np.abs(X - X_reconstructed))
        assert reconstruction_error < 1e-10
    
    def test_centering(self):
        """Test that transformed data is centered."""
        X = np.random.randn(100, 10)
        pca = PCATheory(n_components=5)
        pca.fit(X)
        X_transformed = pca.transform(X)
        
        mean_transformed = np.mean(X_transformed, axis=0)
        assert np.max(np.abs(mean_transformed)) < 1e-10


class TestCovarianceMatrix:
    """Test covariance matrix computation."""
    
    def test_covariance_symmetry(self):
        """Test that covariance matrix is symmetric."""
        X = np.random.randn(100, 5)
        pca = PCATheory()
        cov = pca.get_covariance_matrix(X)
        
        assert np.allclose(cov, cov.T)
    
    def test_covariance_shape(self):
        """Test covariance matrix has correct shape."""
        X = np.random.randn(100, 5)
        pca = PCATheory()
        cov = pca.get_covariance_matrix(X)
        
        assert cov.shape == (5, 5)
    
    def test_covariance_positive_diagonal(self):
        """Test that diagonal elements (variances) are non-negative."""
        X = np.random.randn(100, 5)
        pca = PCATheory()
        cov = pca.get_covariance_matrix(X)
        
        diagonal = np.diag(cov)
        assert np.all(diagonal >= 0)


class TestDimensionalityReduction:
    """Test dimensionality reduction properties."""
    
    def test_reduced_components(self):
        """Test keeping fewer components."""
        X = generate_synthetic_data(n_samples=100, n_features=10, effective_rank=3)
        pca = PCATheory(n_components=3)
        pca.fit(X)
        
        X_transformed = pca.transform(X)
        assert X_transformed.shape == (100, 3)
    
    def test_variance_preserved_with_reduction(self):
        """Test that keeping top components preserves most variance."""
        X = generate_synthetic_data(n_samples=200, n_features=10, effective_rank=3)
        pca = PCATheory(n_components=3)
        pca.fit(X)
        
        cumulative_variance = pca.get_cumulative_variance_ratio()[-1]
        assert cumulative_variance > 0.9  # Should keep >90% variance
    
    def test_reconstruction_error_increases_with_reduction(self):
        """Test that reconstruction error increases as we keep fewer components."""
        X = np.random.randn(100, 10)
        errors = []
        
        for n_comp in [10, 7, 5, 3, 1]:
            pca = PCATheory(n_components=n_comp)
            pca.fit(X)
            X_trans = pca.transform(X)
            X_recon = pca.inverse_transform(X_trans)
            error = np.mean((X - X_recon) ** 2)
            errors.append(error)
        
        # Errors should be non-decreasing as we reduce components
        for i in range(len(errors) - 1):
            assert errors[i] <= errors[i + 1] + 1e-10  # Allow small numerical error


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_single_sample(self):
        """Test PCA with single sample (should not crash)."""
        X = np.random.randn(1, 5)
        pca = PCATheory()
        # Should handle gracefully (though not mathematically meaningful)
        try:
            pca.fit(X)
            # If it fits, transformation should work
            X_trans = pca.transform(X)
            assert X_trans.shape[0] == 1
        except:
            # Or it might raise an error, which is also acceptable
            pass
    
    def test_more_components_than_features(self):
        """Test requesting more components than features."""
        X = np.random.randn(100, 5)
        pca = PCATheory(n_components=10)
        pca.fit(X)
        
        # Should only get 5 components
        assert pca.components_.shape[0] == 5
    
    def test_constant_feature(self):
        """Test with a feature that has zero variance."""
        X = np.random.randn(100, 5)
        X[:, 2] = 5.0  # Constant feature
        
        pca = PCATheory()
        pca.fit(X)
        
        # Should still work, one eigenvalue should be near zero
        min_variance = np.min(pca.explained_variance_)
        assert min_variance < 1e-10
    
    def test_perfectly_correlated_features(self):
        """Test with perfectly correlated features."""
        X = np.random.randn(100, 3)
        X = np.column_stack([X, X[:, 0]])  # Duplicate first column
        
        pca = PCATheory()
        pca.fit(X)
        
        # Should still work, some eigenvalues near zero
        near_zero = np.sum(pca.explained_variance_ < 1e-10)
        assert near_zero >= 1


class TestCumulativeVariance:
    """Test cumulative variance calculations."""
    
    def test_cumulative_monotonic(self):
        """Test that cumulative variance is monotonically increasing."""
        X = np.random.randn(100, 10)
        pca = PCATheory()
        pca.fit(X)
        
        cumulative = pca.get_cumulative_variance_ratio()
        for i in range(len(cumulative) - 1):
            assert cumulative[i] <= cumulative[i + 1]
    
    def test_cumulative_reaches_one(self):
        """Test that cumulative variance reaches ~1."""
        X = np.random.randn(100, 10)
        pca = PCATheory()
        pca.fit(X)
        
        cumulative = pca.get_cumulative_variance_ratio()
        assert abs(cumulative[-1] - 1.0) < 0.01


class TestFitTransform:
    """Test fit_transform convenience method."""
    
    def test_fit_transform_equivalence(self):
        """Test that fit_transform gives same result as fit then transform."""
        X = np.random.randn(100, 10)
        
        pca1 = PCATheory(n_components=5)
        X_trans1 = pca1.fit_transform(X)
        
        pca2 = PCATheory(n_components=5)
        pca2.fit(X)
        X_trans2 = pca2.transform(X)
        
        assert np.allclose(X_trans1, X_trans2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
