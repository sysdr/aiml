"""
Tests for Day 28: Correlation and Covariance
Run with: pytest test_lesson.py
"""

import pytest
import numpy as np
from lesson_code import FeatureAnalyzer, generate_sample_data


class TestFeatureAnalyzer:
    """Test suite for FeatureAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create simple test data."""
        np.random.seed(42)
        # Create 3 features with known relationships
        x = np.random.normal(0, 1, 100)
        y = 2 * x + np.random.normal(0, 0.1, 100)  # Strong positive correlation
        z = np.random.normal(0, 1, 100)  # Independent
        
        data = np.column_stack([x, y, z])
        feature_names = ['X', 'Y', 'Z']
        
        return data, feature_names
    
    def test_initialization(self, sample_data):
        """Test analyzer initialization."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        
        assert analyzer.n_samples == 100
        assert analyzer.n_features == 3
        assert analyzer.feature_names == names
    
    def test_covariance_calculation(self, sample_data):
        """Test covariance matrix calculation."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        
        # Calculate using our method
        cov_manual = analyzer.calculate_covariance_manual()
        
        # Compare with NumPy
        cov_numpy = np.cov(data, rowvar=False)
        
        assert np.allclose(cov_manual, cov_numpy), "Covariance calculation doesn't match NumPy"
        
        # Check shape
        assert cov_manual.shape == (3, 3)
        
        # Check symmetry
        assert np.allclose(cov_manual, cov_manual.T), "Covariance matrix should be symmetric"
    
    def test_correlation_calculation(self, sample_data):
        """Test correlation matrix calculation."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        
        # Calculate using our method
        corr_manual = analyzer.calculate_correlation_manual()
        
        # Compare with NumPy
        corr_numpy = np.corrcoef(data, rowvar=False)
        
        assert np.allclose(corr_manual, corr_numpy), "Correlation calculation doesn't match NumPy"
        
        # Check diagonal is all 1s (feature correlates perfectly with itself)
        assert np.allclose(np.diag(corr_manual), 1.0), "Diagonal should be 1.0"
        
        # Check all values are between -1 and 1
        assert np.all(corr_manual >= -1.0) and np.all(corr_manual <= 1.0), \
            "Correlation values should be between -1 and 1"
    
    def test_high_correlation_detection(self, sample_data):
        """Test finding highly correlated features."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        analyzer.calculate_correlation_manual()
        
        # X and Y should be highly correlated (we created them that way)
        high_corr = analyzer.find_highly_correlated_features(threshold=0.8)
        
        # Should find at least one pair
        assert len(high_corr) > 0, "Should detect high correlation between X and Y"
        
        # Check format
        for feat1, feat2, corr in high_corr:
            assert isinstance(feat1, str)
            assert isinstance(feat2, str)
            assert abs(corr) >= 0.8
    
    def test_feature_removal_suggestions(self, sample_data):
        """Test feature removal suggestions."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        analyzer.calculate_correlation_manual()
        
        suggestions = analyzer.suggest_features_to_remove(threshold=0.95)
        
        # Check return format
        assert 'keep' in suggestions
        assert 'remove' in suggestions
        assert isinstance(suggestions['keep'], list)
        assert isinstance(suggestions['remove'], list)
        
        # Total should equal original features
        total_features = len(suggestions['keep']) + len(suggestions['remove'])
        assert total_features == 3
    
    def test_report_generation(self, sample_data):
        """Test report generation."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        
        report = analyzer.generate_report()
        
        # Check report is non-empty string
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Check key sections are present
        assert "FEATURE RELATIONSHIP ANALYSIS REPORT" in report
        assert "CORRELATION SUMMARY" in report


class TestSampleDataGeneration:
    """Test sample data generation."""
    
    def test_generate_sample_data(self):
        """Test that sample data generation works."""
        data, feature_names = generate_sample_data()
        
        # Check shape
        assert data.shape == (200, 5), "Should have 200 samples and 5 features"
        
        # Check feature names
        assert len(feature_names) == 5
        assert all(isinstance(name, str) for name in feature_names)
        
        # Check data is numeric
        assert np.isfinite(data).all(), "All data should be finite"
        
        # Check reasonable value ranges
        time_spent = data[:, 0]
        assert time_spent.min() > 0, "Time spent should be positive"
        
        scroll_depth = data[:, 1]
        assert 0 <= scroll_depth.min() and scroll_depth.max() <= 100, \
            "Scroll depth should be between 0 and 100"


class TestCorrelationProperties:
    """Test mathematical properties of correlation."""
    
    def test_perfect_correlation(self):
        """Test perfect positive and negative correlation."""
        # Perfect positive correlation
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x  # Perfect linear relationship
        data = np.column_stack([x, y])
        
        analyzer = FeatureAnalyzer(data, ['X', 'Y'])
        corr = analyzer.calculate_correlation_manual()
        
        assert np.isclose(corr[0, 1], 1.0, atol=1e-10), \
            "Perfect positive relationship should have correlation = 1.0"
        
        # Perfect negative correlation
        z = -2 * x
        data = np.column_stack([x, z])
        
        analyzer = FeatureAnalyzer(data, ['X', 'Z'])
        corr = analyzer.calculate_correlation_manual()
        
        assert np.isclose(corr[0, 1], -1.0, atol=1e-10), \
            "Perfect negative relationship should have correlation = -1.0"
    
    def test_zero_correlation(self):
        """Test uncorrelated data."""
        np.random.seed(42)
        
        # Create independent random variables
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 1, 1000)
        data = np.column_stack([x, y])
        
        analyzer = FeatureAnalyzer(data, ['X', 'Y'])
        corr = analyzer.calculate_correlation_manual()
        
        # With enough samples, correlation should be near zero
        assert abs(corr[0, 1]) < 0.1, \
            "Independent variables should have correlation near 0"


# Run tests with verbose output
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
