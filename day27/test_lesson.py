"""
Unit tests for Day 27: Variance and Standard Deviation
Run with: pytest test_lesson.py -v
"""

import pytest
import numpy as np
from lesson_code import DataQualityChecker


class TestVarianceCalculations:
    """Test variance and standard deviation calculations"""
    
    def test_sample_variance(self):
        """Test sample variance calculation (n-1)"""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        checker = DataQualityChecker(data)
        
        # Manual calculation
        mean = np.mean(data)
        variance = sum((x - mean)**2 for x in data) / (len(data) - 1)
        
        assert abs(checker.variance - variance) < 0.001
        assert abs(checker.variance - 4.571) < 0.01
    
    def test_population_variance(self):
        """Test population variance calculation (n)"""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        checker = DataQualityChecker(data)
        
        # Population variance (n)
        mean = np.mean(data)
        pop_var = sum((x - mean)**2 for x in data) / len(data)
        
        assert abs(checker.population_var - pop_var) < 0.001
        assert abs(checker.population_var - 4.0) < 0.01
    
    def test_standard_deviation(self):
        """Test standard deviation is sqrt of variance"""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        checker = DataQualityChecker(data)
        
        assert abs(checker.std - np.sqrt(checker.variance)) < 0.001
        assert abs(checker.std - 2.138) < 0.01
    
    def test_zero_variance(self):
        """Test data with no variance (all same values)"""
        data = [5, 5, 5, 5, 5]
        checker = DataQualityChecker(data)
        
        assert checker.variance == 0
        assert checker.std == 0


class TestOutlierDetection:
    """Test outlier detection using 3-sigma rule"""
    
    def test_no_outliers(self):
        """Test dataset with no outliers"""
        np.random.seed(42)
        data = np.random.normal(100, 10, 100)
        checker = DataQualityChecker(data)
        
        outliers, count, _ = checker.detect_outliers(threshold=3)
        assert count <= 3  # Should be ~0-3 outliers in normal data
    
    def test_with_outliers(self):
        """Test dataset with clear outliers"""
        data = [10, 11, 12, 13, 14, 15, 100, 200]  # Last two are outliers
        checker = DataQualityChecker(data)
        
        outliers, count, indices = checker.detect_outliers(threshold=2)
        assert count >= 1  # Should detect at least one outlier
        assert 100 in outliers or 200 in outliers
    
    def test_outlier_threshold(self):
        """Test different outlier thresholds"""
        data = [10, 11, 12, 13, 14, 15, 16, 30]
        checker = DataQualityChecker(data)
        
        _, count_2sigma, _ = checker.detect_outliers(threshold=2)
        _, count_3sigma, _ = checker.detect_outliers(threshold=3)
        
        # More permissive threshold should find fewer outliers
        assert count_3sigma <= count_2sigma


class TestCoefficientOfVariation:
    """Test coefficient of variation calculations"""
    
    def test_cv_calculation(self):
        """Test CV = (std/mean) * 100"""
        data = [10, 20, 30, 40, 50]
        checker = DataQualityChecker(data)
        
        expected_cv = (checker.std / checker.mean) * 100
        assert abs(checker.coefficient_of_variation() - expected_cv) < 0.001
    
    def test_cv_interpretation(self):
        """Test CV categories for ML readiness"""
        # Low variance data
        low_var = np.random.normal(100, 5, 100)
        checker_low = DataQualityChecker(low_var)
        assert checker_low.coefficient_of_variation() < 15
        
        # High variance data
        high_var = np.random.normal(100, 50, 100)
        checker_high = DataQualityChecker(high_var)
        assert checker_high.coefficient_of_variation() > 30
    
    def test_cv_zero_mean(self):
        """Test CV with zero mean (should return inf)"""
        data = [-5, -3, 0, 3, 5]
        checker = DataQualityChecker(data)
        cv = checker.coefficient_of_variation()
        # With zero mean, CV is undefined (inf)
        assert cv == float('inf') or abs(checker.mean) < 0.001


class TestIQRCalculation:
    """Test Interquartile Range calculations"""
    
    def test_iqr_basic(self):
        """Test IQR calculation"""
        data = list(range(1, 101))  # 1 to 100
        checker = DataQualityChecker(data)
        
        iqr, q25, q75 = checker.calculate_iqr()
        
        assert abs(q25 - 25.5) < 1
        assert abs(q75 - 75.5) < 1
        assert abs(iqr - 50) < 1
    
    def test_iqr_robustness(self):
        """Test IQR is robust to outliers"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]  # One extreme outlier
        checker = DataQualityChecker(data)
        
        iqr, q25, q75 = checker.calculate_iqr()
        
        # IQR should not be heavily affected by the outlier
        assert iqr < 10  # Should be around 4-5


class TestDataQualityReport:
    """Test comprehensive quality report"""
    
    def test_report_returns_dict(self):
        """Test quality report returns proper dictionary"""
        data = [10, 20, 30, 40, 50]
        checker = DataQualityChecker(data)
        
        result = checker.quality_report()
        
        assert isinstance(result, dict)
        assert 'mean' in result
        assert 'std' in result
        assert 'variance' in result
        assert 'cv' in result
        assert 'outliers' in result
    
    def test_report_accuracy(self):
        """Test report values match individual calculations"""
        data = [15, 20, 25, 30, 35]
        checker = DataQualityChecker(data)
        
        result = checker.quality_report()
        
        assert result['mean'] == checker.mean
        assert result['std'] == checker.std
        assert result['variance'] == checker.variance


def test_numpy_consistency():
    """Test our calculations match NumPy's built-in functions"""
    np.random.seed(42)
    data = np.random.normal(50, 15, 100)
    
    checker = DataQualityChecker(data)
    
    # Compare with NumPy
    assert abs(checker.mean - np.mean(data)) < 0.001
    assert abs(checker.variance - np.var(data, ddof=1)) < 0.001
    assert abs(checker.std - np.std(data, ddof=1)) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
