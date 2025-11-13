"""
Tests for Day 30: ML Dataset Analyzer
"""

import pytest
import pandas as pd
import numpy as np
from lesson_code import MLDatasetAnalyzer, create_sample_datasets
import os

@pytest.fixture
def sample_dataframe():
    """Create a simple test dataframe"""
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.normal(50, 10, 200),
        'feature3': np.random.exponential(20, 200),
        'target': np.random.choice([0, 1], 200)
    }
    return pd.DataFrame(data)

@pytest.fixture
def analyzer(sample_dataframe):
    """Create analyzer instance"""
    return MLDatasetAnalyzer(sample_dataframe, target_column='target')

class TestMLDatasetAnalyzer:
    
    def test_initialization(self, sample_dataframe):
        """Test analyzer initialization"""
        analyzer = MLDatasetAnalyzer(sample_dataframe, target_column='target')
        
        assert analyzer.df.shape == sample_dataframe.shape
        assert analyzer.target_column == 'target'
        assert len(analyzer.numeric_features) == 3
        assert 'target' not in analyzer.numeric_features
    
    def test_profile_features(self, analyzer):
        """Test feature profiling"""
        profiles = analyzer.profile_features()
        
        assert len(profiles) == 3  # 3 numeric features
        
        for feature, profile in profiles.items():
            # Check all required statistics are present
            assert 'mean' in profile
            assert 'median' in profile
            assert 'std' in profile
            assert 'iqr' in profile
            assert 'outlier_count' in profile
            assert 'skewness' in profile
            
            # Check statistical properties
            assert profile['count'] > 0
            assert profile['std'] >= 0
            assert profile['iqr'] >= 0
    
    def test_detect_quality_issues(self, analyzer):
        """Test quality issue detection"""
        analyzer.profile_features()
        issues = analyzer.detect_quality_issues()
        
        assert 'high_missing' in issues
        assert 'high_outliers' in issues
        assert 'zero_variance' in issues
        assert 'highly_skewed' in issues
        assert 'imbalanced_target' in issues
        
        # All should be lists or None
        assert isinstance(issues['high_missing'], list)
        assert isinstance(issues['high_outliers'], list)
        assert isinstance(issues['zero_variance'], list)
        assert isinstance(issues['highly_skewed'], list)
    
    def test_analyze_correlations(self, analyzer):
        """Test correlation analysis"""
        corr_matrix, high_corr_pairs = analyzer.analyze_correlations(threshold=0.8)
        
        # Check correlation matrix shape
        assert corr_matrix.shape[0] == len(analyzer.numeric_features)
        assert corr_matrix.shape[1] == len(analyzer.numeric_features)
        
        # Check diagonal is 1.0 (self-correlation)
        for i in range(len(corr_matrix)):
            assert abs(corr_matrix.iloc[i, i] - 1.0) < 1e-10
        
        # Check high correlations format
        assert isinstance(high_corr_pairs, list)
        for pair in high_corr_pairs:
            assert len(pair) == 3  # (feature1, feature2, correlation)
            assert pair[2] > 0.8
    
    def test_test_normality(self, analyzer):
        """Test normality testing"""
        normality_results = analyzer.test_normality()
        
        assert len(normality_results) == len(analyzer.numeric_features)
        
        for feature, result in normality_results.items():
            assert 'p_value' in result
            assert 'is_normal' in result
            assert 'interpretation' in result
            assert 0 <= result['p_value'] <= 1
            assert isinstance(result['is_normal'], bool)
    
    def test_calculate_ml_readiness_score(self, analyzer):
        """Test ML readiness score calculation"""
        analyzer.profile_features()
        analyzer.detect_quality_issues()
        analyzer.analyze_correlations()
        
        readiness = analyzer.calculate_ml_readiness_score()
        
        assert 'score' in readiness
        assert 'interpretation' in readiness
        assert 'deductions' in readiness
        
        # Score should be between 0 and 100
        assert 0 <= readiness['score'] <= 100
        
        # Interpretation should be a string
        assert isinstance(readiness['interpretation'], str)
    
    def test_with_missing_data(self):
        """Test analyzer with missing data"""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, np.nan, np.nan, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        
        analyzer = MLDatasetAnalyzer(df, target_column='target')
        profiles = analyzer.profile_features()
        
        # Check missing percentages
        assert profiles['feature1']['missing_pct'] == 20.0  # 1 out of 5
        assert profiles['feature2']['missing_pct'] == 40.0  # 2 out of 5
    
    def test_with_outliers(self):
        """Test analyzer with outliers"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 100],  # 100 is outlier
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        
        analyzer = MLDatasetAnalyzer(df, target_column='target')
        profiles = analyzer.profile_features()
        
        # Feature1 should have at least one outlier
        assert profiles['feature1']['outlier_count'] > 0
    
    def test_sample_datasets_creation(self):
        """Test sample dataset creation"""
        create_sample_datasets()
        
        assert os.path.exists('clean_dataset.csv')
        assert os.path.exists('messy_dataset.csv')
        
        # Verify datasets can be loaded
        clean_df = pd.read_csv('clean_dataset.csv')
        messy_df = pd.read_csv('messy_dataset.csv')
        
        assert len(clean_df) == 1000
        assert len(messy_df) == 1000
        
        # Clean dataset should have no missing values
        assert clean_df.isnull().sum().sum() == 0
        
        # Messy dataset should have missing values
        assert messy_df.isnull().sum().sum() > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
