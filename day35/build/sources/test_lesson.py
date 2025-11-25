"""
Test suite for Day 35: Data Cleaning and Handling Missing Data
Verifies all cleaning strategies work correctly.
"""

import pytest
import pandas as pd
import numpy as np
from lesson_code import (
    MissingDataDetector,
    DataCleaner,
    generate_messy_data
)


class TestMissingDataDetector:
    """Test the missing data detection functionality."""
    
    def test_detector_initialization(self):
        """Test detector can be initialized with a DataFrame."""
        df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
        detector = MissingDataDetector(df)
        assert detector.df is not None
        
    def test_generate_report(self):
        """Test report generation shows correct missing counts."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [np.nan, np.nan, 3, 4],
            'C': [1, 2, 3, 4]
        })
        detector = MissingDataDetector(df)
        report = detector.generate_report()
        
        assert len(report) == 3  # Three columns
        assert report[report['column'] == 'A']['missing_count'].values[0] == 1
        assert report[report['column'] == 'B']['missing_count'].values[0] == 2
        assert report[report['column'] == 'C']['missing_count'].values[0] == 0
        
    def test_report_percentages(self):
        """Test report calculates correct percentages."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, np.nan],  # 50% missing
            'B': [1, 2, 3, 4]  # 0% missing
        })
        detector = MissingDataDetector(df)
        report = detector.generate_report()
        
        assert report[report['column'] == 'A']['missing_pct'].values[0] == 50.0
        assert report[report['column'] == 'B']['missing_pct'].values[0] == 0.0
        
    def test_strategy_recommendations(self):
        """Test appropriate strategies are recommended."""
        df = pd.DataFrame({
            'mostly_missing': [np.nan] * 8 + [1, 2],  # 80% missing
            'few_missing': [1, 2, np.nan, 4, 5],  # 20% missing
            'no_missing': [1, 2, 3, 4, 5]
        })
        detector = MissingDataDetector(df)
        report = detector.generate_report()
        
        mostly = report[report['column'] == 'mostly_missing']['recommended_strategy'].values[0]
        few = report[report['column'] == 'few_missing']['recommended_strategy'].values[0]
        none = report[report['column'] == 'no_missing']['recommended_strategy'].values[0]
        
        assert 'drop' in mostly.lower()
        assert none == "None needed"


class TestDataCleaner:
    """Test all data cleaning operations."""
    
    def test_cleaner_initialization(self):
        """Test cleaner creates a copy of the data."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        cleaner = DataCleaner(df)
        assert cleaner.original_shape == (3, 1)
        
    def test_drop_high_missing_columns(self):
        """Test columns with high missingness are dropped."""
        df = pd.DataFrame({
            'keep': [1, 2, 3, 4, 5],
            'drop': [np.nan] * 4 + [1]  # 80% missing
        })
        cleaner = DataCleaner(df)
        result = cleaner.drop_high_missing_columns(threshold=0.7).get_cleaned_data()
        
        assert 'keep' in result.columns
        assert 'drop' not in result.columns
        
    def test_drop_missing_target(self):
        """Test rows with missing target are dropped."""
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4],
            'target': [1, np.nan, 0, 1]
        })
        cleaner = DataCleaner(df)
        result = cleaner.drop_rows_with_missing_target('target').get_cleaned_data()
        
        assert len(result) == 3  # One row dropped
        assert result['target'].isna().sum() == 0
        
    def test_fill_numeric_mean(self):
        """Test mean imputation works correctly."""
        df = pd.DataFrame({
            'values': [1.0, 2.0, np.nan, 4.0]
        })
        cleaner = DataCleaner(df)
        result = cleaner.fill_numeric_mean(['values']).get_cleaned_data()
        
        # Mean of [1, 2, 4] is 2.33...
        assert result['values'].isna().sum() == 0
        assert abs(result['values'].iloc[2] - 2.333) < 0.01
        
    def test_fill_numeric_median(self):
        """Test median imputation works correctly."""
        df = pd.DataFrame({
            'values': [1.0, 2.0, np.nan, 100.0]  # Outlier present
        })
        cleaner = DataCleaner(df)
        result = cleaner.fill_numeric_median(['values']).get_cleaned_data()
        
        # Median of [1, 2, 100] is 2
        assert result['values'].isna().sum() == 0
        assert result['values'].iloc[2] == 2.0
        
    def test_fill_categorical_mode(self):
        """Test mode imputation works correctly."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', np.nan, 'A']
        })
        cleaner = DataCleaner(df)
        result = cleaner.fill_categorical_mode(['category']).get_cleaned_data()
        
        assert result['category'].isna().sum() == 0
        assert result['category'].iloc[3] == 'A'  # Most common value
        
    def test_fill_forward(self):
        """Test forward fill works correctly."""
        df = pd.DataFrame({
            'time_series': [1, np.nan, np.nan, 4, np.nan]
        })
        cleaner = DataCleaner(df)
        result = cleaner.fill_forward(['time_series']).get_cleaned_data()
        
        # Values should be: [1, 1, 1, 4, 4]
        assert result['time_series'].iloc[1] == 1
        assert result['time_series'].iloc[2] == 1
        assert result['time_series'].iloc[4] == 4
        
    def test_fill_backward(self):
        """Test backward fill works correctly."""
        df = pd.DataFrame({
            'time_series': [np.nan, np.nan, 3, 4, np.nan]
        })
        cleaner = DataCleaner(df)
        result = cleaner.fill_backward(['time_series']).get_cleaned_data()
        
        # Values should be: [3, 3, 3, 4, nan] (last one stays nan)
        assert result['time_series'].iloc[0] == 3
        assert result['time_series'].iloc[1] == 3
        
    def test_fill_constant(self):
        """Test constant fill works correctly."""
        df = pd.DataFrame({
            'col1': [1, np.nan, 3],
            'col2': ['A', np.nan, 'C']
        })
        cleaner = DataCleaner(df)
        result = cleaner.fill_constant({'col1': 0, 'col2': 'Unknown'}).get_cleaned_data()
        
        assert result['col1'].iloc[1] == 0
        assert result['col2'].iloc[1] == 'Unknown'
        
    def test_chaining_operations(self):
        """Test multiple cleaning operations can be chained."""
        df = pd.DataFrame({
            'numeric': [1, np.nan, 3],
            'category': ['A', np.nan, 'A']
        })
        
        cleaner = DataCleaner(df)
        result = (cleaner
                  .fill_numeric_mean(['numeric'])
                  .fill_categorical_mode(['category'])
                  .get_cleaned_data())
        
        assert result['numeric'].isna().sum() == 0
        assert result['category'].isna().sum() == 0
        
    def test_validation_report(self):
        """Test validation report is generated correctly."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [4, 5, np.nan]
        })
        
        cleaner = DataCleaner(df)
        cleaner.fill_numeric_mean()
        validation = cleaner.validate_cleaning()
        
        assert validation['original_shape'] == (3, 2)
        assert validation['remaining_missing_values'] == 0
        assert validation['is_clean'] == True
        assert len(validation['cleaning_log']) > 0


class TestGeneratMessyData:
    """Test the messy data generator."""
    
    def test_generates_correct_shape(self):
        """Test messy data has expected dimensions."""
        df = generate_messy_data()
        assert df.shape[0] == 1000
        assert df.shape[1] == 8
        
    def test_has_missing_values(self):
        """Test generated data actually has missing values."""
        df = generate_messy_data()
        total_missing = df.isna().sum().sum()
        assert total_missing > 0
        
    def test_has_expected_columns(self):
        """Test all expected columns are present."""
        df = generate_messy_data()
        expected_columns = [
            'user_id', 'age', 'income', 'session_duration',
            'country', 'subscription_type', 'last_login_days', 'conversion'
        ]
        for col in expected_columns:
            assert col in df.columns


class TestProductionPipeline:
    """Test complete end-to-end pipeline."""
    
    def test_full_pipeline(self):
        """Test complete cleaning pipeline runs without errors."""
        df = generate_messy_data()
        
        cleaner = DataCleaner(df)
        result = (cleaner
                  .drop_high_missing_columns(threshold=0.7)
                  .fill_numeric_median()
                  .fill_categorical_mode()
                  .get_cleaned_data())
        
        # Should have significantly fewer missing values
        original_missing = df.isna().sum().sum()
        cleaned_missing = result.isna().sum().sum()
        assert cleaned_missing < original_missing
        
    def test_pipeline_preserves_data_types(self):
        """Test cleaning doesn't corrupt data types."""
        df = generate_messy_data()
        
        cleaner = DataCleaner(df)
        result = cleaner.fill_numeric_median().fill_categorical_mode().get_cleaned_data()
        
        # Check numeric columns are still numeric
        assert pd.api.types.is_numeric_dtype(result['age'])
        assert pd.api.types.is_numeric_dtype(result['income'])
        
        # Check categorical columns are still object type
        assert result['country'].dtype == 'object'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
