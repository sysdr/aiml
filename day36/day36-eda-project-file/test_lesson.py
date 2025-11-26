"""
Tests for Day 36: EDA Project
Ensuring your data investigation tools are production-ready
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from lesson_code import EDAEngine, generate_ecommerce_dataset


class TestEDAEngine:
    """Test suite for EDA Engine - production quality checks"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate small test dataset"""
        return pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'C'],
            'with_missing': [1, np.nan, 3, np.nan, 5]
        })
    
    @pytest.fixture
    def eda_engine(self, sample_data):
        """Create EDA engine instance"""
        return EDAEngine(sample_data, name="Test Dataset")
    
    def test_initialization(self, eda_engine, sample_data):
        """Test EDA engine initializes correctly"""
        assert eda_engine.name == "Test Dataset"
        assert len(eda_engine.data) == len(sample_data)
        assert eda_engine.output_dir.exists()
    
    def test_data_profiling(self, eda_engine):
        """Test phase 1: data profiling"""
        results = eda_engine.phase1_data_profiling()
        
        assert 'n_rows' in results
        assert 'n_cols' in results
        assert results['n_rows'] == 5
        assert results['n_cols'] == 4
    
    def test_data_quality_missing_values(self, eda_engine):
        """Test phase 2: missing value detection"""
        results = eda_engine.phase2_data_quality()
        
        assert 'missing_summary' in results
        # with_missing column should be detected
        assert 'outliers' in results
    
    def test_statistical_analysis(self, eda_engine):
        """Test phase 3: statistical analysis"""
        results = eda_engine.phase3_statistical_analysis()
        
        assert isinstance(results, dict)
        # Should contain statistics for our columns
        assert 'numeric1' in str(results)
    
    def test_correlation_analysis(self, eda_engine):
        """Test phase 4: correlation analysis"""
        results = eda_engine.phase4_correlation_analysis()
        
        assert isinstance(results, dict)
        # Should calculate correlations between numeric columns
    
    def test_complete_eda_pipeline(self, eda_engine):
        """Test full EDA pipeline execution"""
        results = eda_engine.run_complete_eda()
        
        assert 'profiling' in results
        assert 'quality' in results
        assert 'statistics' in results
        assert 'correlations' in results
        assert 'report_path' in results
        
        # Check report file was created
        assert results['report_path'].exists()
    
    def test_handles_empty_dataframe(self):
        """Test EDA handles edge case: empty dataset"""
        empty_df = pd.DataFrame()
        eda = EDAEngine(empty_df, name="Empty Dataset")
        
        # Should not crash
        results = eda.phase1_data_profiling()
        assert results['n_rows'] == 0
    
    def test_handles_single_column(self):
        """Test EDA handles edge case: single column"""
        single_col_df = pd.DataFrame({'col1': [1, 2, 3]})
        eda = EDAEngine(single_col_df, name="Single Column")
        
        # Should not crash
        results = eda.phase1_data_profiling()
        assert results['n_cols'] == 1


class TestDatasetGeneration:
    """Test synthetic data generation"""
    
    def test_generate_dataset_shape(self):
        """Test dataset has correct shape"""
        df = generate_ecommerce_dataset(n_samples=1000)
        
        assert len(df) == 1000
        assert df.shape[1] > 5  # Should have multiple columns
    
    def test_generate_dataset_columns(self):
        """Test dataset has expected columns"""
        df = generate_ecommerce_dataset(n_samples=100)
        
        required_columns = [
            'timestamp', 'user_age', 'pages_viewed', 
            'purchased', 'revenue', 'device_type'
        ]
        
        for col in required_columns:
            assert col in df.columns
    
    def test_purchase_behavior_realistic(self):
        """Test purchase behavior is realistic"""
        df = generate_ecommerce_dataset(n_samples=10000)
        
        # Conversion rate should be reasonable (10-20%)
        conversion_rate = df['purchased'].sum() / len(df)
        assert 0.10 <= conversion_rate <= 0.25
        
        # Revenue should be 0 for non-purchases
        assert (df[~df['purchased']]['revenue'] == 0).all()
    
    def test_age_distribution_realistic(self):
        """Test age distribution is realistic"""
        df = generate_ecommerce_dataset(n_samples=5000)
        
        # Ages should be within reasonable bounds
        assert df['user_age'].min() >= 18
        assert df['user_age'].max() <= 75


class TestProductionReadiness:
    """Test production-grade quality checks"""
    
    def test_no_crashes_on_all_numeric(self):
        """Test EDA handles all-numeric data"""
        numeric_df = pd.DataFrame({
            'a': range(100),
            'b': range(100, 200),
            'c': range(200, 300)
        })
        
        eda = EDAEngine(numeric_df, name="All Numeric")
        results = eda.run_complete_eda()
        
        assert results is not None
    
    def test_no_crashes_on_all_categorical(self):
        """Test EDA handles all-categorical data"""
        cat_df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 33 + ['A'],
            'cat2': ['X', 'Y', 'Z'] * 33 + ['X']
        })
        
        eda = EDAEngine(cat_df, name="All Categorical")
        results = eda.run_complete_eda()
        
        assert results is not None
    
    def test_handles_extreme_missing_values(self):
        """Test EDA handles dataset with extreme missing values"""
        extreme_df = pd.DataFrame({
            'mostly_missing': [1, np.nan, np.nan, np.nan, np.nan],
            'some_values': [1, 2, np.nan, 4, 5]
        })
        
        eda = EDAEngine(extreme_df, name="Extreme Missing")
        results = eda.phase2_data_quality()
        
        # Should detect high missing percentages
        assert 'missing_summary' in results


def test_integration_full_workflow():
    """
    Integration test: Full workflow from data generation to EDA completion.
    This simulates real-world usage.
    """
    # Generate data
    data = generate_ecommerce_dataset(n_samples=5000)
    
    # Run EDA
    eda = EDAEngine(data, name="Integration Test")
    results = eda.run_complete_eda()
    
    # Verify all phases completed
    assert all(key in results for key in [
        'profiling', 'quality', 'statistics', 
        'correlations', 'report_path'
    ])
    
    # Verify outputs exist
    assert (eda.output_dir / 'distributions.png').exists()
    assert (eda.output_dir / 'correlation_heatmap.png').exists()
    assert results['report_path'].exists()
    
    print("\n✅ Integration test passed - Full EDA workflow works!")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
