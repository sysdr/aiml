"""
Day 33: Test Suite
Verify understanding of Pandas fundamentals
"""

import pytest
import pandas as pd
import numpy as np

class TestSeriesFundamentals:
    """Test understanding of Pandas Series"""
    
    def test_series_creation_with_index(self):
        """Series should maintain labeled index"""
        data = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
        assert data['b'] == 20
        assert len(data) == 3
    
    def test_series_vectorized_operations(self):
        """Series should support vectorized math"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = data * 2
        assert list(result) == [2, 4, 6, 8, 10]
    
    def test_series_statistics(self):
        """Series should compute statistics correctly"""
        data = pd.Series([10, 20, 30, 40, 50])
        assert data.mean() == 30
        assert data.sum() == 150
        assert data.max() == 50
    
    def test_series_boolean_indexing(self):
        """Series should support boolean filtering"""
        data = pd.Series([15, 25, 35, 45])
        filtered = data[data > 30]
        assert len(filtered) == 2
        assert list(filtered) == [35, 45]

class TestDataFrameFundamentals:
    """Test understanding of Pandas DataFrames"""
    
    def test_dataframe_creation(self):
        """DataFrame should create from dictionary"""
        df = pd.DataFrame({
            'col_a': [1, 2, 3],
            'col_b': [4, 5, 6]
        })
        assert df.shape == (3, 2)
        assert list(df.columns) == ['col_a', 'col_b']
    
    def test_column_access(self):
        """Should access columns by name"""
        df = pd.DataFrame({
            'accuracy': [0.8, 0.85, 0.9],
            'loss': [0.5, 0.4, 0.3]
        })
        assert list(df['accuracy']) == [0.8, 0.85, 0.9]
    
    def test_row_access_with_loc(self):
        """Should access rows by label with .loc"""
        df = pd.DataFrame(
            {'value': [100, 200, 300]},
            index=['first', 'second', 'third']
        )
        assert df.loc['second', 'value'] == 200
    
    def test_add_new_column(self):
        """Should add computed columns"""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        df['sum'] = df['a'] + df['b']
        assert list(df['sum']) == [5, 7, 9]

class TestDataExploration:
    """Test data exploration techniques"""
    
    def test_head_returns_first_n(self):
        """head() should return first n rows"""
        df = pd.DataFrame({'x': range(100)})
        assert len(df.head()) == 5
        assert len(df.head(10)) == 10
    
    def test_describe_statistics(self):
        """describe() should return statistical summary"""
        df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
        stats = df.describe()
        assert 'mean' in stats.index
        assert 'std' in stats.index
        assert stats.loc['mean', 'values'] == 3.0
    
    def test_missing_value_detection(self):
        """Should detect missing values"""
        df = pd.DataFrame({
            'complete': [1, 2, 3],
            'has_null': [1, None, 3]
        })
        missing = df.isnull().sum()
        assert missing['complete'] == 0
        assert missing['has_null'] == 1
    
    def test_unique_values(self):
        """Should find unique values"""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        assert df['category'].nunique() == 3
        assert set(df['category'].unique()) == {'A', 'B', 'C'}

class TestFeatureEngineering:
    """Test feature engineering basics"""
    
    def test_computed_feature(self):
        """Should create features from existing columns"""
        df = pd.DataFrame({
            'distance': [100, 200, 150],
            'time': [10, 20, 15]
        })
        df['speed'] = df['distance'] / df['time']
        assert list(df['speed']) == [10.0, 10.0, 10.0]
    
    def test_groupby_aggregation(self):
        """Should aggregate by groups"""
        df = pd.DataFrame({
            'model': ['A', 'A', 'B', 'B'],
            'score': [80, 90, 70, 75]
        })
        avg_scores = df.groupby('model')['score'].mean()
        assert avg_scores['A'] == 85
        assert avg_scores['B'] == 72.5

class TestRealWorldScenarios:
    """Test real-world AI/ML scenarios"""
    
    def test_model_comparison(self):
        """Should compare model performance"""
        df = pd.DataFrame({
            'model_id': ['m1', 'm1', 'm2', 'm2'],
            'epoch': [1, 2, 1, 2],
            'accuracy': [0.6, 0.8, 0.5, 0.7]
        })
        
        best_per_model = df.groupby('model_id')['accuracy'].max()
        assert best_per_model['m1'] == 0.8
        assert best_per_model['m2'] == 0.7
    
    def test_training_efficiency(self):
        """Should calculate training efficiency metrics"""
        df = pd.DataFrame({
            'accuracy': [0.8, 0.9],
            'training_time': [100, 200]
        })
        df['efficiency'] = df['accuracy'] / df['training_time']
        
        # First model more efficient (0.8/100 = 0.008 vs 0.9/200 = 0.0045)
        assert df['efficiency'].iloc[0] > df['efficiency'].iloc[1]

def run_tests():
    """Run all tests and display results"""
    pytest.main([__file__, '-v', '--tb=short'])

if __name__ == "__main__":
    run_tests()
