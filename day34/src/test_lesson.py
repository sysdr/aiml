"""
Tests for Day 34: DataFrame Operations
"""

import pytest
import pandas as pd
import numpy as np
from lesson_code import ContentRecommendationEngine


class TestDataFrameOperations:
    """Test suite for DataFrame indexing, slicing, and filtering."""
    
    @pytest.fixture
    def engine(self):
        """Create a test engine with small dataset."""
        np.random.seed(42)
        return ContentRecommendationEngine(n_content_items=100)
    
    @pytest.fixture
    def sample_df(self):
        """Create a simple test DataFrame."""
        data = {
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'score': [0.5, 0.7, 0.9, 0.6, 0.8]
        }
        df = pd.DataFrame(data)
        df.set_index('id', inplace=True)
        return df
    
    def test_indexing_loc(self, sample_df):
        """Test label-based indexing with .loc[]"""
        # Single row access
        row = sample_df.loc[3]
        assert row['value'] == 30
        assert row['category'] == 'A'
        
        # Single value access
        value = sample_df.loc[2, 'value']
        assert value == 20
    
    def test_indexing_iloc(self, sample_df):
        """Test position-based indexing with .iloc[]"""
        # First row
        first_row = sample_df.iloc[0]
        assert first_row['value'] == 10
        
        # Slice of rows
        slice_df = sample_df.iloc[1:3]
        assert len(slice_df) == 2
        assert slice_df.iloc[0]['value'] == 20
    
    def test_indexing_at(self, sample_df):
        """Test fast scalar access with .at[]"""
        value = sample_df.at[4, 'score']
        assert value == 0.6
    
    def test_slicing_columns(self, sample_df):
        """Test column slicing."""
        # Single column
        values = sample_df['value']
        assert len(values) == 5
        
        # Multiple columns
        subset = sample_df[['value', 'score']]
        assert subset.shape == (5, 2)
    
    def test_slicing_rows(self, sample_df):
        """Test row slicing."""
        # First 3 rows
        first_three = sample_df.iloc[0:3]
        assert len(first_three) == 3
        
        # Last 2 rows
        last_two = sample_df.iloc[-2:]
        assert len(last_two) == 2
    
    def test_filtering_single_condition(self, sample_df):
        """Test filtering with single condition."""
        # Filter by value
        high_value = sample_df[sample_df['value'] > 25]
        assert len(high_value) == 3
        
        # Filter by category
        category_a = sample_df[sample_df['category'] == 'A']
        assert len(category_a) == 3
    
    def test_filtering_multiple_conditions(self, sample_df):
        """Test filtering with multiple conditions."""
        # AND condition
        filtered = sample_df[
            (sample_df['value'] > 20) &
            (sample_df['score'] > 0.6)
        ]
        assert len(filtered) == 2
        
        # OR condition
        filtered = sample_df[
            (sample_df['category'] == 'A') |
            (sample_df['score'] > 0.8)
        ]
        assert len(filtered) == 3
    
    def test_engine_initialization(self, engine):
        """Test ContentRecommendationEngine initialization."""
        assert len(engine.df) == 100
        assert 'views' in engine.df.columns
        assert 'completion_rate' in engine.df.columns
        assert 'rating' in engine.df.columns
    
    def test_recommendation_filter(self, engine):
        """Test the recommendation filter functionality."""
        recommendations = engine.build_recommendation_filter(
            user_category='tech',
            min_quality=0.5
        )
        
        # Should return DataFrame
        assert isinstance(recommendations, pd.DataFrame)
        
        # Should have engagement score
        assert 'engagement_score' in recommendations.columns
        
        # Should be sorted
        scores = recommendations['engagement_score'].values
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    def test_data_quality_analysis(self, engine):
        """Test data quality analysis."""
        results = engine.analyze_data_quality()
        
        # Check required keys
        assert 'total' in results
        assert 'low_quality' in results
        assert 'high_engagement' in results
        assert 'underperforming' in results
        
        # Check values are reasonable
        assert results['total'] == 100
        assert 0 <= results['low_quality'] <= 100
        assert 0 <= results['high_engagement'] <= 100
    
    def test_performance_difference(self):
        """Test that .at[] is faster than .loc[] for scalar access."""
        # Create large DataFrame
        df = pd.DataFrame({
            'value': range(10000)
        })
        
        import time
        
        # Time .loc[]
        start = time.time()
        for _ in range(1000):
            _ = df.loc[5000, 'value']
        loc_time = time.time() - start
        
        # Time .at[]
        start = time.time()
        for _ in range(1000):
            _ = df.at[5000, 'value']
        at_time = time.time() - start
        
        # .at[] should be faster
        assert at_time < loc_time
    
    def test_combined_operations(self, sample_df):
        """Test combining indexing, slicing, and filtering."""
        # Filter, then slice, then index
        result = sample_df[sample_df['score'] > 0.6].iloc[0:2].loc[:, 'value']
        
        assert len(result) <= 2
        assert isinstance(result, pd.Series)


class TestRealWorldScenarios:
    """Test real-world data selection scenarios."""
    
    @pytest.fixture
    def user_data(self):
        """Create realistic user behavior data."""
        np.random.seed(42)
        n = 500
        data = {
            'user_id': range(1, n + 1),
            'login_count': np.random.randint(1, 100, n),
            'purchase_amount': np.random.uniform(0, 1000, n),
            'engagement_score': np.random.uniform(0, 1, n),
            'is_premium': np.random.choice([True, False], n),
            'signup_date': pd.date_range('2023-01-01', periods=n, freq='D')
        }
        df = pd.DataFrame(data)
        df.set_index('user_id', inplace=True)
        return df
    
    def test_active_user_filter(self, user_data):
        """Test filtering for active users."""
        active_users = user_data[
            (user_data['login_count'] > 10) &
            (user_data['engagement_score'] > 0.5)
        ]
        
        assert all(active_users['login_count'] > 10)
        assert all(active_users['engagement_score'] > 0.5)
    
    def test_high_value_customer_filter(self, user_data):
        """Test filtering for high-value customers."""
        high_value = user_data[
            (user_data['purchase_amount'] > 500) |
            ((user_data['is_premium'] == True) & 
             (user_data['engagement_score'] > 0.7))
        ]
        
        assert len(high_value) > 0
        # Verify condition logic
        for idx, row in high_value.iterrows():
            assert (row['purchase_amount'] > 500) or \
                   (row['is_premium'] and row['engagement_score'] > 0.7)
    
    def test_recent_user_slice(self, user_data):
        """Test slicing for recent signups."""
        # Get last 50 signups
        recent = user_data.sort_values('signup_date').iloc[-50:]
        
        assert len(recent) == 50
        # Verify sorting
        dates = recent['signup_date'].values
        assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
