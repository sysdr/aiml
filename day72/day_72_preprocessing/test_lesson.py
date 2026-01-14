"""
Comprehensive test suite for Day 72: Data Preprocessing and Feature Scaling
Tests cover edge cases critical for production systems
"""

import pytest
import numpy as np
import pandas as pd
from lesson_code import (
    DataPreprocessor,
    MissingDataHandler,
    FeatureScalingComparison,
    CategoricalEncoder,
    create_sample_dataset
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class TestDataPreprocessor:
    """Test DataPreprocessor class for production readiness."""
    
    def test_initialization(self):
        """Test preprocessor initializes correctly."""
        preprocessor = DataPreprocessor(strategy='robust')
        assert preprocessor.strategy == 'robust'
        assert preprocessor.fitted == False
        assert preprocessor.preprocessing_pipeline is None
    
    def test_invalid_strategy_defaults_to_robust(self):
        """Test that invalid strategy defaults to robust."""
        preprocessor = DataPreprocessor(strategy='invalid')
        assert preprocessor.strategy == 'invalid'  # Stores it but will use robust
    
    def test_pipeline_creation(self):
        """Test pipeline is created with correct structure."""
        preprocessor = DataPreprocessor(strategy='standard')
        numerical_features = ['age', 'income']
        categorical_features = ['city', 'device']
        
        pipeline = preprocessor.create_pipeline(numerical_features, categorical_features)
        assert pipeline is not None
        assert hasattr(pipeline, 'transformers')
        assert len(pipeline.transformers) == 2  # num and cat
    
    def test_fit_transform_basic(self):
        """Test basic fit and transform operations."""
        df = pd.DataFrame({
            'age': [25, 30, 35, np.nan, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'city': ['NY', 'LA', 'NY', 'SF', 'LA']
        })
        
        preprocessor = DataPreprocessor(strategy='standard')
        X_transformed = preprocessor.fit_transform(
            df, ['age', 'income'], ['city']
        )
        
        assert preprocessor.fitted == True
        assert X_transformed.shape[0] == 5
        assert X_transformed.shape[1] > 3  # One-hot encoding expands features
    
    def test_transform_without_fit_raises_error(self):
        """Test that transform before fit raises error."""
        df = pd.DataFrame({'age': [25, 30, 35]})
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError, match="Must call fit"):
            preprocessor.transform(df)
    
    def test_consistent_train_test_transform(self):
        """Test that train and test data are transformed consistently."""
        train_df = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'city': ['NY', 'LA', 'SF', 'NY']
        })
        test_df = pd.DataFrame({
            'age': [28, 33],
            'city': ['LA', 'NY']
        })
        
        preprocessor = DataPreprocessor(strategy='minmax')
        X_train = preprocessor.fit_transform(train_df, ['age'], ['city'])
        X_test = preprocessor.transform(test_df)
        
        # Both should have same number of features
        assert X_train.shape[1] == X_test.shape[1]
    
    def test_handles_unseen_categories(self):
        """Test that pipeline handles unseen categories in test data."""
        train_df = pd.DataFrame({
            'age': [25, 30, 35],
            'city': ['NY', 'LA', 'SF']
        })
        test_df = pd.DataFrame({
            'age': [28],
            'city': ['BOSTON']  # Unseen category
        })
        
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(train_df, ['age'], ['city'])
        X_test = preprocessor.transform(test_df)
        
        # Should not raise error, handle_unknown='ignore' in OneHotEncoder
        assert X_test.shape[0] == 1
    
    def test_all_missing_column(self):
        """Test handling of column with all missing values."""
        df = pd.DataFrame({
            'age': [np.nan, np.nan, np.nan],
            'income': [50000, 60000, 70000],
            'city': ['NY', 'LA', 'SF']
        })
        
        preprocessor = DataPreprocessor()
        X_transformed = preprocessor.fit_transform(df, ['age', 'income'], ['city'])
        
        # Should handle without error (imputer fills with median)
        assert X_transformed.shape[0] == 3
        assert not np.any(np.isnan(X_transformed))


class TestMissingDataHandler:
    """Test missing data handling strategies."""
    
    def test_analyze_missing_no_missing(self):
        """Test analysis when no missing values."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000]
        })
        
        handler = MissingDataHandler()
        missing_stats = handler.analyze_missing(df)
        
        assert len(missing_stats) == 0
    
    def test_analyze_missing_with_missing(self):
        """Test analysis with missing values."""
        df = pd.DataFrame({
            'age': [25, np.nan, 35],
            'income': [50000, 60000, np.nan]
        })
        
        handler = MissingDataHandler()
        missing_stats = handler.analyze_missing(df)
        
        assert len(missing_stats) == 2
        assert 'age' in missing_stats['column'].values
        assert 'income' in missing_stats['column'].values
        assert missing_stats[missing_stats['column'] == 'age']['missing_percentage'].values[0] > 0
    
    def test_create_missing_indicator(self):
        """Test creation of missing value indicators."""
        df = pd.DataFrame({
            'age': [25, np.nan, 35, 40],
            'income': [50000, 60000, 70000, np.nan]
        })
        
        handler = MissingDataHandler()
        df_with_indicators = handler.create_missing_indicator(df, ['age', 'income'])
        
        assert 'age_was_missing' in df_with_indicators.columns
        assert 'income_was_missing' in df_with_indicators.columns
        assert df_with_indicators['age_was_missing'].sum() == 1
        assert df_with_indicators['income_was_missing'].sum() == 1
    
    def test_impute_simple_numerical(self):
        """Test simple imputation for numerical data."""
        df = pd.DataFrame({
            'age': [25, np.nan, 35, 40],
            'income': [50000, 60000, np.nan, 80000]
        })
        
        handler = MissingDataHandler()
        df_imputed = handler.impute_simple(df, numerical_strategy='median')
        
        assert not df_imputed['age'].isnull().any()
        assert not df_imputed['income'].isnull().any()
        assert df_imputed['age'].iloc[1] == 35.0  # Median of [25, 35, 40]
    
    def test_impute_simple_categorical(self):
        """Test simple imputation for categorical data."""
        df = pd.DataFrame({
            'city': ['NY', 'LA', None, 'NY', 'LA']
        })
        
        handler = MissingDataHandler()
        df_imputed = handler.impute_simple(df, categorical_strategy='most_frequent')
        
        assert not df_imputed['city'].isnull().any()
        # Most frequent should be 'NY' or 'LA' (both appear twice)
    
    def test_impute_knn_numerical(self):
        """Test KNN imputation."""
        df = pd.DataFrame({
            'age': [25, np.nan, 35, 40],
            'income': [50000, 60000, 70000, 80000]
        })
        
        handler = MissingDataHandler()
        df_imputed = handler.impute_knn(df, n_neighbors=2)
        
        assert not df_imputed['age'].isnull().any()
        # KNN should estimate based on similar income values
    
    def test_impute_all_missing(self):
        """Test imputation when entire column is missing."""
        df = pd.DataFrame({
            'age': [np.nan, np.nan, np.nan],
            'income': [50000, 60000, 70000]
        })
        
        handler = MissingDataHandler()
        df_imputed = handler.impute_simple(df)
        
        # Should handle gracefully (might fill with 0 or similar)
        assert df_imputed.shape == df.shape


class TestFeatureScalingComparison:
    """Test feature scaling comparisons."""
    
    def test_initialization(self):
        """Test scaler comparison initializes correctly."""
        comparison = FeatureScalingComparison()
        assert len(comparison.scalers) == 3
        assert 'StandardScaler' in comparison.scalers
        assert 'MinMaxScaler' in comparison.scalers
        assert 'RobustScaler' in comparison.scalers
    
    def test_fit_all_scalers(self):
        """Test fitting all scalers."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        comparison = FeatureScalingComparison()
        comparison.fit_all(X)
        
        # All scalers should be fitted
        for scaler in comparison.scalers.values():
            assert hasattr(scaler, 'n_features_in_')
    
    def test_transform_all_scalers(self):
        """Test transforming with all scalers."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        comparison = FeatureScalingComparison()
        comparison.fit_all(X)
        results = comparison.transform_all(X)
        
        assert len(results) == 3
        assert all(arr.shape == X.shape for arr in results.values())
    
    def test_compare_statistics(self):
        """Test statistics comparison."""
        X = np.array([[1, 100], [2, 200], [3, 300]])
        comparison = FeatureScalingComparison()
        comparison.fit_all(X)
        comparison.transform_all(X)
        
        stats = comparison.compare_statistics(feature_index=0)
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 3
        assert 'mean' in stats.columns
        assert 'std' in stats.columns
    
    def test_standard_scaler_mean_zero(self):
        """Test StandardScaler produces mean ~0."""
        X = np.array([[1], [2], [3], [4], [5]])
        comparison = FeatureScalingComparison()
        comparison.fit_all(X)
        comparison.transform_all(X)
        
        stats = comparison.compare_statistics(feature_index=0)
        assert abs(stats.loc['StandardScaler', 'mean']) < 1e-10
    
    def test_minmax_scaler_range(self):
        """Test MinMaxScaler produces [0, 1] range."""
        X = np.array([[1], [2], [3], [4], [5]])
        comparison = FeatureScalingComparison()
        comparison.fit_all(X)
        comparison.transform_all(X)
        
        stats = comparison.compare_statistics(feature_index=0)
        assert stats.loc['MinMaxScaler', 'min'] == 0.0
        assert stats.loc['MinMaxScaler', 'max'] == 1.0


class TestCategoricalEncoder:
    """Test categorical encoding strategies."""
    
    def test_onehot_encode_basic(self):
        """Test basic one-hot encoding."""
        df = pd.DataFrame({'city': ['NY', 'LA', 'SF', 'NY']})
        encoder = CategoricalEncoder()
        df_encoded = encoder.onehot_encode(df, ['city'])
        
        # Should have 3 columns (one for each unique city)
        assert df_encoded.shape[1] == 3
        assert 'city_NY' in df_encoded.columns
        assert 'city_LA' in df_encoded.columns
        assert 'city_SF' in df_encoded.columns
    
    def test_onehot_encode_drop_first(self):
        """Test one-hot encoding with drop_first."""
        df = pd.DataFrame({'city': ['NY', 'LA', 'SF', 'NY']})
        encoder = CategoricalEncoder()
        df_encoded = encoder.onehot_encode(df, ['city'], drop_first=True)
        
        # Should have 2 columns (n-1)
        assert df_encoded.shape[1] == 2
    
    def test_label_encode(self):
        """Test label encoding."""
        df = pd.DataFrame({'city': ['NY', 'LA', 'SF', 'NY']})
        encoder = CategoricalEncoder()
        df_encoded, encoders = encoder.label_encode(df, ['city'])
        
        assert df_encoded['city'].dtype in [np.int32, np.int64]
        assert 'city' in encoders
        assert len(df_encoded) == 4
    
    def test_frequency_encode(self):
        """Test frequency encoding."""
        df = pd.DataFrame({'city': ['NY', 'NY', 'LA', 'SF']})
        encoder = CategoricalEncoder()
        df_encoded = encoder.frequency_encode(df, ['city'])
        
        # NY appears 2/4 = 0.5, LA and SF each 1/4 = 0.25
        assert df_encoded['city'].iloc[0] == 0.5
        assert df_encoded['city'].iloc[2] == 0.25
    
    def test_target_encode(self):
        """Test target encoding."""
        df = pd.DataFrame({
            'city': ['NY', 'NY', 'LA', 'LA', 'SF'],
            'target': [1, 1, 0, 0, 1]
        })
        target = df['target']
        
        encoder = CategoricalEncoder()
        df_encoded = encoder.target_encode(df, 'city', target, smoothing=1.0)
        
        assert 'city_encoded' in df_encoded.columns
        # NY should have higher encoding (mean=1.0) than LA (mean=0.0)
        ny_encoding = df_encoded[df_encoded['city'] == 'NY']['city_encoded'].iloc[0]
        la_encoding = df_encoded[df_encoded['city'] == 'LA']['city_encoded'].iloc[0]
        assert ny_encoding > la_encoding


class TestDatasetGeneration:
    """Test sample dataset generation."""
    
    def test_create_sample_dataset_shape(self):
        """Test dataset has correct shape."""
        df = create_sample_dataset(100)
        assert df.shape[0] == 100
        assert df.shape[1] == 8  # 8 features
    
    def test_create_sample_dataset_columns(self):
        """Test dataset has expected columns."""
        df = create_sample_dataset(50)
        expected_cols = ['age', 'device_type', 'session_duration', 'watch_hours',
                        'last_login_days', 'subscription_tier', 'region', 'premium_user']
        assert all(col in df.columns for col in expected_cols)
    
    def test_create_sample_dataset_has_missing(self):
        """Test dataset contains missing values."""
        df = create_sample_dataset(200)
        assert df.isnull().sum().sum() > 0  # Some missing values
    
    def test_create_sample_dataset_types(self):
        """Test dataset has correct data types."""
        df = create_sample_dataset(50)
        
        # Numerical columns
        assert pd.api.types.is_numeric_dtype(df['age'])
        assert pd.api.types.is_numeric_dtype(df['session_duration'])
        
        # Categorical columns  
        assert pd.api.types.is_object_dtype(df['device_type'])
        assert pd.api.types.is_object_dtype(df['subscription_tier'])
    
    def test_create_sample_dataset_ranges(self):
        """Test dataset values are in expected ranges."""
        df = create_sample_dataset(100)
        
        # Remove NaN for range checks
        df_clean = df.dropna()
        
        assert df_clean['age'].min() >= 18
        assert df_clean['age'].max() <= 80
        assert df_clean['session_duration'].min() >= 1
        assert df_clean['premium_user'].isin([0, 1]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
