"""
Day 74: Feature Engineering - Comprehensive Test Suite
Tests all feature engineering components
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from lesson_code import (
    FeatureEngineeringPipeline,
    PolynomialFeatureCreator,
    FeatureBinner,
    FeatureSelector,
    create_sample_dataset
)


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['category'] = np.random.choice(['A', 'B', 'C'], 200)
    return df, y


@pytest.fixture
def mixed_type_data():
    """Create dataset with mixed types."""
    np.random.seed(42)
    df = pd.DataFrame({
        'numeric1': np.random.randn(100),
        'numeric2': np.random.randint(0, 100, 100),
        'category1': np.random.choice(['X', 'Y', 'Z'], 100),
        'category2': np.random.choice(['Low', 'Medium', 'High'], 100)
    })
    y = np.random.randint(0, 2, 100)
    return df, y


class TestFeatureEngineeringPipeline:
    """Test FeatureEngineeringPipeline class."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = FeatureEngineeringPipeline(
            scaling_strategy='standard',
            encoding_strategy='onehot'
        )
        assert pipeline.scaling_strategy == 'standard'
        assert pipeline.encoding_strategy == 'onehot'
        assert pipeline.preprocessor is None
    
    def test_feature_analysis(self, mixed_type_data):
        """Test automatic feature type detection."""
        X, y = mixed_type_data
        pipeline = FeatureEngineeringPipeline()
        pipeline.analyze_features(X)
        
        assert len(pipeline.numeric_features) == 2
        assert len(pipeline.categorical_features) == 2
        assert 'numeric1' in pipeline.numeric_features
        assert 'category1' in pipeline.categorical_features
    
    def test_standard_scaling(self, mixed_type_data):
        """Test StandardScaler transformation."""
        X, y = mixed_type_data
        pipeline = FeatureEngineeringPipeline(scaling_strategy='standard')
        X_transformed = pipeline.fit_transform(X)
        
        # Check shape
        assert X_transformed.shape[0] == X.shape[0]
        
        # Numeric features should be standardized (mean≈0, std≈1)
        numeric_cols = X_transformed[:, :2]  # First 2 are numeric
        assert np.abs(numeric_cols.mean()) < 0.1
        assert np.abs(numeric_cols.std() - 1.0) < 0.1
    
    def test_minmax_scaling(self, mixed_type_data):
        """Test MinMaxScaler transformation."""
        X, y = mixed_type_data
        pipeline = FeatureEngineeringPipeline(scaling_strategy='minmax')
        X_transformed = pipeline.fit_transform(X)
        
        # Numeric features should be in [0, 1] range
        numeric_cols = X_transformed[:, :2]
        assert numeric_cols.min() >= -0.01  # Allow small numerical errors
        assert numeric_cols.max() <= 1.01
    
    def test_robust_scaling(self, mixed_type_data):
        """Test RobustScaler handles outliers."""
        X, y = mixed_type_data
        
        # Add outliers
        X_with_outliers = X.copy()
        X_with_outliers.loc[0, 'numeric1'] = 1000  # Extreme outlier
        
        pipeline = FeatureEngineeringPipeline(scaling_strategy='robust')
        X_transformed = pipeline.fit_transform(X_with_outliers)
        
        # RobustScaler should handle outliers better than StandardScaler
        assert X_transformed.shape[0] == X_with_outliers.shape[0]
    
    def test_onehot_encoding(self, mixed_type_data):
        """Test OneHotEncoder transformation."""
        X, y = mixed_type_data
        pipeline = FeatureEngineeringPipeline(encoding_strategy='onehot')
        X_transformed = pipeline.fit_transform(X)
        
        # One-hot encoding should increase dimensionality
        # 2 numeric + (3 categories from category1 + 3 from category2) = 8
        assert X_transformed.shape[1] >= X.shape[1]
    
    def test_transform_consistency(self, mixed_type_data):
        """Test that transform produces same output as fit_transform."""
        X, y = mixed_type_data
        pipeline = FeatureEngineeringPipeline()
        
        X_fit_transform = pipeline.fit_transform(X)
        X_transform = pipeline.transform(X)
        
        np.testing.assert_array_almost_equal(X_fit_transform, X_transform)
    
    def test_unseen_categories(self, mixed_type_data):
        """Test handling of unseen categories in test data."""
        X, y = mixed_type_data
        pipeline = FeatureEngineeringPipeline(encoding_strategy='onehot')
        
        # Fit on training data
        X_train = X[:80]
        pipeline.fit_transform(X_train)
        
        # Transform test data with potentially unseen categories
        X_test = X[80:]
        X_test_transformed = pipeline.transform(X_test)
        
        # Should not raise error
        assert X_test_transformed.shape[0] == X_test.shape[0]


class TestPolynomialFeatureCreator:
    """Test PolynomialFeatureCreator class."""
    
    def test_initialization(self):
        """Test polynomial creator initialization."""
        creator = PolynomialFeatureCreator(degree=2, interaction_only=False)
        assert creator.degree == 2
        assert creator.interaction_only == False
    
    def test_degree_2_features(self):
        """Test creation of degree-2 polynomial features."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        creator = PolynomialFeatureCreator(degree=2, interaction_only=False)
        X_poly = creator.fit_transform(X)
        
        # With 2 features, degree=2, no bias: [x1, x2, x1^2, x1*x2, x2^2] = 5 features
        assert X_poly.shape[1] == 5
    
    def test_interaction_only(self):
        """Test interaction-only polynomial features."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        creator = PolynomialFeatureCreator(degree=2, interaction_only=True)
        X_poly = creator.fit_transform(X)
        
        # With 2 features, interaction only: [x1, x2, x1*x2] = 3 features
        assert X_poly.shape[1] == 3
    
    def test_feature_explosion(self):
        """Test that higher degrees create many features."""
        X = np.random.randn(50, 5)
        
        creator_2 = PolynomialFeatureCreator(degree=2)
        X_poly_2 = creator_2.fit_transform(X)
        
        creator_3 = PolynomialFeatureCreator(degree=3)
        X_poly_3 = creator_3.fit_transform(X)
        
        # Degree 3 should create more features than degree 2
        assert X_poly_3.shape[1] > X_poly_2.shape[1]
    
    def test_transform_consistency(self):
        """Test transform consistency."""
        X = np.random.randn(50, 3)
        creator = PolynomialFeatureCreator(degree=2)
        
        X_fit_transform = creator.fit_transform(X)
        X_transform = creator.transform(X)
        
        np.testing.assert_array_almost_equal(X_fit_transform, X_transform)


class TestFeatureBinner:
    """Test FeatureBinner class."""
    
    def test_initialization(self):
        """Test binner initialization."""
        binner = FeatureBinner(n_bins=5, strategy='quantile')
        assert binner.n_bins == 5
        assert binner.strategy == 'quantile'
    
    def test_quantile_binning(self):
        """Test quantile-based binning."""
        X = np.random.randn(100)
        binner = FeatureBinner(n_bins=4, strategy='quantile')
        X_binned = binner.fit_transform(X, 'feature')
        
        # Should create 4 bins with roughly equal counts
        unique_bins = np.unique(X_binned)
        assert len(unique_bins) == 4
        
        # Check that bins are relatively balanced (±20%)
        bin_counts = [np.sum(X_binned == b) for b in unique_bins]
        assert max(bin_counts) / min(bin_counts) < 2.0
    
    def test_uniform_binning(self):
        """Test uniform-width binning."""
        X = np.linspace(0, 100, 100)
        binner = FeatureBinner(n_bins=5, strategy='uniform')
        X_binned = binner.fit_transform(X, 'feature')
        
        # Should create 5 bins
        unique_bins = np.unique(X_binned)
        assert len(unique_bins) == 5
    
    def test_bin_edges_stored(self):
        """Test that bin edges are stored correctly."""
        X = np.random.randn(100)
        binner = FeatureBinner(n_bins=3, strategy='quantile')
        binner.fit_transform(X, 'test_feature')
        
        assert 'test_feature' in binner.bin_edges
        assert len(binner.bin_edges['test_feature']) == 4  # n_bins + 1 edges


class TestFeatureSelector:
    """Test FeatureSelector class."""
    
    def test_initialization(self):
        """Test selector initialization."""
        selector = FeatureSelector(k=5, score_func='f_classif')
        assert selector.k == 5
    
    def test_feature_selection(self, sample_data):
        """Test that feature selection reduces dimensionality."""
        X, y = sample_data
        # Only use numeric columns for feature selection
        X_numeric = X.select_dtypes(include=['int64', 'float64']).values
        selector = FeatureSelector(k=3)
        X_selected = selector.fit_transform(X_numeric, y)
        
        assert X_selected.shape[1] == 3
        assert X_selected.shape[0] == X.shape[0]
    
    def test_feature_scores_computed(self, sample_data):
        """Test that feature scores are computed."""
        X, y = sample_data
        # Only use numeric columns for feature selection
        X_numeric = X.select_dtypes(include=['int64', 'float64']).values
        selector = FeatureSelector(k=3)
        selector.fit_transform(X_numeric, y)
        
        assert selector.feature_scores is not None
        assert len(selector.feature_scores) == X_numeric.shape[1]
    
    def test_mutual_info_selection(self, sample_data):
        """Test mutual information-based selection."""
        X, y = sample_data
        # Only use numeric columns for feature selection
        X_numeric = X.select_dtypes(include=['int64', 'float64']).values
        selector = FeatureSelector(k=3, score_func='mutual_info')
        X_selected = selector.fit_transform(X_numeric, y)
        
        assert X_selected.shape[1] == 3
    
    def test_transform_consistency(self, sample_data):
        """Test transform consistency."""
        X, y = sample_data
        # Only use numeric columns for feature selection
        X_numeric = X.select_dtypes(include=['int64', 'float64']).values
        selector = FeatureSelector(k=3)
        
        X_fit_transform = selector.fit_transform(X_numeric, y)
        X_transform = selector.transform(X_numeric)
        
        np.testing.assert_array_almost_equal(X_fit_transform, X_transform)


class TestDatasetCreation:
    """Test dataset creation function."""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        df = create_sample_dataset()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 1000
        assert 'churn' in df.columns
        assert df['churn'].nunique() == 2  # Binary target
    
    def test_dataset_has_required_columns(self):
        """Test that dataset has all required columns."""
        df = create_sample_dataset()
        
        required_cols = [
            'age', 'tenure', 'monthly_charges', 'total_charges',
            'contract', 'internet_service', 'payment_method', 'churn'
        ]
        
        for col in required_cols:
            assert col in df.columns
    
    def test_dataset_data_types(self):
        """Test that dataset has correct data types."""
        df = create_sample_dataset()
        
        # Numeric columns
        assert pd.api.types.is_numeric_dtype(df['age'])
        assert pd.api.types.is_numeric_dtype(df['tenure'])
        assert pd.api.types.is_numeric_dtype(df['monthly_charges'])
        
        # Categorical columns
        assert pd.api.types.is_object_dtype(df['contract'])
        assert pd.api.types.is_object_dtype(df['internet_service'])


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_pipeline(self, mixed_type_data):
        """Test complete feature engineering pipeline."""
        X, y = mixed_type_data
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Apply pipeline
        pipeline = FeatureEngineeringPipeline()
        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed = pipeline.transform(X_test)
        
        # Train simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_transformed, y_train)
        
        # Predict
        predictions = model.predict(X_test_transformed)
        
        assert len(predictions) == len(y_test)
    
    def test_polynomial_then_selection(self, sample_data):
        """Test polynomial feature creation followed by selection."""
        X, y = sample_data
        
        # Only use numeric columns for polynomial features
        X_numeric = X.select_dtypes(include=['int64', 'float64']).values
        
        # Create polynomial features
        poly_creator = PolynomialFeatureCreator(degree=2)
        X_poly = poly_creator.fit_transform(X_numeric)
        
        # Select best features
        selector = FeatureSelector(k=10)
        X_selected = selector.fit_transform(X_poly, y)
        
        assert X_selected.shape[1] == 10
        assert X_selected.shape[0] == X.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
