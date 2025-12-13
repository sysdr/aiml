"""
Tests for Day 40: Regression vs. Classification
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lesson_code import HousePriceSystem


@pytest.fixture
def house_system():
    """Create HousePriceSystem instance for testing"""
    return HousePriceSystem(n_samples=100, random_state=42)


@pytest.fixture
def sample_data(house_system):
    """Generate sample data for testing"""
    df = house_system.generate_data()
    feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
    X = df[feature_cols]
    y_price = df['price']
    y_tier = df['tier']
    
    X_train, X_test, y_price_train, y_price_test, y_tier_train, y_tier_test = train_test_split(
        X, y_price, y_tier, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_price_train, y_price_test, y_tier_train, y_tier_test


class TestDataGeneration:
    """Test data generation functionality"""
    
    def test_data_shape(self, house_system):
        """Test that generated data has correct shape"""
        df = house_system.generate_data()
        assert len(df) == 100
        assert len(df.columns) == 7  # 5 features + price + tier
    
    def test_feature_ranges(self, house_system):
        """Test that features are in expected ranges"""
        df = house_system.generate_data()
        
        assert df['sqft'].min() >= 800
        assert df['sqft'].max() <= 4000
        assert df['bedrooms'].min() >= 1
        assert df['bedrooms'].max() <= 5
        assert df['bathrooms'].min() >= 1
        assert df['bathrooms'].max() <= 4
        assert df['age'].min() >= 0
        assert df['age'].max() <= 50
        assert df['location_score'].min() >= 1
        assert df['location_score'].max() <= 10
    
    def test_price_positive(self, house_system):
        """Test that all prices are positive"""
        df = house_system.generate_data()
        assert (df['price'] > 0).all()
    
    def test_tier_categories(self, house_system):
        """Test that tiers have correct categories"""
        df = house_system.generate_data()
        expected_tiers = ['Budget', 'Mid-Range', 'Luxury', 'Ultra-Luxury']
        assert set(df['tier'].unique()).issubset(set(expected_tiers))


class TestRegressionModel:
    """Test regression model functionality"""
    
    def test_model_training(self, house_system, sample_data):
        """Test that regression model trains successfully"""
        X_train, _, y_price_train, _, _, _ = sample_data
        
        predictions = house_system.train_regression_model(X_train, y_price_train)
        
        assert house_system.regression_model is not None
        assert len(predictions) == len(X_train)
        assert predictions.shape[0] == X_train.shape[0]
    
    def test_prediction_output_type(self, house_system, sample_data):
        """Test that predictions are continuous values"""
        X_train, X_test, y_price_train, _, _, _ = sample_data
        
        house_system.train_regression_model(X_train, y_price_train)
        X_test_scaled = house_system.scaler.transform(X_test)
        predictions = house_system.regression_model.predict(X_test_scaled)
        
        # Regression outputs continuous values
        assert predictions.dtype in [np.float64, np.float32]
        assert len(np.unique(predictions)) > 10  # Many unique values
    
    def test_reasonable_predictions(self, house_system, sample_data):
        """Test that price predictions are reasonable"""
        X_train, X_test, y_price_train, _, _, _ = sample_data
        
        house_system.train_regression_model(X_train, y_price_train)
        X_test_scaled = house_system.scaler.transform(X_test)
        predictions = house_system.regression_model.predict(X_test_scaled)
        
        # Prices should be positive and in reasonable range
        assert (predictions > 0).all()
        assert predictions.min() > 10000  # At least $10k
        assert predictions.max() < 2000000  # Less than $2M


class TestClassificationModel:
    """Test classification model functionality"""
    
    def test_model_training(self, house_system, sample_data):
        """Test that classification model trains successfully"""
        X_train, _, _, _, y_tier_train, _ = sample_data
        
        # Need to train regression first to scale features
        house_system.scaler.fit(X_train)
        predictions = house_system.train_classification_model(X_train, y_tier_train)
        
        assert house_system.classification_model is not None
        assert len(predictions) == len(X_train)
    
    def test_prediction_output_type(self, house_system, sample_data):
        """Test that predictions are discrete categories"""
        X_train, X_test, _, _, y_tier_train, _ = sample_data
        
        house_system.scaler.fit(X_train)
        house_system.train_classification_model(X_train, y_tier_train)
        
        X_test_scaled = house_system.scaler.transform(X_test)
        predictions = house_system.classification_model.predict(X_test_scaled)
        
        # Classification outputs discrete categories
        expected_categories = ['Budget', 'Mid-Range', 'Luxury', 'Ultra-Luxury']
        assert all(pred in expected_categories for pred in predictions)
        assert len(np.unique(predictions)) <= 4  # Limited categories
    
    def test_probability_output(self, house_system, sample_data):
        """Test that model can output probabilities"""
        X_train, X_test, _, _, y_tier_train, _ = sample_data
        
        house_system.scaler.fit(X_train)
        house_system.train_classification_model(X_train, y_tier_train)
        
        X_test_scaled = house_system.scaler.transform(X_test)
        probabilities = house_system.classification_model.predict_proba(X_test_scaled)
        
        # Probabilities should sum to 1 and be valid
        # Number of classes depends on what's in training data (can be 3-4)
        assert probabilities.shape[1] >= 3  # At least 3 classes should be present
        assert probabilities.shape[1] <= 4  # Maximum 4 classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()


class TestModelEvaluation:
    """Test model evaluation functionality"""
    
    def test_evaluation_runs(self, house_system, sample_data):
        """Test that evaluation completes without errors"""
        X_train, X_test, y_price_train, y_price_test, y_tier_train, y_tier_test = sample_data
        
        house_system.train_regression_model(X_train, y_price_train)
        house_system.train_classification_model(X_train, y_tier_train)
        
        price_pred, tier_pred = house_system.evaluate_models(
            X_test, y_price_test, y_tier_test
        )
        
        assert price_pred is not None
        assert tier_pred is not None
        assert len(price_pred) == len(X_test)
        assert len(tier_pred) == len(X_test)
    
    def test_different_output_types(self, house_system, sample_data):
        """Test that regression and classification produce different output types"""
        X_train, X_test, y_price_train, y_price_test, y_tier_train, y_tier_test = sample_data
        
        house_system.train_regression_model(X_train, y_price_train)
        house_system.train_classification_model(X_train, y_tier_train)
        
        price_pred, tier_pred = house_system.evaluate_models(
            X_test, y_price_test, y_tier_test
        )
        
        # Regression: continuous numerical
        assert price_pred.dtype in [np.float64, np.float32]
        
        # Classification: discrete categories
        assert tier_pred.dtype == object or isinstance(tier_pred[0], str)


class TestKeyDifferences:
    """Test understanding of key differences"""
    
    def test_regression_continuous_output(self):
        """Verify regression produces continuous output"""
        # Two very similar houses should have very similar prices
        from lesson_code import HousePriceSystem
        system = HousePriceSystem(n_samples=10, random_state=42)
        
        df = system.generate_data()
        feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
        X = df[feature_cols]
        y_price = df['price']
        
        from sklearn.linear_model import LinearRegression
        system.scaler.fit(X)
        system.regression_model = LinearRegression()
        X_scaled = system.scaler.transform(X)
        system.regression_model.fit(X_scaled, y_price)
        
        predictions = system.regression_model.predict(X_scaled)
        
        # Should have many unique values (continuous)
        unique_ratio = len(np.unique(predictions)) / len(predictions)
        assert unique_ratio > 0.8  # Most predictions are unique
    
    def test_classification_discrete_output(self):
        """Verify classification produces discrete output"""
        from lesson_code import HousePriceSystem
        system = HousePriceSystem(n_samples=10, random_state=42)
        
        df = system.generate_data()
        feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
        X = df[feature_cols]
        y_tier = df['tier']
        
        from sklearn.linear_model import LogisticRegression
        system.scaler.fit(X)
        system.classification_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        X_scaled = system.scaler.transform(X)
        system.classification_model.fit(X_scaled, y_tier)
        
        predictions = system.classification_model.predict(X_scaled)
        
        # Should have limited unique values (discrete)
        unique_count = len(np.unique(predictions))
        assert unique_count <= 4  # Maximum 4 categories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
