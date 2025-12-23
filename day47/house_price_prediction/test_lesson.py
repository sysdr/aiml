"""
Comprehensive test suite for Day 47: Housing Price Prediction
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import lesson code
from lesson_code import HousingDataGenerator, HousingPricePredictor


class TestDataGeneration:
    """Test data generation functionality"""
    
    def test_data_generator_creates_correct_size(self):
        """Test that generator creates requested number of samples"""
        generator = HousingDataGenerator(n_samples=100)
        df = generator.generate()
        assert len(df) == 100
    
    def test_data_generator_has_required_columns(self):
        """Test that generated data has all required columns"""
        generator = HousingDataGenerator(n_samples=50)
        df = generator.generate()
        required_cols = ["sqft", "bedrooms", "bathrooms", "lot_size", "age", "garage", "price"]
        assert all(col in df.columns for col in required_cols)
    
    def test_data_generator_realistic_ranges(self):
        """Test that generated data has realistic value ranges"""
        generator = HousingDataGenerator(n_samples=100)
        df = generator.generate()
        
        assert df["sqft"].min() >= 500
        assert df["sqft"].max() <= 6000
        assert df["bedrooms"].min() >= 1
        assert df["bedrooms"].max() <= 6
        assert df["price"].min() >= 50000
        assert df["price"].max() <= 2000000


class TestModelTraining:
    """Test model training functionality"""
    
    def test_model_training_sets_fitted_flag(self):
        """Test that training sets the is_fitted flag"""
        generator = HousingDataGenerator(n_samples=100)
        df = generator.generate()
        
        predictor = HousingPricePredictor()
        df_processed = predictor.engineer_features(df)
        X_train, _, _, y_train, _, _ = predictor.prepare_data(df_processed)
        
        assert not predictor.is_fitted
        predictor.train(X_train, y_train)
        assert predictor.is_fitted


class TestPrediction:
    """Test prediction functionality"""
    
    def test_prediction_requires_trained_model(self):
        """Test that prediction fails without training"""
        predictor = HousingPricePredictor()
        
        features = {
            "sqft": 2000,
            "bedrooms": 3,
            "bathrooms": 2,
            "lot_size": 0.25,
            "age": 10,
            "garage": 2
        }
        
        with pytest.raises(ValueError, match="Model not trained"):
            predictor.predict(features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
