"""
Test Suite for Day 46: Multiple Linear Regression
Comprehensive tests covering all functionality
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from lesson_code import MultipleLinearRegressionModel


@pytest.fixture
def model():
    """Fixture providing a fresh model instance"""
    return MultipleLinearRegressionModel()


@pytest.fixture
def sample_data(model):
    """Fixture providing sample training data"""
    df = model.generate_realistic_dataset(n_samples=100)
    X, y = model.prepare_data(df)
    return X, y


class TestDataGeneration:
    """Test dataset generation functionality"""
    
    def test_dataset_shape(self, model):
        """Test dataset has correct shape"""
        df = model.generate_realistic_dataset(n_samples=200)
        assert df.shape == (200, 9), "Dataset should have 200 rows and 9 columns"
    
    def test_dataset_columns(self, model):
        """Test dataset has all required columns"""
        df = model.generate_realistic_dataset(n_samples=50)
        expected_columns = [
            'square_feet', 'bedrooms', 'bathrooms', 'age_years',
            'location_score', 'has_pool', 'has_garage', 
            'distance_to_city_km', 'price'
        ]
        assert list(df.columns) == expected_columns, "Dataset missing required columns"
    
    def test_no_negative_prices(self, model):
        """Test all generated prices are positive"""
        df = model.generate_realistic_dataset(n_samples=200)
        assert (df['price'] > 0).all(), "All prices should be positive"
    
    def test_feature_ranges(self, model):
        """Test features are within realistic ranges"""
        df = model.generate_realistic_dataset(n_samples=200)
        
        assert (df['square_feet'] >= 800).all(), "Square feet should be >= 800"
        assert (df['square_feet'] <= 5000).all(), "Square feet should be <= 5000"
        assert (df['bedrooms'] >= 1).all(), "Bedrooms should be >= 1"
        assert (df['bedrooms'] <= 5).all(), "Bedrooms should be <= 5"
        assert (df['location_score'] >= 1).all(), "Location score should be >= 1"
        assert (df['location_score'] <= 10).all(), "Location score should be <= 10"
    
    def test_binary_features(self, model):
        """Test binary features contain only 0 and 1"""
        df = model.generate_realistic_dataset(n_samples=200)
        
        assert set(df['has_pool'].unique()).issubset({0, 1}), "has_pool should be binary"
        assert set(df['has_garage'].unique()).issubset({0, 1}), "has_garage should be binary"


class TestDataPreparation:
    """Test data preparation functionality"""
    
    def test_prepare_data_shapes(self, model):
        """Test prepare_data returns correct shapes"""
        df = model.generate_realistic_dataset(n_samples=100)
        X, y = model.prepare_data(df)
        
        assert X.shape == (100, 8), "X should have 100 rows and 8 features"
        assert y.shape == (100,), "y should have 100 values"
    
    def test_prepare_data_types(self, model):
        """Test prepare_data returns numpy arrays"""
        df = model.generate_realistic_dataset(n_samples=50)
        X, y = model.prepare_data(df)
        
        assert isinstance(X, np.ndarray), "X should be numpy array"
        assert isinstance(y, np.ndarray), "y should be numpy array"
    
    def test_no_missing_values(self, model):
        """Test prepared data has no missing values"""
        df = model.generate_realistic_dataset(n_samples=100)
        X, y = model.prepare_data(df)
        
        assert not np.isnan(X).any(), "X should not contain NaN values"
        assert not np.isnan(y).any(), "y should not contain NaN values"


class TestModelTraining:
    """Test model training functionality"""
    
    def test_train_updates_state(self, model, sample_data):
        """Test training updates model state"""
        X, y = sample_data
        assert not model.is_trained, "Model should not be trained initially"
        
        model.train(X, y)
        assert model.is_trained, "Model should be trained after fit"
    
    def test_train_returns_metrics(self, model, sample_data):
        """Test training returns valid metrics"""
        X, y = sample_data
        metrics = model.train(X, y)
        
        assert 'r2_score' in metrics, "Should return R² score"
        assert 'rmse' in metrics, "Should return RMSE"
        assert 'mae' in metrics, "Should return MAE"
        assert 0 <= metrics['r2_score'] <= 1, "R² should be between 0 and 1"
        assert metrics['rmse'] > 0, "RMSE should be positive"
        assert metrics['mae'] > 0, "MAE should be positive"
    
    def test_coefficients_learned(self, model, sample_data):
        """Test model learns coefficients"""
        X, y = sample_data
        model.train(X, y)
        
        assert len(model.model.coef_) == 8, "Should have 8 coefficients"
        assert model.model.intercept_ is not None, "Should have intercept"
    
    def test_training_quality(self, model, sample_data):
        """Test model achieves good training performance"""
        X, y = sample_data
        metrics = model.train(X, y)
        
        # With well-structured data, should achieve good R²
        assert metrics['r2_score'] > 0.7, "Training R² should be > 0.7"


class TestModelPrediction:
    """Test model prediction functionality"""
    
    def test_predict_before_training_raises_error(self, model):
        """Test prediction before training raises error"""
        X_test = np.random.rand(10, 8)
        
        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X_test)
    
    def test_predict_returns_correct_shape(self, model, sample_data):
        """Test prediction returns correct shape"""
        X, y = sample_data
        model.train(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape, "Predictions should match target shape"
    
    def test_predict_positive_prices(self, model, sample_data):
        """Test predictions are reasonable (mostly positive)"""
        X, y = sample_data
        model.train(X, y)
        
        predictions = model.predict(X)
        # At least 95% of predictions should be positive
        assert (predictions > 0).sum() / len(predictions) > 0.95, "Most predictions should be positive"
    
    def test_predict_single_sample(self, model, sample_data):
        """Test prediction on single sample"""
        X, y = sample_data
        model.train(X, y)
        
        single_prediction = model.predict(X[0].reshape(1, -1))
        assert single_prediction.shape == (1,), "Single prediction should have shape (1,)"
    
    def test_predict_multiple_samples(self, model, sample_data):
        """Test prediction on multiple samples"""
        X, y = sample_data
        model.train(X, y)
        
        predictions = model.predict(X[:10])
        assert predictions.shape == (10,), "Should predict for all 10 samples"


class TestModelEvaluation:
    """Test model evaluation functionality"""
    
    def test_evaluate_before_training_raises_error(self, model):
        """Test evaluation before training raises error"""
        X_test = np.random.rand(10, 8)
        y_test = np.random.rand(10)
        
        with pytest.raises(ValueError, match="must be trained"):
            model.evaluate(X_test, y_test)
    
    def test_evaluate_returns_all_metrics(self, model, sample_data):
        """Test evaluation returns all expected metrics"""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        expected_metrics = ['r2_score', 'rmse', 'mae', 'mape']
        for metric in expected_metrics:
            assert metric in metrics, f"Should return {metric}"
    
    def test_evaluation_metrics_valid(self, model, sample_data):
        """Test evaluation metrics are valid"""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        assert 0 <= metrics['r2_score'] <= 1, "R² should be between 0 and 1"
        assert metrics['rmse'] > 0, "RMSE should be positive"
        assert metrics['mae'] > 0, "MAE should be positive"
        assert 0 <= metrics['mape'] <= 100, "MAPE should be percentage"


class TestFeatureImportance:
    """Test feature importance functionality"""
    
    def test_get_importance_before_training_raises_error(self, model):
        """Test getting importance before training raises error"""
        with pytest.raises(ValueError, match="must be trained"):
            model.get_feature_importance()
    
    def test_feature_importance_shape(self, model, sample_data):
        """Test feature importance DataFrame has correct shape"""
        X, y = sample_data
        model.train(X, y)
        
        importance_df = model.get_feature_importance()
        assert importance_df.shape == (8, 3), "Should have 8 features and 3 columns"
    
    def test_feature_importance_columns(self, model, sample_data):
        """Test feature importance has required columns"""
        X, y = sample_data
        model.train(X, y)
        
        importance_df = model.get_feature_importance()
        expected_columns = ['feature', 'coefficient', 'abs_coefficient']
        assert list(importance_df.columns) == expected_columns, "Missing required columns"
    
    def test_feature_importance_sorted(self, model, sample_data):
        """Test feature importance is sorted by absolute value"""
        X, y = sample_data
        model.train(X, y)
        
        importance_df = model.get_feature_importance()
        abs_coefs = importance_df['abs_coefficient'].values
        
        # Check if sorted in descending order
        assert all(abs_coefs[i] >= abs_coefs[i+1] for i in range(len(abs_coefs)-1)), \
            "Features should be sorted by absolute coefficient"


class TestModelPersistence:
    """Test model save/load functionality"""
    
    def test_save_before_training_raises_error(self, model, tmp_path):
        """Test saving before training raises error"""
        filepath = tmp_path / "model.joblib"
        
        with pytest.raises(ValueError, match="must be trained"):
            model.save_model(str(filepath))
    
    def test_save_and_load_model(self, model, sample_data, tmp_path):
        """Test model can be saved and loaded"""
        X, y = sample_data
        model.train(X, y)
        
        # Save model
        filepath = tmp_path / "model.joblib"
        model.save_model(str(filepath))
        assert filepath.exists(), "Model file should be created"
        
        # Load model
        new_model = MultipleLinearRegressionModel()
        new_model.load_model(str(filepath))
        
        assert new_model.is_trained, "Loaded model should be marked as trained"
    
    def test_loaded_model_predictions_match(self, model, sample_data, tmp_path):
        """Test loaded model makes same predictions"""
        X, y = sample_data
        model.train(X, y)
        
        original_predictions = model.predict(X)
        
        # Save and load
        filepath = tmp_path / "model.joblib"
        model.save_model(str(filepath))
        
        new_model = MultipleLinearRegressionModel()
        new_model.load_model(str(filepath))
        loaded_predictions = new_model.predict(X)
        
        np.testing.assert_array_almost_equal(
            original_predictions, loaded_predictions,
            decimal=10,
            err_msg="Loaded model should make identical predictions"
        )


class TestProductionScenarios:
    """Test realistic production scenarios"""
    
    def test_large_dataset_performance(self, model):
        """Test model handles large datasets efficiently"""
        df = model.generate_realistic_dataset(n_samples=5000)
        X, y = model.prepare_data(df)
        
        # Should train without errors
        metrics = model.train(X, y)
        assert metrics['r2_score'] > 0.7, "Should maintain good performance at scale"
    
    def test_feature_coefficient_signs(self, model, sample_data):
        """Test learned coefficients have expected signs"""
        X, y = sample_data
        model.train(X, y)
        
        coef_dict = dict(zip(model.feature_names, model.model.coef_))
        
        # These features should increase price (positive coefficient)
        assert coef_dict['square_feet'] > 0, "Larger size should increase price"
        assert coef_dict['location_score'] > 0, "Better location should increase price"
        
        # These features should decrease price (negative coefficient)
        assert coef_dict['age_years'] < 0, "Older houses should decrease price"
        assert coef_dict['distance_to_city_km'] < 0, "Distance from city should decrease price"
    
    def test_prediction_consistency(self, model, sample_data):
        """Test predictions are consistent across calls"""
        X, y = sample_data
        model.train(X, y)
        
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        
        np.testing.assert_array_equal(
            pred1, pred2,
            err_msg="Predictions should be consistent"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
