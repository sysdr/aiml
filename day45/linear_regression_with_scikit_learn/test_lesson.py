"""
Day 45: Linear Regression with Scikit-learn - Test Suite
Comprehensive tests to verify understanding and implementation
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys


class TestDataPreparation:
    """Test data creation and preparation"""
    
    def test_data_creation(self):
        """Verify sample data can be created"""
        from lesson_code import create_sample_data
        df = create_sample_data()
        
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) > 0, "Should have data"
        assert 'YearsExperience' in df.columns, "Should have YearsExperience column"
        assert 'Salary' in df.columns, "Should have Salary column"
        
    def test_data_shape(self):
        """Verify data has correct shape"""
        from lesson_code import create_sample_data
        df = create_sample_data()
        
        assert df.shape[1] == 2, "Should have 2 columns"
        assert df.shape[0] >= 20, "Should have at least 20 samples"
    
    def test_no_missing_values(self):
        """Verify no missing values in data"""
        from lesson_code import create_sample_data
        df = create_sample_data()
        
        assert df.isnull().sum().sum() == 0, "Should have no missing values"
    
    def test_train_test_split(self):
        """Verify train/test split works correctly"""
        from lesson_code import create_sample_data, prepare_data
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2)
        
        total_samples = len(df)
        expected_test = int(total_samples * 0.2)
        
        assert len(X_test) == expected_test, f"Test set should have {expected_test} samples"
        assert len(X_train) == total_samples - expected_test, "Train set should have remaining samples"


class TestModelTraining:
    """Test model creation and training"""
    
    def test_model_creation(self):
        """Verify LinearRegression model can be created"""
        model = LinearRegression()
        assert model is not None, "Model should be created"
        
    def test_model_training(self):
        """Verify model can be trained"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        assert hasattr(model, 'coef_'), "Model should have coefficients after fitting"
        assert hasattr(model, 'intercept_'), "Model should have intercept after fitting"
        
    def test_learned_parameters(self):
        """Verify learned parameters are reasonable"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        # Coefficient should be positive (more experience → higher salary)
        assert model.coef_[0] > 0, "Coefficient should be positive"
        
        # Intercept should be reasonable base salary
        assert model.intercept_ > 20000, "Intercept should represent reasonable base salary"
        assert model.intercept_ < 50000, "Intercept should be realistic"


class TestModelPredictions:
    """Test model predictions and accuracy"""
    
    def test_predictions_shape(self):
        """Verify predictions have correct shape"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test), "Should predict for all test samples"
        
    def test_predictions_are_numeric(self):
        """Verify predictions are valid numbers"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert np.all(np.isfinite(predictions)), "All predictions should be finite numbers"
        assert np.all(predictions > 0), "Salary predictions should be positive"
        
    def test_r2_score_quality(self):
        """Verify model achieves good R² score"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        assert r2 > 0.6, f"R² score ({r2:.4f}) should be > 0.6 for reasonable fit"
        
    def test_no_overfitting(self):
        """Verify model doesn't overfit badly"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        
        gap = train_r2 - test_r2
        assert gap < 0.15, f"R² gap ({gap:.4f}) should be < 0.15 to avoid overfitting"


class TestNewPredictions:
    """Test predictions on new data"""
    
    def test_single_prediction(self):
        """Verify can predict for single new value"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        new_X = np.array([[5.0]])
        prediction = model.predict(new_X)
        
        assert len(prediction) == 1, "Should return single prediction"
        assert prediction[0] > 0, "Prediction should be positive"
        
    def test_multiple_predictions(self):
        """Verify can predict for multiple new values"""
        from lesson_code import create_sample_data, prepare_data, train_model, make_predictions
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        predictions = make_predictions(model, [3.0, 5.0, 7.0])
        
        assert len(predictions) == 3, "Should return 3 predictions"
        assert all(p > 0 for p in predictions), "All predictions should be positive"
        
    def test_prediction_ordering(self):
        """Verify more experience predicts higher salary"""
        from lesson_code import create_sample_data, prepare_data, train_model, make_predictions
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        predictions = make_predictions(model, [2.0, 5.0, 8.0])
        
        assert predictions[0] < predictions[1] < predictions[2], \
            "Predictions should increase with experience"


class TestMetrics:
    """Test evaluation metrics"""
    
    def test_mse_calculation(self):
        """Verify MSE can be calculated"""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        mse = mean_squared_error(y_true, y_pred)
        expected = ((10**2) + (10**2) + (10**2)) / 3
        
        assert abs(mse - expected) < 0.01, "MSE calculation should be correct"
        
    def test_r2_range(self):
        """Verify R² is in valid range"""
        from lesson_code import create_sample_data, prepare_data, train_model, evaluate_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        assert -1 <= metrics['test_r2'] <= 1, "R² should be between -1 and 1"


class TestFileOperations:
    """Test file creation and saving"""
    
    def test_data_file_created(self):
        """Verify CSV file is created"""
        from lesson_code import create_sample_data
        df = create_sample_data()
        df.to_csv('test_salary_data.csv', index=False)
        
        assert os.path.exists('test_salary_data.csv'), "CSV file should be created"
        
        # Cleanup
        os.remove('test_salary_data.csv')
        
    def test_model_can_be_saved(self):
        """Verify model can be saved with joblib"""
        import joblib
        from lesson_code import create_sample_data, prepare_data, train_model
        
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        joblib.dump(model, 'test_model.pkl')
        assert os.path.exists('test_model.pkl'), "Model file should be created"
        
        # Verify can be loaded
        loaded_model = joblib.load('test_model.pkl')
        assert loaded_model.coef_[0] == model.coef_[0], "Loaded model should match original"
        
        # Cleanup
        os.remove('test_model.pkl')


class TestProductionReadiness:
    """Test production-ready features"""
    
    def test_reproducibility(self):
        """Verify results are reproducible with random_state"""
        from lesson_code import create_sample_data, prepare_data, train_model
        
        # Run 1
        df1 = create_sample_data()
        X_train1, X_test1, y_train1, y_test1 = prepare_data(df1, random_state=42)
        model1 = train_model(X_train1, y_train1)
        
        # Run 2
        df2 = create_sample_data()
        X_train2, X_test2, y_train2, y_test2 = prepare_data(df2, random_state=42)
        model2 = train_model(X_train2, y_train2)
        
        assert np.allclose(model1.coef_, model2.coef_), "Results should be reproducible"
        assert np.allclose(model1.intercept_, model2.intercept_), "Results should be reproducible"
        
    def test_handles_edge_cases(self):
        """Verify model handles edge case inputs"""
        from lesson_code import create_sample_data, prepare_data, train_model
        df = create_sample_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        
        # Test with zero experience
        pred_zero = model.predict([[0]])
        assert pred_zero[0] > 0, "Should handle zero experience"
        
        # Test with high experience
        pred_high = model.predict([[20]])
        assert pred_high[0] > pred_zero[0], "Should handle high experience values"


def run_tests():
    """Run all tests with detailed output"""
    print("=" * 60)
    print("Running Day 45 Test Suite")
    print("=" * 60)
    
    pytest_args = [
        __file__,
        '-v',  # Verbose
        '--tb=short',  # Short traceback format
        '--color=yes'  # Colored output
    ]
    
    result = pytest.main(pytest_args)
    
    if result == 0:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nYou've successfully mastered:")
        print("  ✓ Data preparation and train/test splitting")
        print("  ✓ Training scikit-learn linear regression models")
        print("  ✓ Making predictions and evaluating accuracy")
        print("  ✓ Production-ready implementation patterns")
        print("\nReady for Day 46: Multiple Linear Regression!")
    
    return result


if __name__ == "__main__":
    run_tests()
