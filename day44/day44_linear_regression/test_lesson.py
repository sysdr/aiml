"""
Day 44: Simple Linear Regression Theory - Comprehensive Test Suite

Tests cover mathematical correctness, edge cases, and production scenarios.
"""

import pytest
import numpy as np
from lesson_code import SimpleLinearRegression, generate_sample_data


class TestLinearRegressionBasics:
    """Test basic functionality and mathematical correctness."""
    
    def test_initialization(self):
        """Test model initializes with correct default parameters."""
        model = SimpleLinearRegression()
        assert model.learning_rate == 0.01
        assert model.iterations == 1000
        assert model.w == 0.0
        assert model.b == 0.0
        assert model.loss_history == []
    
    def test_custom_initialization(self):
        """Test model accepts custom hyperparameters."""
        model = SimpleLinearRegression(learning_rate=0.05, iterations=500)
        assert model.learning_rate == 0.05
        assert model.iterations == 500
    
    def test_perfect_linear_data(self):
        """Test model learns perfect linear relationship without noise."""
        # Generate perfect linear data: y = 2x + 3
        X = np.array([1, 2, 3, 4, 5])
        y = 2 * X + 3
        
        model = SimpleLinearRegression(learning_rate=0.01, iterations=1000)
        model.fit(X, y)
        
        # Should learn near-perfect parameters
        assert abs(model.w - 2.0) < 0.1, f"Expected w≈2.0, got {model.w}"
        assert abs(model.b - 3.0) < 0.5, f"Expected b≈3.0, got {model.b}"
        
        # Loss should be near zero
        assert model.loss_history[-1] < 0.01
    
    def test_noisy_data_convergence(self):
        """Test model handles realistic noisy data."""
        X, y = generate_sample_data(n_samples=100, noise=10.0)
        
        model = SimpleLinearRegression(learning_rate=0.01, iterations=1000)
        model.fit(X, y)
        
        # Loss should decrease significantly
        initial_loss = model.loss_history[0]
        final_loss = model.loss_history[-1]
        assert final_loss < initial_loss * 0.1, "Loss should decrease by at least 90%"
        
        # Parameters should be reasonable for house price data
        assert 100 < model.w < 200, f"Weight out of expected range: {model.w}"
        assert 0 < model.b < 100000, f"Bias out of expected range: {model.b}"
    
    def test_negative_correlation(self):
        """Test model learns negative slopes correctly."""
        # Negative relationship: y = -3x + 100
        X = np.array([1, 2, 3, 4, 5])
        y = -3 * X + 100
        
        model = SimpleLinearRegression(learning_rate=0.01, iterations=1000)
        model.fit(X, y)
        
        assert model.w < 0, "Should learn negative slope"
        assert abs(model.w - (-3.0)) < 1.0
        assert abs(model.b - 100.0) < 5.0
    
    def test_predictions_shape(self):
        """Test predictions return correct shape."""
        X_train = np.array([1, 2, 3, 4, 5])
        y_train = 2 * X_train + 3
        
        model = SimpleLinearRegression()
        model.fit(X_train, y_train)
        
        # Single prediction
        pred_single = model.predict([10])
        assert pred_single.shape == (1,)
        
        # Multiple predictions
        X_test = np.array([6, 7, 8, 9, 10])
        pred_multiple = model.predict(X_test)
        assert pred_multiple.shape == (5,)
    
    def test_prediction_accuracy(self):
        """Test predictions are mathematically correct."""
        X = np.array([1, 2, 3, 4, 5])
        y = 2 * X + 3
        
        model = SimpleLinearRegression(learning_rate=0.01, iterations=2000)
        model.fit(X, y)
        
        # Test prediction for new point
        pred = model.predict([10])
        expected = 2 * 10 + 3  # = 23
        assert abs(pred[0] - expected) < 0.1


class TestGradientDescent:
    """Test gradient descent optimization mechanics."""
    
    def test_loss_decreases_monotonically(self):
        """Test loss decreases consistently during training."""
        X, y = generate_sample_data(n_samples=100, noise=5.0)
        
        model = SimpleLinearRegression(learning_rate=0.01, iterations=500)
        model.fit(X, y)
        
        # Check loss trend over windows
        window_size = 50
        for i in range(0, len(model.loss_history) - window_size, window_size):
            early_avg = np.mean(model.loss_history[i:i+window_size])
            late_avg = np.mean(model.loss_history[i+window_size:i+2*window_size])
            assert late_avg <= early_avg, "Loss should decrease over time"
    
    def test_loss_history_length(self):
        """Test loss history tracks all iterations."""
        iterations = 500
        model = SimpleLinearRegression(iterations=iterations)
        
        X = np.array([1, 2, 3, 4, 5])
        y = 2 * X + 3
        model.fit(X, y)
        
        assert len(model.loss_history) == iterations
    
    def test_learning_rate_impact(self):
        """Test different learning rates affect convergence."""
        X = np.array([1, 2, 3, 4, 5])
        y = 2 * X + 3
        
        # Low learning rate
        model_slow = SimpleLinearRegression(learning_rate=0.001, iterations=1000)
        model_slow.fit(X, y)
        
        # Optimal learning rate
        model_fast = SimpleLinearRegression(learning_rate=0.01, iterations=1000)
        model_fast.fit(X, y)
        
        # Fast should converge better in same iterations
        assert model_fast.loss_history[-1] < model_slow.loss_history[-1]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_data_point(self):
        """Test handling of minimal dataset."""
        X = np.array([5])
        y = np.array([10])
        
        model = SimpleLinearRegression(iterations=100)
        model.fit(X, y)
        
        # Should complete without errors
        assert len(model.loss_history) == 100
        # With one point, infinite solutions exist
        pred = model.predict([5])
        assert abs(pred[0] - 10) < 1.0
    
    def test_identical_x_values(self):
        """Test handling of no variance in X."""
        X = np.array([5, 5, 5, 5, 5])
        y = np.array([10, 11, 10, 12, 11])
        
        model = SimpleLinearRegression(iterations=100)
        # Should complete without division by zero
        model.fit(X, y)
        assert len(model.loss_history) == 100
    
    def test_zero_values(self):
        """Test handling of zero in data."""
        X = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 2, 4, 6, 8])
        
        model = SimpleLinearRegression(learning_rate=0.01, iterations=1000)
        model.fit(X, y)
        
        # Should learn y = 2x
        assert abs(model.w - 2.0) < 0.1
        assert abs(model.b) < 0.5
    
    def test_large_scale_data(self):
        """Test performance with larger dataset."""
        X, y = generate_sample_data(n_samples=1000, noise=10.0)
        
        model = SimpleLinearRegression(learning_rate=0.01, iterations=1000)
        model.fit(X, y)
        
        # Should still converge well
        assert model.loss_history[-1] < model.loss_history[0] * 0.1


class TestMathematicalCorrectness:
    """Test mathematical formulas are implemented correctly."""
    
    def test_mse_calculation(self):
        """Test MSE formula is correct."""
        model = SimpleLinearRegression()
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        mse = model._calculate_mse(y_true, y_pred)
        
        # Manual calculation
        expected = np.mean((y_true - y_pred) ** 2)
        assert abs(mse - expected) < 1e-10
    
    def test_gradient_formulas(self):
        """Test gradient calculations are mathematically correct."""
        model = SimpleLinearRegression()
        
        X = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        y_pred = np.array([2.5, 4.5, 6.5])
        
        grad_w, grad_b = model._calculate_gradients(X, y, y_pred)
        
        # Manual calculation
        n = len(y)
        error = y - y_pred
        expected_grad_w = -(2/n) * np.sum(X * error)
        expected_grad_b = -(2/n) * np.sum(error)
        
        assert abs(grad_w - expected_grad_w) < 1e-10
        assert abs(grad_b - expected_grad_b) < 1e-10
    
    def test_prediction_formula(self):
        """Test prediction uses correct linear formula."""
        model = SimpleLinearRegression()
        model.w = 3.0
        model.b = 5.0
        
        X = np.array([1, 2, 3, 4])
        predictions = model.predict(X)
        
        # Manual calculation: y = 3x + 5
        expected = 3.0 * X + 5.0
        np.testing.assert_array_almost_equal(predictions, expected)


class TestDataGeneration:
    """Test sample data generation utilities."""
    
    def test_data_generation_shape(self):
        """Test generated data has correct shape."""
        X, y = generate_sample_data(n_samples=100)
        assert X.shape == (100,)
        assert y.shape == (100,)
    
    def test_data_generation_range(self):
        """Test generated data is in expected range."""
        X, y = generate_sample_data(n_samples=100)
        assert X.min() >= 1000
        assert X.max() <= 3000
    
    def test_data_reproducibility(self):
        """Test data generation is reproducible with same seed."""
        X1, y1 = generate_sample_data(n_samples=50)
        X2, y2 = generate_sample_data(n_samples=50)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestProductionScenarios:
    """Test scenarios mimicking production use cases."""
    
    def test_train_once_predict_many(self):
        """Test typical production pattern: train once, predict many times."""
        X_train, y_train = generate_sample_data(n_samples=100)
        
        # Training phase (expensive, done once)
        model = SimpleLinearRegression(iterations=1000)
        model.fit(X_train, y_train)
        
        # Prediction phase (cheap, done millions of times)
        X_test = np.random.uniform(1000, 3000, 10000)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (10000,)
        assert np.all(predictions > 0), "All house prices should be positive"
    
    def test_parameter_extraction(self):
        """Test extracting parameters for model deployment."""
        X = np.array([1, 2, 3, 4, 5])
        y = 2 * X + 3
        
        model = SimpleLinearRegression(iterations=1000)
        model.fit(X, y)
        
        w, b = model.get_parameters()
        assert isinstance(w, float)
        assert isinstance(b, float)
        
        # Can be serialized for deployment
        import json
        params_json = json.dumps({"weight": w, "bias": b})
        assert len(params_json) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
