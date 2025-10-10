"""
Tests for Day 13: Derivatives and Their Applications
Simple tests to verify understanding and implementation
"""

import numpy as np
import pytest
from lesson_code import SimplePredictor, DerivativeCalculator, HousePricePredictor

def test_simple_predictor_initialization():
    """Test that SimplePredictor initializes correctly"""
    predictor = SimplePredictor()
    
    # Check that parameters are initialized
    assert hasattr(predictor, 'weight')
    assert hasattr(predictor, 'bias')
    assert hasattr(predictor, 'learning_rate')
    
    # Check that they are numbers
    assert isinstance(predictor.weight, (int, float, np.number))
    assert isinstance(predictor.bias, (int, float, np.number))
    
    print("✅ SimplePredictor initialization test passed")

def test_prediction_functionality():
    """Test that predictions work correctly"""
    predictor = SimplePredictor()
    predictor.weight = 2.0
    predictor.bias = 1.0
    
    # Test single prediction
    x = np.array([3.0])
    result = predictor.predict(x)
    expected = 2.0 * 3.0 + 1.0  # weight * x + bias = 7.0
    
    assert np.isclose(result[0], expected), f"Expected {expected}, got {result[0]}"
    
    # Test multiple predictions
    x = np.array([1.0, 2.0, 3.0])
    results = predictor.predict(x)
    expected = np.array([3.0, 5.0, 7.0])
    
    assert np.allclose(results, expected), f"Expected {expected}, got {results}"
    
    print("✅ Prediction functionality test passed")

def test_gradient_calculation():
    """Test that gradients are calculated correctly"""
    predictor = SimplePredictor()
    predictor.weight = 1.0
    predictor.bias = 0.0
    
    # Simple test case where we know the answer
    x = np.array([1.0, 2.0])
    y_true = np.array([2.0, 4.0])  # Perfect linear relationship
    
    weight_grad, bias_grad = predictor.calculate_gradients(x, y_true)
    
    # With perfect data, gradients should be small
    assert abs(weight_grad) < 1.0, f"Weight gradient too large: {weight_grad}"
    assert abs(bias_grad) < 1.0, f"Bias gradient too large: {bias_grad}"
    
    print("✅ Gradient calculation test passed")

def test_training_reduces_error():
    """Test that training actually reduces the error"""
    predictor = SimplePredictor(learning_rate=0.01)
    
    # Generate simple linear data
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y_true = np.array([2.0, 4.0, 6.0, 8.0])  # y = 2x
    
    # Get initial error
    initial_error = predictor.calculate_error(x, y_true)
    
    # Train for a few steps
    for _ in range(10):
        predictor.train_step(x, y_true)
    
    # Get final error
    final_error = predictor.calculate_error(x, y_true)
    
    # Error should decrease
    assert final_error < initial_error, f"Error increased: {initial_error} -> {final_error}"
    
    print("✅ Training reduces error test passed")

def test_derivative_calculator():
    """Test the derivative calculator utility"""
    calc = DerivativeCalculator()
    
    # Test analytical derivative of x²
    def quadratic(x):
        return x**2
    
    # Test at x = 3, derivative should be 2x = 6
    numerical = calc.numerical_derivative(quadratic, 3.0)
    analytical = 2 * 3.0  # d/dx(x²) = 2x
    
    # Should be very close
    assert abs(numerical - analytical) < 1e-5, f"Derivatives don't match: {numerical} vs {analytical}"
    
    print("✅ Derivative calculator test passed")

def test_house_price_predictor():
    """Test the house price predictor"""
    house_predictor = HousePricePredictor()
    
    # Test data generation
    x, y = house_predictor.generate_sample_data(50)
    
    assert len(x) == 50, f"Wrong number of samples: {len(x)}"
    assert len(y) == 50, f"Wrong number of samples: {len(y)}"
    assert np.all(x > 0), "Square footage should be positive"
    assert np.all(y > 0), "Prices should be positive"
    
    print("✅ House price predictor test passed")

def run_all_tests():
    """Run all tests"""
    print("🧪 Running Day 13 Tests...")
    print("=" * 40)
    
    test_simple_predictor_initialization()
    test_prediction_functionality()
    test_gradient_calculation()
    test_training_reduces_error()
    test_derivative_calculator()
    test_house_price_predictor()
    
    print("\n🎉 All tests passed! Your implementation is working correctly.")
    print("💡 Understanding check:")
    print("   • Can you explain why we subtract the gradient from parameters?")
    print("   • What happens if the learning rate is too large or too small?")
    print("   • How do derivatives help the AI 'learn' from mistakes?")

if __name__ == "__main__":
    run_all_tests()
