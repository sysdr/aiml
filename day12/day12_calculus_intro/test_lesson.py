#!/usr/bin/env python3
"""
Test suite for Day 12: Introduction to Calculus for AI/ML
Verify that all implementations work correctly
"""

import unittest
import numpy as np
from lesson_code import AICalculusToolkit

class TestCalculusToolkit(unittest.TestCase):
    """Test cases for the AI Calculus Toolkit"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.toolkit = AICalculusToolkit()
        self.tolerance = 1e-5
    
    def test_simple_function(self):
        """Test the basic quadratic function"""
        # Test known values
        self.assertAlmostEqual(self.toolkit.simple_function(0), 1, places=5)
        self.assertAlmostEqual(self.toolkit.simple_function(-1), 0, places=5)
        self.assertAlmostEqual(self.toolkit.simple_function(2), 9, places=5)
    
    def test_derivative_function(self):
        """Test analytical derivative calculations"""
        # Test known derivative values
        self.assertAlmostEqual(self.toolkit.derivative_function(0), 2, places=5)
        self.assertAlmostEqual(self.toolkit.derivative_function(-1), 0, places=5)
        self.assertAlmostEqual(self.toolkit.derivative_function(3), 8, places=5)
    
    def test_numerical_derivative(self):
        """Test numerical derivative approximation"""
        test_points = [0, 1, -2, 3.5]
        for x in test_points:
            analytical = self.toolkit.derivative_function(x)
            numerical = self.toolkit.numerical_derivative(self.toolkit.simple_function, x)
            self.assertAlmostEqual(analytical, numerical, places=5,
                                 msg=f"Derivative mismatch at x={x}")
    
    def test_gradient_descent_convergence(self):
        """Test that gradient descent finds the correct minimum"""
        minimum_x, _ = self.toolkit.gradient_descent_1d(
            self.toolkit.simple_function,
            self.toolkit.derivative_function,
            start_x=5,
            learning_rate=0.1,
            steps=100
        )
        
        # The true minimum is at x = -1
        self.assertAlmostEqual(minimum_x, -1, places=2,
                             msg="Gradient descent should converge to x=-1")
    
    def test_loss_function(self):
        """Test loss function calculations"""
        # Test squared error loss
        self.assertEqual(self.toolkit.loss_function(5, 3), 4)  # (5-3)^2 = 4
        self.assertEqual(self.toolkit.loss_function(2, 2), 0)  # Perfect prediction
        self.assertEqual(self.toolkit.loss_function(1, 4), 9)  # (1-4)^2 = 9
    
    def test_loss_derivative(self):
        """Test loss function derivative"""
        # Test derivative of squared error
        self.assertEqual(self.toolkit.loss_derivative(5, 3), 4)  # 2(5-3) = 4
        self.assertEqual(self.toolkit.loss_derivative(2, 2), 0)  # 2(2-2) = 0
        self.assertEqual(self.toolkit.loss_derivative(1, 4), -6) # 2(1-4) = -6
    
    def test_ai_learning_simulation(self):
        """Test AI learning simulation convergence"""
        history = self.toolkit.simulate_ai_learning(
            initial_prediction=8.0,
            target=3.0,
            learning_rate=0.1,
            steps=50
        )
        
        # Check that loss decreases over time
        initial_loss = history[0]['loss']
        final_loss = history[-1]['loss']
        self.assertLess(final_loss, initial_loss,
                       msg="Loss should decrease during learning")
        
        # Check that final prediction is close to target
        final_prediction = history[-1]['prediction']
        self.assertAlmostEqual(final_prediction, 3.0, places=1,
                             msg="Final prediction should be close to target")
    
    def test_compare_derivatives_accuracy(self):
        """Test derivative comparison functionality"""
        result = self.toolkit.compare_derivatives(2.5)
        
        # Check that all required keys are present
        required_keys = ['x', 'analytical', 'numerical', 'error']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check that error is small
        self.assertLess(result['error'], 1e-5,
                       msg="Error between analytical and numerical should be very small")

class TestMathematicalProperties(unittest.TestCase):
    """Test mathematical properties and edge cases"""
    
    def setUp(self):
        self.toolkit = AICalculusToolkit()
    
    def test_gradient_descent_different_learning_rates(self):
        """Test gradient descent with different learning rates"""
        learning_rates_results = [
            (0.01, 200, 0.3),  # (lr, steps, tolerance)
            (0.1, 50, 0.2),
            (0.5, 50, 0.2)
        ]
        
        for lr, steps, tolerance in learning_rates_results:
            minimum_x, history = self.toolkit.gradient_descent_1d(
                self.toolkit.simple_function,
                self.toolkit.derivative_function,
                start_x=3,
                learning_rate=lr,
                steps=steps
            )
            
            # All should converge within tolerance of -1
            self.assertAlmostEqual(minimum_x, -1, delta=tolerance,
                                 msg=f"Failed convergence with learning_rate={lr}")
    
    def test_function_minimum_verification(self):
        """Verify that x=-1 is indeed the minimum"""
        # Check that derivative is zero at x=-1
        derivative_at_min = self.toolkit.derivative_function(-1)
        self.assertAlmostEqual(derivative_at_min, 0, places=10)
        
        # Check that function value increases as we move away from -1
        min_value = self.toolkit.simple_function(-1)
        nearby_points = [-1.1, -0.9, -1.01, -0.99]
        
        for x in nearby_points:
            value = self.toolkit.simple_function(x)
            self.assertGreaterEqual(value, min_value,
                                  msg=f"Value at x={x} should be >= minimum")

def run_tests():
    """Run all tests and display results"""
    print("🧪 Running Day 12 Calculus Tests...")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestCalculusToolkit))
    suite.addTest(unittest.makeSuite(TestMathematicalProperties))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Display summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed! Your calculus implementation is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 50)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests()
