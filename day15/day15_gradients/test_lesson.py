#!/usr/bin/env python3
"""
Day 15: Tests for Gradients and Gradient Descent
Verify your understanding with these tests
"""

import numpy as np
import sys
from lesson_code import GradientDescentVisualizer

class TestGradientDescent:
    """Test suite for gradient descent implementation"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def assert_close(self, actual, expected, tolerance=0.1, test_name="Test"):
        """Check if values are close enough"""
        if abs(actual - expected) <= tolerance:
            print(f"✅ {test_name}: PASSED")
            self.passed += 1
        else:
            print(f"❌ {test_name}: FAILED (got {actual:.3f}, expected {expected:.3f})")
            self.failed += 1
    
    def test_perfect_linear_data(self):
        """Test with perfect linear relationship"""
        print("\n🔍 Test 1: Perfect Linear Data")
        
        X = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1
        
        model = GradientDescentVisualizer(learning_rate=0.01)
        model.train(X, y, epochs=1000, verbose=False)
        
        # Should learn weight ≈ 2.0 and bias ≈ 1.0
        self.assert_close(model.weight, 2.0, 0.1, "Weight learning")
        self.assert_close(model.bias, 1.0, 0.1, "Bias learning")
        
        # Final loss should be very small
        final_loss = model.history['losses'][-1]
        if final_loss < 0.01:
            print("✅ Loss convergence: PASSED")
            self.passed += 1
        else:
            print(f"❌ Loss convergence: FAILED (loss = {final_loss:.4f})")
            self.failed += 1
    
    def test_gradient_calculation(self):
        """Test gradient calculation matches mathematical expectation"""
        print("\n🔍 Test 2: Gradient Calculation")
        
        model = GradientDescentVisualizer()
        model.weight = 1.0
        model.bias = 0.0
        
        X = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        
        weight_grad, bias_grad = model.compute_gradients(X, y)
        
        # With these values, gradients should be 0 (perfect fit)
        self.assert_close(weight_grad, 0.0, 0.001, "Weight gradient")
        self.assert_close(bias_grad, 0.0, 0.001, "Bias gradient")
    
    def test_prediction_accuracy(self):
        """Test prediction function"""
        print("\n🔍 Test 3: Prediction Function")
        
        model = GradientDescentVisualizer()
        model.weight = 2.0
        model.bias = 1.0
        
        X = np.array([1, 2, 3])
        predictions = model.predict(X)
        expected = np.array([3, 5, 7])  # 2*x + 1
        
        for i, (pred, exp) in enumerate(zip(predictions, expected)):
            self.assert_close(pred, exp, 0.001, f"Prediction {i+1}")
    
    def test_loss_decreases(self):
        """Test that loss decreases during training"""
        print("\n🔍 Test 4: Loss Improvement")
        
        X = np.array([1, 2, 3, 4])
        y = np.array([2, 4, 6, 8])
        
        model = GradientDescentVisualizer(learning_rate=0.01)
        initial_loss = model.compute_loss(X, y)
        
        model.train(X, y, epochs=100, verbose=False)
        final_loss = model.history['losses'][-1]
        
        if final_loss < initial_loss:
            print(f"✅ Loss improvement: PASSED ({initial_loss:.3f} → {final_loss:.3f})")
            self.passed += 1
        else:
            print(f"❌ Loss improvement: FAILED ({initial_loss:.3f} → {final_loss:.3f})")
            self.failed += 1
    
    def test_learning_rate_effects(self):
        """Test different learning rates"""
        print("\n🔍 Test 5: Learning Rate Effects")
        
        X = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        
        # Test reasonable learning rate
        model_good = GradientDescentVisualizer(learning_rate=0.01)
        model_good.train(X, y, epochs=100, verbose=False)
        good_loss = model_good.history['losses'][-1]
        
        # Test very small learning rate (should learn slower)
        model_slow = GradientDescentVisualizer(learning_rate=0.001)
        model_slow.train(X, y, epochs=100, verbose=False)
        slow_loss = model_slow.history['losses'][-1]
        
        if good_loss <= slow_loss:
            print("✅ Learning rate comparison: PASSED")
            self.passed += 1
        else:
            print("❌ Learning rate comparison: FAILED")
            self.failed += 1
    
    def run_all_tests(self):
        """Run all tests and report results"""
        print("🧪 Testing Gradient Descent Implementation")
        print("=" * 45)
        
        self.test_perfect_linear_data()
        self.test_gradient_calculation()
        self.test_prediction_accuracy()
        self.test_loss_decreases()
        self.test_learning_rate_effects()
        
        print("\n" + "=" * 45)
        print(f"📊 Test Results: {self.passed} passed, {self.failed} failed")
        
        if self.failed == 0:
            print("🎉 All tests passed! You understand gradient descent! 🚀")
            return True
        else:
            print("🔧 Some tests failed. Review the concepts and try again.")
            return False

def knowledge_check():
    """Interactive knowledge verification"""
    print("\n💭 Knowledge Check Questions")
    print("=" * 30)
    
    questions = [
        {
            "q": "What does a gradient tell us?",
            "options": [
                "A) The steepest uphill direction",
                "B) The current error value", 
                "C) The learning rate to use",
                "D) How many epochs to train"
            ],
            "correct": 0,
            "explanation": "The gradient points in the direction of steepest increase"
        },
        {
            "q": "Why do we go OPPOSITE to the gradient in gradient descent?",
            "options": [
                "A) To increase the error",
                "B) To minimize the error (go downhill)",
                "C) To speed up training",
                "D) To avoid overfitting"
            ],
            "correct": 1,
            "explanation": "We want to minimize error, so we go downhill (opposite to gradient)"
        },
        {
            "q": "What happens if learning rate is too high?",
            "options": [
                "A) Training is very slow",
                "B) Model converges faster", 
                "C) Model might not converge (bounces around)",
                "D) Nothing changes"
            ],
            "correct": 2,
            "explanation": "Too high learning rate causes instability and poor convergence"
        }
    ]
    
    score = 0
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question['q']}")
        for option in question['options']:
            print(f"   {option}")
        
        print("\nThink about your answer... (press Enter to see solution)")
        input()
        
        print(f"✅ Correct Answer: {question['options'][question['correct']]}")
        print(f"💡 Explanation: {question['explanation']}")

if __name__ == "__main__":
    # Run automated tests
    tester = TestGradientDescent()
    all_passed = tester.run_all_tests()
    
    # Run knowledge check
    knowledge_check()
    
    print("\n🎯 What You've Mastered:")
    print("   ✓ Gradient calculation and interpretation")
    print("   ✓ Gradient descent algorithm implementation") 
    print("   ✓ Learning rate effects on training")
    print("   ✓ Connection to real AI systems")
    
    if all_passed:
        print("\n🌟 Ready for the review week! Great work! 🌟")
    else:
        print("\n🔄 Review the failed tests and try again.")
