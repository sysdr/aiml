#!/usr/bin/env python3
"""
Test suite for Day 14: Chain Rule and Partial Derivatives
Verify your understanding with these interactive tests.
"""

import numpy as np
import sympy as sp
from lesson_code import ChainRuleDemo
import unittest

class TestChainRulePartialDerivatives(unittest.TestCase):
    """Test cases to verify understanding of chain rule and partial derivatives."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.demo = ChainRuleDemo()
        
    def test_sigmoid_function(self):
        """Test sigmoid function and its derivative."""
        # Test sigmoid properties
        self.assertAlmostEqual(self.demo.sigmoid(0), 0.5, places=6)
        self.assertTrue(0 < self.demo.sigmoid(-10) < 0.1)
        self.assertTrue(0.9 < self.demo.sigmoid(10) < 1)
        
        # Test sigmoid derivative at key points
        self.assertAlmostEqual(self.demo.sigmoid_derivative(0), 0.25, places=6)
        
        print("✅ Sigmoid function tests passed!")
    
    def test_chain_rule_symbolic(self):
        """Test chain rule with symbolic computation."""
        x = sp.symbols('x')
        
        # Test case: d/dx[sin(x²)] = cos(x²) × 2x
        g = x**2
        f_of_g = sp.sin(g)
        
        # Manual chain rule
        g_prime = sp.diff(g, x)  # 2x
        f_prime_of_g = sp.cos(g)  # cos(x²)
        manual_result = f_prime_of_g * g_prime
        
        # Automatic differentiation
        auto_result = sp.diff(f_of_g, x)
        
        # Simplify and compare
        manual_simplified = sp.simplify(manual_result)
        auto_simplified = sp.simplify(auto_result)
        
        self.assertEqual(manual_simplified, auto_simplified)
        print("✅ Chain rule symbolic test passed!")
    
    def test_partial_derivatives(self):
        """Test partial derivative calculations."""
        w1, w2, x1, x2, target = sp.symbols('w1 w2 x1 x2 target')
        
        # Loss function: L = (w1*x1 + w2*x2 - target)²
        prediction = w1*x1 + w2*x2
        loss = (prediction - target)**2
        
        # Calculate partial derivatives
        partial_w1 = sp.diff(loss, w1)
        partial_w2 = sp.diff(loss, w2)
        
        # Expected results
        expected_w1 = 2*(w1*x1 + w2*x2 - target)*x1
        expected_w2 = 2*(w1*x1 + w2*x2 - target)*x2
        
        self.assertEqual(sp.simplify(partial_w1 - expected_w1), 0)
        self.assertEqual(sp.simplify(partial_w2 - expected_w2), 0)
        print("✅ Partial derivatives test passed!")
    
    def test_numerical_gradient(self):
        """Test numerical gradient calculation."""
        # Simple function f(x, y) = x² + y²
        def f(x, y):
            return x**2 + y**2
        
        # Analytical gradients
        def df_dx(x, y):
            return 2*x
        
        def df_dy(x, y):
            return 2*y
        
        # Test point
        x, y = 3.0, 4.0
        
        # Numerical gradient (finite differences)
        h = 1e-7
        numerical_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
        numerical_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
        
        # Compare with analytical
        analytical_dx = df_dx(x, y)
        analytical_dy = df_dy(x, y)
        
        self.assertAlmostEqual(numerical_dx, analytical_dx, places=5)
        self.assertAlmostEqual(numerical_dy, analytical_dy, places=5)
        print("✅ Numerical gradient test passed!")

def interactive_quiz():
    """Interactive quiz to test understanding."""
    print("\n🧠 INTERACTIVE UNDERSTANDING CHECK")
    print("=" * 50)
    
    questions = [
        {
            "question": "What does the chain rule help us calculate?",
            "options": [
                "a) The derivative of simple functions",
                "b) The derivative of composite functions", 
                "c) The area under a curve",
                "d) The maximum of a function"
            ],
            "correct": "b",
            "explanation": "The chain rule helps us find derivatives of composite functions f(g(x))."
        },
        {
            "question": "In neural networks, what does backpropagation use?",
            "options": [
                "a) Only partial derivatives",
                "b) Only the chain rule",
                "c) Both chain rule and partial derivatives",
                "d) Integration"
            ],
            "correct": "c",
            "explanation": "Backpropagation uses chain rule to propagate errors backward and partial derivatives to handle multiple weights."
        },
        {
            "question": "If f(x,y) = x²y + xy², what is ∂f/∂x?",
            "options": [
                "a) 2xy + y²",
                "b) x² + 2xy",
                "c) 2x + 2y",
                "d) xy + xy"
            ],
            "correct": "a",
            "explanation": "∂f/∂x treats y as constant: ∂/∂x(x²y) = 2xy and ∂/∂x(xy²) = y²"
        }
    ]
    
    score = 0
    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}: {q['question']}")
        for option in q['options']:
            print(f"  {option}")
        
        while True:
            answer = input("\nYour answer (a/b/c/d): ").lower().strip()
            if answer in ['a', 'b', 'c', 'd']:
                break
            print("Please enter a, b, c, or d")
        
        if answer == q['correct']:
            print("✅ Correct!")
            score += 1
        else:
            print(f"❌ Incorrect. The correct answer is {q['correct']}")
        
        print(f"💡 Explanation: {q['explanation']}")
    
    print(f"\n🎯 Final Score: {score}/{len(questions)}")
    if score == len(questions):
        print("🎉 Perfect! You've mastered the concepts!")
    elif score >= len(questions) * 0.7:
        print("👍 Great job! You understand the key concepts.")
    else:
        print("📚 Review the lesson and try again. You've got this!")

def main():
    """Run all tests and interactive quiz."""
    print("🧪 Day 14: Chain Rule and Partial Derivatives - Test Suite")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=0)
    
    # Run interactive quiz
    interactive_quiz()
    
    print("\n🎓 Testing complete! Ready for Day 15: Gradients and Gradient Descent!")

if __name__ == "__main__":
    main()
