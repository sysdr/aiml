"""
Test file for Day 16-22: Linear Algebra & Calculus Review
Simple tests to verify understanding of core concepts
"""

import numpy as np
import unittest
from lesson_code import MathReviewSession

class TestMathReview(unittest.TestCase):
    """Test cases for mathematical concepts"""
    
    def setUp(self):
        self.review = MathReviewSession()
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        vec3 = np.array([0, 1, 0])
        
        # Identical vectors should have similarity 1
        self.assertAlmostEqual(self.review.cosine_similarity(vec1, vec2), 1.0, places=5)
        
        # Orthogonal vectors should have similarity 0
        self.assertAlmostEqual(self.review.cosine_similarity(vec1, vec3), 0.0, places=5)
    
    def test_vector_operations(self):
        """Test basic vector operations"""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        
        # Test dot product
        dot_product = np.dot(a, b)
        expected = 1*4 + 2*5 + 3*6  # 32
        self.assertEqual(dot_product, expected)
        
        # Test vector addition
        sum_vec = a + b
        expected_sum = np.array([5, 7, 9])
        np.testing.assert_array_equal(sum_vec, expected_sum)
    
    def test_matrix_operations(self):
        """Test matrix multiplication"""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[2, 0], [1, 2]])
        
        C = np.dot(A, B)
        expected = np.array([[4, 4], [10, 8]])
        
        np.testing.assert_array_equal(C, expected)
    
    def test_gradient_calculation(self):
        """Test gradient calculation for simple function"""
        # f(x) = x^2, f'(x) = 2x
        def gradient(x):
            return 2 * x
        
        # Test at x = 3
        grad = gradient(3)
        self.assertEqual(grad, 6)
        
        # Test gradient descent step
        x = 1.0
        learning_rate = 0.1
        new_x = x - learning_rate * gradient(x)
        expected_x = 1.0 - 0.1 * 2.0  # 0.8
        self.assertAlmostEqual(new_x, expected_x, places=5)
    
    def test_neural_network_forward_pass(self):
        """Test simple neural network computation"""
        # Input: 1 sample, 2 features
        X = np.array([[1.0, 2.0]])
        
        # Weights: 2 inputs to 1 output
        W = np.array([[0.5], [0.3]])
        
        # Bias
        b = np.array([0.1])
        
        # Forward pass
        Y = np.dot(X, W) + b
        expected = np.array([[1.0*0.5 + 2.0*0.3 + 0.1]])  # [[1.2]]
        
        np.testing.assert_array_almost_equal(Y, expected)

class TestAIApplications(unittest.TestCase):
    """Test AI-specific applications of math concepts"""
    
    def test_recommendation_similarity(self):
        """Test user similarity for recommendation systems"""
        # Similar users should have high similarity
        user_a = np.array([5, 4, 1, 5, 1])  # Likes action/sci-fi, dislikes drama/horror
        user_b = np.array([4, 5, 1, 4, 2])  # Similar preferences
        user_c = np.array([1, 1, 5, 1, 5])  # Opposite preferences
        
        review = MathReviewSession()
        
        sim_ab = review.cosine_similarity(user_a, user_b)
        sim_ac = review.cosine_similarity(user_a, user_c)
        
        # Similar users should be more similar than dissimilar users
        self.assertGreater(sim_ab, sim_ac)
    
    def test_data_normalization(self):
        """Test feature normalization for ML"""
        data = np.array([1, 2, 3, 4, 5])
        
        # Z-score normalization
        normalized = (data - np.mean(data)) / np.std(data)
        
        # Mean should be ~0, std should be ~1
        self.assertAlmostEqual(np.mean(normalized), 0, places=5)
        self.assertAlmostEqual(np.std(normalized), 1, places=5)
    
    def test_loss_function_minimum(self):
        """Test that we can find minimum of loss function"""
        def loss(x):
            return (x - 2)**2 + 1  # Minimum at x=2, value=1
        
        def gradient(x):
            return 2 * (x - 2)
        
        # Gradient descent
        x = 0.0
        learning_rate = 0.1
        
        for _ in range(100):  # Many steps to converge
            x = x - learning_rate * gradient(x)
        
        # Should converge close to minimum
        self.assertAlmostEqual(x, 2.0, places=1)
        self.assertAlmostEqual(loss(x), 1.0, places=1)

def run_tests():
    """Run all tests and display results"""
    print("🧪 Running Math Review Tests...")
    print("=" * 50)
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed! Your math foundation is solid.")
    else:
        print("❌ Some tests failed. Review the concepts and try again.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests()
