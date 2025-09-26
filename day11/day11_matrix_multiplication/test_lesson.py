"""
Test suite for Day 11: Matrix Multiplication and Dot Products
Run with: python -m pytest test_lesson.py -v
"""

import pytest
import numpy as np
from lesson_code import MatrixMultiplicationLab

class TestMatrixMultiplication:
    """Test matrix multiplication implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lab = MatrixMultiplicationLab()
    
    def test_manual_dot_product(self):
        """Test manual dot product implementation."""
        # Test basic functionality
        vector_a = [1, 2, 3]
        vector_b = [4, 5, 6]
        expected = 32  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        
        result = self.lab.manual_dot_product(vector_a, vector_b)
        assert result == expected
        
        # Verify against NumPy
        numpy_result = np.dot(vector_a, vector_b)
        assert result == numpy_result
    
    def test_manual_dot_product_edge_cases(self):
        """Test edge cases for dot product."""
        # Test with zeros
        zero_vector = [0, 0, 0]
        ones_vector = [1, 1, 1]
        result = self.lab.manual_dot_product(zero_vector, ones_vector)
        assert result == 0
        
        # Test with negative numbers
        neg_vector = [-1, -2, -3]
        pos_vector = [1, 2, 3]
        result = self.lab.manual_dot_product(neg_vector, pos_vector)
        assert result == -14  # -1*1 + -2*2 + -3*3 = -1 - 4 - 9 = -14
    
    def test_manual_dot_product_error_handling(self):
        """Test error handling for mismatched vector lengths."""
        vector_a = [1, 2, 3]
        vector_b = [4, 5]  # Different length
        
        with pytest.raises(ValueError, match="Vectors must have the same length"):
            self.lab.manual_dot_product(vector_a, vector_b)
    
    def test_manual_matrix_multiply(self):
        """Test manual matrix multiplication."""
        # Test 2x2 matrices
        matrix_a = [[1, 2], [3, 4]]
        matrix_b = [[5, 6], [7, 8]]
        
        result = self.lab.manual_matrix_multiply(matrix_a, matrix_b)
        expected = [[19, 22], [43, 50]]  # Standard matrix multiplication
        
        assert result == expected
        
        # Verify against NumPy
        numpy_result = np.dot(matrix_a, matrix_b).tolist()
        assert result == numpy_result
    
    def test_matrix_multiply_shapes(self):
        """Test matrix multiplication with different shapes."""
        # 2x3 × 3x2 = 2x2
        matrix_a = [[1, 2, 3], [4, 5, 6]]
        matrix_b = [[1, 2], [3, 4], [5, 6]]
        
        result = self.lab.manual_matrix_multiply(matrix_a, matrix_b)
        
        # Check dimensions
        assert len(result) == 2  # rows
        assert len(result[0]) == 2  # columns
        
        # Verify against NumPy
        numpy_result = np.dot(matrix_a, matrix_b).tolist()
        assert result == numpy_result
    
    def test_matrix_multiply_error_handling(self):
        """Test error handling for incompatible matrix dimensions."""
        matrix_a = [[1, 2, 3], [4, 5, 6]]  # 2x3
        matrix_b = [[1, 2], [3, 4]]        # 2x2 (incompatible)
        
        with pytest.raises(ValueError, match="Cannot multiply"):
            self.lab.manual_matrix_multiply(matrix_a, matrix_b)
    
    def test_similarity_detection(self):
        """Test similarity detection functionality."""
        similarities = self.lab.demonstrate_similarity_detection()
        
        # Check that we have similarity scores
        assert len(similarities) > 0
        
        # All similarities should be numeric
        for similarity in similarities.values():
            assert isinstance(similarity, (int, float))
    
    def test_recommendation_engine(self):
        """Test recommendation engine functionality."""
        recommendations = self.lab.build_recommendation_engine()
        
        # Check that all users have recommendations
        assert len(recommendations) == len(self.lab.users)
        
        # Each user should have recommendations for all movies
        for user_recs in recommendations.values():
            assert len(user_recs) == len(self.lab.movie_names)
            
            # Recommendations should be sorted (highest score first)
            scores = [score for _, score in user_recs]
            assert scores == sorted(scores, reverse=True)
    
    def test_neural_network_demo(self):
        """Test neural network demonstration."""
        output = self.lab.neural_network_demo()
        
        # Check output shape (2 examples, 4 hidden neurons)
        assert output.shape == (2, 4)
        
        # After ReLU activation, all values should be non-negative
        assert np.all(output >= 0)

class TestDataIntegrity:
    """Test data integrity and setup."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lab = MatrixMultiplicationLab()
    
    def test_movie_features_shape(self):
        """Test movie features matrix shape."""
        assert self.lab.movie_features.shape == (5, 5)  # 5 movies, 5 features
    
    def test_movie_names_count(self):
        """Test movie names count matches features."""
        assert len(self.lab.movie_names) == len(self.lab.movie_features)
    
    def test_user_preferences_shape(self):
        """Test user preferences have correct dimensions."""
        for user_prefs in self.lab.users.values():
            assert len(user_prefs) == 5  # 5 features to match movies
    
    def test_feature_values_range(self):
        """Test that feature values are in reasonable range."""
        # All movie features should be between 0 and 1
        assert np.all(self.lab.movie_features >= 0)
        assert np.all(self.lab.movie_features <= 1)
        
        # All user preferences should be between 0 and 1
        for user_prefs in self.lab.users.values():
            assert np.all(user_prefs >= 0)
            assert np.all(user_prefs <= 1)

def test_imports():
    """Test that all required imports work."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from typing import List, Tuple, Union
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
