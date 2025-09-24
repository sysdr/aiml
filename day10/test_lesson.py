"""
Test suite for Day 10: Matrices and Matrix Operations
Verifies that students understand key concepts
"""

import numpy as np
import unittest
from io import StringIO
import sys

class TestMatrixOperations(unittest.TestCase):
    """Test basic matrix operations understanding"""
    
    def setUp(self):
        """Set up test matrices"""
        self.matrix_a = np.array([[1, 2, 3], [4, 5, 6]])
        self.matrix_b = np.array([[7, 8, 9], [10, 11, 12]])
        self.small_matrix = np.array([[1, 2], [3, 4]])
    
    def test_matrix_creation(self):
        """Test that students can create matrices correctly"""
        # Test zeros matrix
        zeros = np.zeros((3, 4))
        self.assertEqual(zeros.shape, (3, 4))
        self.assertTrue(np.all(zeros == 0))
        
        # Test identity matrix
        identity = np.eye(3)
        self.assertEqual(identity.shape, (3, 3))
        self.assertEqual(identity[0, 0], 1)
        self.assertEqual(identity[0, 1], 0)
    
    def test_matrix_indexing(self):
        """Test matrix indexing operations"""
        # Test single element access
        self.assertEqual(self.matrix_a[0, 1], 2)
        self.assertEqual(self.matrix_a[1, 2], 6)
        
        # Test row access
        np.testing.assert_array_equal(self.matrix_a[0, :], [1, 2, 3])
        
        # Test column access
        np.testing.assert_array_equal(self.matrix_a[:, 1], [2, 5])
    
    def test_matrix_operations(self):
        """Test basic matrix operations"""
        # Test addition
        result = self.matrix_a + self.matrix_b
        expected = np.array([[8, 10, 12], [14, 16, 18]])
        np.testing.assert_array_equal(result, expected)
        
        # Test scalar multiplication
        result = self.small_matrix * 2
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_array_equal(result, expected)
        
        # Test transpose
        result = self.small_matrix.T
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(result, expected)
    
    def test_image_filter_simulation(self):
        """Test image filtering operations"""
        # Create simple test image
        image = np.array([[100, 150], [50, 200]])
        
        # Test brightness adjustment
        bright = np.clip(image + 50, 0, 255)
        dark = np.clip(image - 50, 0, 255)
        
        # Verify brightness changes
        self.assertGreater(np.mean(bright), np.mean(image))
        self.assertLess(np.mean(dark), np.mean(image))
        
        # Verify clipping works
        self.assertTrue(np.all(bright >= 0))
        self.assertTrue(np.all(bright <= 255))
    
    def test_matrix_properties(self):
        """Test understanding of matrix properties"""
        # Test shape
        self.assertEqual(self.matrix_a.shape, (2, 3))
        
        # Test size
        self.assertEqual(self.matrix_a.size, 6)
        
        # Test statistical operations
        self.assertEqual(np.sum(self.small_matrix), 10)
        self.assertEqual(np.mean(self.small_matrix), 2.5)

class TestAIConnections(unittest.TestCase):
    """Test understanding of AI connections"""
    
    def test_data_representation(self):
        """Test understanding of how AI represents data as matrices"""
        # Simulate image data
        image_height, image_width, channels = 224, 224, 3
        image_size = image_height * image_width * channels
        
        # Verify understanding of data sizes
        self.assertEqual(image_size, 150528)
        
        # Simulate neural network layer
        input_size, output_size = 784, 128
        weight_matrix_size = input_size * output_size
        self.assertEqual(weight_matrix_size, 100352)
    
    def test_matrix_shapes_compatibility(self):
        """Test understanding of matrix shape requirements"""
        # These shapes should be compatible for operations
        a = np.random.randn(3, 4)
        b = np.random.randn(3, 4)
        
        # Element-wise operations require same shape
        result = a + b
        self.assertEqual(result.shape, (3, 4))
        
        # Transpose changes shape
        a_t = a.T
        self.assertEqual(a_t.shape, (4, 3))

def run_student_challenges():
    """Interactive challenges for students"""
    print("🎯 STUDENT CHALLENGES")
    print("=" * 40)
    
    print("\nChallenge 1: Create a 5x5 identity matrix")
    identity_5x5 = np.eye(5)
    print("✅ Solution shape:", identity_5x5.shape)
    
    print("\nChallenge 2: Create a 3x3 matrix of random values")
    random_matrix = np.random.rand(3, 3)
    print("✅ Created random matrix with shape:", random_matrix.shape)
    
    print("\nChallenge 3: Extract the diagonal from a matrix")
    test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    diagonal = np.diag(test_matrix)
    print("✅ Diagonal elements:", diagonal)
    
    print("\nChallenge 4: Calculate row and column means")
    row_means = np.mean(test_matrix, axis=1)
    col_means = np.mean(test_matrix, axis=0)
    print("✅ Row means:", row_means)
    print("✅ Column means:", col_means)
    
    print("\n🎉 All challenges completed successfully!")

if __name__ == "__main__":
    print("🧪 Running Day 10 Matrix Operations Tests...\n")
    
    # Capture test output
    test_output = StringIO()
    runner = unittest.TextTestRunner(stream=test_output, verbosity=2)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMatrixOperations))
    suite.addTest(unittest.makeSuite(TestAIConnections))
    
    # Run tests
    result = runner.run(suite)
    
    # Display results
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print("\n" + "=" * 50)
    run_student_challenges()
