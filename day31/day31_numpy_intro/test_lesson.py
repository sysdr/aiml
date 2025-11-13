"""
Tests for Day 31: Introduction to NumPy
"""

import numpy as np
import pytest
from lesson_code import ImagePreprocessor


class TestImagePreprocessor:
    """Test suite for image preprocessing pipeline"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance for testing"""
        return ImagePreprocessor(seed=42)
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for testing"""
        return np.random.randint(0, 256, size=(100, 64, 64, 3), dtype=np.uint8)
    
    def test_generate_synthetic_images(self, preprocessor):
        """Test synthetic image generation"""
        images = preprocessor.generate_synthetic_images(num_images=10, height=32, width=32)
        
        assert images.shape == (10, 32, 32, 3), "Image shape should be (10, 32, 32, 3)"
        assert images.dtype == np.uint8, "Images should be uint8"
        assert np.min(images) >= 0 and np.max(images) <= 255, "Pixel values should be in [0, 255]"
    
    def test_normalization(self, preprocessor, sample_images):
        """Test image normalization"""
        normalized = preprocessor.normalize_vectorized(sample_images)
        
        assert normalized.shape == sample_images.shape, "Shape should be preserved"
        assert normalized.dtype == np.float32, "Normalized images should be float32"
        assert np.min(normalized) >= 0.0 and np.max(normalized) <= 1.0, "Values should be in [0, 1]"
        
        # Check that normalization is correct
        expected = sample_images.astype(np.float32) / 255.0
        np.testing.assert_allclose(normalized, expected, rtol=1e-5)
    
    def test_standardization(self, preprocessor):
        """Test image standardization"""
        # Create images with known statistics
        images = np.random.randn(100, 32, 32, 3).astype(np.float32) * 50 + 100
        
        standardized = preprocessor.apply_standardization(images)
        
        # Check that mean is close to 0 and std is close to 1
        assert abs(np.mean(standardized)) < 0.1, "Mean should be close to 0"
        assert abs(np.std(standardized) - 1.0) < 0.1, "Std should be close to 1"
    
    def test_feature_extraction(self, preprocessor, sample_images):
        """Test feature extraction"""
        normalized = sample_images.astype(np.float32) / 255.0
        features = preprocessor.extract_features_vectorized(normalized)
        
        assert features.shape == (100, 3), "Should extract 3 features per image"
        assert np.all(np.isfinite(features)), "All features should be finite"
    
    def test_batch_processing(self, preprocessor, sample_images):
        """Test batch processing"""
        processed = preprocessor.batch_process(sample_images, batch_size=25)
        
        assert processed.shape == sample_images.shape, "Shape should be preserved"
        assert processed.dtype == np.float32, "Processed images should be float32"
        assert np.min(processed) >= 0.0 and np.max(processed) <= 1.0, "Values should be in [0, 1]"
    
    def test_save_and_load(self, preprocessor, tmp_path):
        """Test saving and loading preprocessed data"""
        data = np.random.randn(100, 10).astype(np.float32)
        filepath = tmp_path / "test_data.npy"
        
        # Save
        preprocessor.save_preprocessed_data(data, str(filepath))
        
        # Load
        loaded_data = np.load(str(filepath))
        
        assert loaded_data.shape == data.shape, "Loaded data should have same shape"
        np.testing.assert_array_equal(loaded_data, data, "Loaded data should match saved data")


class TestNumPyFundamentals:
    """Test fundamental NumPy operations"""
    
    def test_array_creation(self):
        """Test different ways to create arrays"""
        # From list
        arr1 = np.array([1, 2, 3, 4, 5])
        assert arr1.shape == (5,), "1D array shape"
        
        # 2D array
        arr2 = np.array([[1, 2], [3, 4]])
        assert arr2.shape == (2, 2), "2D array shape"
        
        # Zeros and ones
        zeros = np.zeros((3, 3))
        ones = np.ones((2, 4))
        assert zeros.shape == (3, 3) and ones.shape == (2, 4), "Shape initialization"
    
    def test_vectorized_operations(self):
        """Test vectorized operations are correct"""
        arr = np.array([1, 2, 3, 4, 5])
        
        # Arithmetic operations
        result = arr * 2 + 1
        expected = np.array([3, 5, 7, 9, 11])
        np.testing.assert_array_equal(result, expected)
        
        # Comparison operations
        mask = arr > 3
        expected_mask = np.array([False, False, False, True, True])
        np.testing.assert_array_equal(mask, expected_mask)
    
    def test_broadcasting(self):
        """Test broadcasting behavior"""
        # Matrix + vector
        matrix = np.ones((3, 4))
        vector = np.array([1, 2, 3, 4])
        
        result = matrix + vector
        assert result.shape == (3, 4), "Broadcasting preserves shape"
        assert np.all(result[0] == vector + 1), "Broadcasting applies correctly"
    
    def test_array_indexing(self):
        """Test array indexing and slicing"""
        arr = np.arange(20).reshape(4, 5)
        
        # Row indexing
        assert arr[0].shape == (5,), "Row indexing"
        
        # Column indexing
        assert arr[:, 0].shape == (4,), "Column indexing"
        
        # Slicing
        sub = arr[1:3, 2:4]
        assert sub.shape == (2, 2), "2D slicing"
        
        # Boolean indexing
        mask = arr > 10
        filtered = arr[mask]
        assert len(filtered) == 9, "Boolean indexing filters correctly"
    
    def test_reshaping(self):
        """Test array reshaping"""
        arr = np.arange(24)
        
        # Reshape to 2D
        reshaped = arr.reshape(4, 6)
        assert reshaped.shape == (4, 6), "Reshape to 2D"
        
        # Reshape with -1 (auto-calculate dimension)
        reshaped2 = arr.reshape(3, -1)
        assert reshaped2.shape == (3, 8), "Reshape with auto-dimension"
        
        # Flatten
        flat = reshaped.flatten()
        assert flat.shape == (24,), "Flatten back to 1D"
    
    def test_statistical_operations(self):
        """Test statistical operations"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Mean
        assert np.mean(arr) == 3.5, "Overall mean"
        assert np.allclose(np.mean(arr, axis=0), [2.5, 3.5, 4.5]), "Column-wise mean"
        assert np.allclose(np.mean(arr, axis=1), [2.0, 5.0]), "Row-wise mean"
        
        # Standard deviation
        std = np.std(arr)
        assert std > 0, "Std should be positive"
        
        # Min/Max
        assert np.min(arr) == 1, "Minimum value"
        assert np.max(arr) == 6, "Maximum value"


def test_performance_improvement():
    """Test that NumPy is significantly faster than Python lists"""
    size = 100000
    
    # NumPy operation
    arr = np.random.randn(size)
    import time
    start = time.time()
    result_np = arr ** 2 + 2 * arr + 1
    numpy_time = time.time() - start
    
    # Python list operation (smaller size to avoid timeout)
    lst = list(arr[:1000])
    start = time.time()
    result_py = [x**2 + 2*x + 1 for x in lst]
    python_time = time.time() - start
    
    # NumPy should be significantly faster (at least 10x for this operation)
    speedup = (python_time / numpy_time) * (size / 1000)
    assert speedup > 10, f"NumPy should be >10x faster, got {speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
