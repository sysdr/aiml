"""
Tests for Day 32: NumPy Array Manipulation and Vectorization
Run with: pytest test_lesson.py -v
"""

import numpy as np
import pytest
from lesson_code import (
    ImageBatchProcessor,
    WeightInitializer,
    PerformanceBenchmark,
    AdvancedIndexing
)


class TestImageBatchProcessor:
    """Test image preprocessing pipeline."""
    
    def test_normalize_output_range(self):
        """Normalized values should be roughly in [-3, 3]."""
        processor = ImageBatchProcessor()
        images = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
        
        normalized = processor.normalize(images)
        
        assert normalized.min() > -5
        assert normalized.max() < 5
        assert normalized.dtype == np.float32
    
    def test_prepare_batch_shape(self):
        """Output should be channels-first format."""
        processor = ImageBatchProcessor()
        images = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
        
        processed = processor.prepare_batch(images)
        
        # Should be (batch, channels, height, width)
        assert processed.shape == (8, 3, 224, 224)
    
    def test_extract_patches_count(self):
        """Correct number of patches should be extracted."""
        processor = ImageBatchProcessor()
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        patches = processor.extract_patches(image, patch_size=16)
        
        # 224/16 = 14 patches per dimension
        expected_patches = 14 * 14
        assert patches.shape[0] == expected_patches
        assert patches.shape[1:] == (16, 16, 3)
    
    def test_normalize_broadcasting(self):
        """Mean subtraction should use broadcasting correctly."""
        processor = ImageBatchProcessor()
        
        # Create uniform images
        images = np.full((2, 32, 32, 3), 128, dtype=np.uint8)
        normalized = processor.normalize(images)
        
        # Each channel should have different value due to different means
        channel_values = normalized[0, 0, 0, :]
        assert not np.allclose(channel_values[0], channel_values[1])


class TestWeightInitializer:
    """Test neural network weight initialization."""
    
    def test_xavier_bounds(self):
        """Xavier weights should be within expected bounds."""
        shape = (256, 128)
        weights = WeightInitializer.xavier_uniform(shape)
        
        fan_in, fan_out = shape
        expected_bound = np.sqrt(6.0 / (fan_in + fan_out))
        
        assert weights.min() >= -expected_bound
        assert weights.max() <= expected_bound
    
    def test_kaiming_variance(self):
        """Kaiming weights should have appropriate variance."""
        shape = (512, 256)
        weights = WeightInitializer.kaiming_normal(shape)
        
        # Variance should be approximately 2/fan_in
        expected_var = 2.0 / shape[0]
        actual_var = np.var(weights)
        
        # Allow 20% tolerance due to random sampling
        assert abs(actual_var - expected_var) / expected_var < 0.2
    
    def test_initialize_network_shapes(self):
        """Network weights should have correct shapes."""
        layer_sizes = [784, 256, 128, 10]
        weights = WeightInitializer.initialize_network(layer_sizes)
        
        assert len(weights) == 3
        assert weights[0].shape == (784, 256)
        assert weights[1].shape == (256, 128)
        assert weights[2].shape == (128, 10)
    
    def test_conv_weight_initialization(self):
        """Convolutional weight shapes should be handled correctly."""
        # Conv layer: 64 filters, 3 input channels, 3x3 kernel
        shape = (64, 3, 3, 3)
        weights = WeightInitializer.kaiming_normal(shape)
        
        assert weights.shape == shape
        assert weights.dtype == np.float32


class TestPerformanceBenchmark:
    """Test vectorization correctness (not just speed)."""
    
    def test_normalize_equivalence(self):
        """Loop and vectorized normalization should give same result."""
        data = np.random.randn(50, 50).astype(np.float32)
        
        loop_result = PerformanceBenchmark.normalize_loop(data)
        vec_result = PerformanceBenchmark.normalize_vectorized(data)
        
        np.testing.assert_allclose(loop_result, vec_result, rtol=1e-5)
    
    def test_matmul_equivalence(self):
        """Loop and vectorized matmul should give same result."""
        a = np.random.randn(20, 30).astype(np.float32)
        b = np.random.randn(30, 25).astype(np.float32)
        
        loop_result = PerformanceBenchmark.matrix_multiply_loop(a, b)
        vec_result = PerformanceBenchmark.matrix_multiply_vectorized(a, b)
        
        np.testing.assert_allclose(loop_result, vec_result, rtol=1e-4)
    
    def test_benchmark_runs(self):
        """Benchmark should complete without errors."""
        results = PerformanceBenchmark.run_benchmark(size=50)
        
        assert 'normalize' in results
        assert 'matmul' in results
        assert results['normalize']['speedup'] > 1
        assert results['matmul']['speedup'] > 1


class TestAdvancedIndexing:
    """Test advanced indexing operations."""
    
    def test_top_k_correct_values(self):
        """Top-k should return indices of highest values."""
        scores = np.array([0.1, 0.9, 0.3, 0.8, 0.5])
        top_2 = AdvancedIndexing.top_k_indices(scores, k=2)
        
        # Indices 1 and 3 have highest values
        assert set(top_2) == {1, 3}
        # Should be sorted descending
        assert scores[top_2[0]] >= scores[top_2[1]]
    
    def test_filter_threshold(self):
        """Filter should return values above threshold."""
        values = np.array([0.1, 0.95, 0.3, 0.99, 0.5])
        filtered, indices = AdvancedIndexing.filter_by_threshold(values, 0.9)
        
        assert len(filtered) == 2
        assert all(filtered > 0.9)
        assert set(indices) == {1, 3}
    
    def test_gather(self):
        """Gather should select correct elements."""
        data = np.array([10, 20, 30, 40, 50])
        indices = np.array([4, 0, 2])
        
        gathered = AdvancedIndexing.gather(data, indices)
        
        np.testing.assert_array_equal(gathered, [50, 10, 30])
    
    def test_scatter_add(self):
        """Scatter-add should accumulate at indices."""
        target = np.zeros(5)
        indices = np.array([1, 1, 3])
        values = np.array([1.0, 2.0, 5.0])
        
        result = AdvancedIndexing.scatter_add(target, indices, values)
        
        expected = np.array([0, 3, 0, 5, 0])
        np.testing.assert_array_equal(result, expected)
    
    def test_scatter_add_preserves_original(self):
        """Scatter-add should not modify original array."""
        target = np.zeros(3)
        original = target.copy()
        
        AdvancedIndexing.scatter_add(target, np.array([0]), np.array([1.0]))
        
        np.testing.assert_array_equal(target, original)


class TestIntegration:
    """Integration tests for realistic workflows."""
    
    def test_full_image_pipeline(self):
        """Complete image preprocessing should work end-to-end."""
        processor = ImageBatchProcessor()
        
        # Simulate loading batch
        batch = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
        
        # Preprocess
        processed = processor.prepare_batch(batch)
        
        # Should be ready for model input
        assert processed.shape == (16, 3, 224, 224)
        assert processed.dtype == np.float32
        assert not np.isnan(processed).any()
        assert not np.isinf(processed).any()
    
    def test_network_forward_pass_shapes(self):
        """Initialized network should support forward pass shapes."""
        layer_sizes = [784, 256, 64, 10]
        weights = WeightInitializer.initialize_network(layer_sizes)
        
        # Simulate input batch
        batch_size = 32
        x = np.random.randn(batch_size, 784).astype(np.float32)
        
        # Forward pass (without activations)
        for w in weights:
            x = x @ w
        
        # Output should be (batch_size, num_classes)
        assert x.shape == (batch_size, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
