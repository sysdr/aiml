"""
Day 32: NumPy Array Manipulation and Vectorization
Production-ready techniques used at Tesla, Google, and OpenAI
"""

import numpy as np
import time
from typing import Tuple

# =============================================================================
# IMAGE BATCH PROCESSOR
# The exact preprocessing pipeline used for ResNet, VGG, and similar models
# =============================================================================

class ImageBatchProcessor:
    """
    Processes batches of images using vectorized operations.
    This is how Tesla, Waymo, and other vision systems preprocess camera feeds.
    """
    
    # ImageNet normalization statistics (standard across industry)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            target_size: Output image dimensions (height, width)
        """
        self.target_size = target_size
    
    def normalize(self, images: np.ndarray) -> np.ndarray:
        """
        Normalize images using ImageNet statistics.
        Uses broadcasting to apply normalization across entire batch.
        
        Args:
            images: Array of shape (batch, height, width, channels) with values [0, 255]
        
        Returns:
            Normalized images with values roughly in [-2.5, 2.5]
        """
        # Convert to float and scale to [0, 1]
        normalized = images.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization using broadcasting
        # Shape (3,) broadcasts to (batch, height, width, 3)
        normalized = (normalized - self.IMAGENET_MEAN) / self.IMAGENET_STD
        
        return normalized
    
    def prepare_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for neural network inference.
        
        Args:
            images: Raw images (batch, height, width, channels)
        
        Returns:
            Preprocessed images ready for model input (batch, channels, height, width)
        """
        # Normalize using ImageNet statistics
        processed = self.normalize(images)
        
        # Transpose to channels-first format (PyTorch convention)
        # (batch, height, width, channels) -> (batch, channels, height, width)
        processed = np.transpose(processed, (0, 3, 1, 2))
        
        return processed
    
    def extract_patches(self, image: np.ndarray, patch_size: int = 16) -> np.ndarray:
        """
        Extract non-overlapping patches from image (used in Vision Transformers).
        
        Args:
            image: Single image (height, width, channels)
            patch_size: Size of square patches
        
        Returns:
            Patches of shape (num_patches, patch_size, patch_size, channels)
        """
        h, w, c = image.shape
        
        # Reshape to extract patches using stride tricks
        patches = image.reshape(
            h // patch_size, patch_size,
            w // patch_size, patch_size,
            c
        )
        
        # Rearrange dimensions
        patches = patches.transpose(0, 2, 1, 3, 4)
        
        # Flatten patch grid
        num_patches = (h // patch_size) * (w // patch_size)
        patches = patches.reshape(num_patches, patch_size, patch_size, c)
        
        return patches


# =============================================================================
# NEURAL NETWORK WEIGHT INITIALIZER
# Proper initialization prevents vanishing/exploding gradients
# =============================================================================

class WeightInitializer:
    """
    Initialize neural network weights using industry-standard methods.
    Poor initialization was a major bottleneck in deep learning until ~2010.
    """
    
    @staticmethod
    def xavier_uniform(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
        """
        Xavier/Glorot uniform initialization.
        Maintains variance of activations across layers.
        
        Used by default in most frameworks for layers with sigmoid/tanh activations.
        
        Args:
            shape: Weight tensor shape (fan_in, fan_out) or (out, in, kh, kw)
            gain: Scaling factor for different activation functions
        
        Returns:
            Initialized weight array
        """
        if len(shape) == 2:
            fan_in, fan_out = shape
        else:
            # Convolutional layer: (out_channels, in_channels, kh, kw)
            receptive_field = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field
            fan_out = shape[0] * receptive_field
        
        # Xavier formula
        bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
        
        return np.random.uniform(-bound, bound, shape).astype(np.float32)
    
    @staticmethod
    def kaiming_normal(shape: Tuple[int, ...], mode: str = 'fan_in') -> np.ndarray:
        """
        Kaiming/He initialization for ReLU networks.
        Accounts for ReLU zeroing out half of activations.
        
        This is the default in PyTorch for Conv and Linear layers.
        
        Args:
            shape: Weight tensor shape
            mode: 'fan_in' (default) or 'fan_out'
        
        Returns:
            Initialized weight array
        """
        if len(shape) == 2:
            fan_in, fan_out = shape
        else:
            receptive_field = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field
            fan_out = shape[0] * receptive_field
        
        fan = fan_in if mode == 'fan_in' else fan_out
        
        # He formula (accounts for ReLU)
        std = np.sqrt(2.0 / fan)
        
        return np.random.normal(0, std, shape).astype(np.float32)
    
    @staticmethod
    def initialize_network(layer_sizes: list) -> list:
        """
        Initialize all weights for a fully-connected network.
        
        Args:
            layer_sizes: List of layer dimensions [input, hidden1, hidden2, ..., output]
        
        Returns:
            List of weight matrices
        """
        weights = []
        
        for i in range(len(layer_sizes) - 1):
            shape = (layer_sizes[i], layer_sizes[i + 1])
            
            # Use Kaiming for hidden layers (ReLU), Xavier for output
            if i < len(layer_sizes) - 2:
                w = WeightInitializer.kaiming_normal(shape)
            else:
                w = WeightInitializer.xavier_uniform(shape)
            
            weights.append(w)
        
        return weights


# =============================================================================
# VECTORIZATION PERFORMANCE BENCHMARK
# Demonstrates 50-200x speedup over Python loops
# =============================================================================

class PerformanceBenchmark:
    """
    Compare vectorized vs loop-based implementations.
    Shows why vectorization is non-negotiable for production AI.
    """
    
    @staticmethod
    def normalize_loop(data: np.ndarray) -> np.ndarray:
        """Normalize using Python loops (slow)."""
        result = np.zeros_like(data, dtype=np.float32)
        mean = np.mean(data)
        std = np.std(data)
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                result[i, j] = (data[i, j] - mean) / std
        
        return result
    
    @staticmethod
    def normalize_vectorized(data: np.ndarray) -> np.ndarray:
        """Normalize using vectorization (fast)."""
        return (data - np.mean(data)) / np.std(data)
    
    @staticmethod
    def matrix_multiply_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication using loops (slow)."""
        m, k = a.shape
        k2, n = b.shape
        
        result = np.zeros((m, n), dtype=np.float32)
        
        for i in range(m):
            for j in range(n):
                for p in range(k):
                    result[i, j] += a[i, p] * b[p, j]
        
        return result
    
    @staticmethod
    def matrix_multiply_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication using NumPy (fast)."""
        return np.dot(a, b)
    
    @staticmethod
    def run_benchmark(size: int = 500) -> dict:
        """
        Run performance comparison.
        
        Args:
            size: Matrix dimension for benchmarks
        
        Returns:
            Dictionary with timing results
        """
        results = {}
        
        # Generate test data
        data = np.random.randn(size, size).astype(np.float32)
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        # Benchmark normalization
        start = time.time()
        PerformanceBenchmark.normalize_loop(data)
        loop_time = time.time() - start
        
        start = time.time()
        PerformanceBenchmark.normalize_vectorized(data)
        vec_time = time.time() - start
        
        results['normalize'] = {
            'loop_ms': loop_time * 1000,
            'vectorized_ms': vec_time * 1000,
            'speedup': loop_time / vec_time
        }
        
        # Benchmark matrix multiplication (smaller size due to O(n³))
        small_size = min(100, size)
        a_small = a[:small_size, :small_size]
        b_small = b[:small_size, :small_size]
        
        start = time.time()
        PerformanceBenchmark.matrix_multiply_loop(a_small, b_small)
        loop_time = time.time() - start
        
        start = time.time()
        PerformanceBenchmark.matrix_multiply_vectorized(a_small, b_small)
        vec_time = time.time() - start
        
        results['matmul'] = {
            'loop_ms': loop_time * 1000,
            'vectorized_ms': vec_time * 1000,
            'speedup': loop_time / vec_time
        }
        
        return results


# =============================================================================
# ADVANCED INDEXING OPERATIONS
# Used in attention mechanisms, top-k selection, and data filtering
# =============================================================================

class AdvancedIndexing:
    """
    Demonstrate advanced indexing patterns used in production AI.
    """
    
    @staticmethod
    def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
        """
        Get indices of top-k scores (used in beam search, sampling).
        
        Args:
            scores: 1D array of scores
            k: Number of top elements
        
        Returns:
            Indices of top-k elements (highest first)
        """
        # argsort returns ascending, so take last k and reverse
        return np.argsort(scores)[-k:][::-1]
    
    @staticmethod
    def filter_by_threshold(
        values: np.ndarray, 
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter values above threshold (used in object detection NMS).
        
        Returns:
            Tuple of (filtered_values, original_indices)
        """
        mask = values > threshold
        return values[mask], np.where(mask)[0]
    
    @staticmethod
    def gather(data: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Gather elements from data using indices (like torch.gather).
        Common in attention mechanisms and embedding lookups.
        
        Args:
            data: Source array
            indices: Indices to gather
        
        Returns:
            Gathered elements
        """
        return data[indices]
    
    @staticmethod
    def scatter_add(
        target: np.ndarray, 
        indices: np.ndarray, 
        values: np.ndarray
    ) -> np.ndarray:
        """
        Add values at specified indices (gradient accumulation pattern).
        
        Args:
            target: Target array to modify
            indices: Where to add values
            values: Values to add
        
        Returns:
            Modified array
        """
        result = target.copy()
        np.add.at(result, indices, values)
        return result


# =============================================================================
# DEMONSTRATION
# =============================================================================

def main():
    print("=" * 60)
    print("Day 32: NumPy Array Manipulation and Vectorization")
    print("=" * 60)
    
    # Demo 1: Image Batch Processing
    print("\n1. IMAGE BATCH PROCESSING")
    print("-" * 40)
    
    processor = ImageBatchProcessor()
    
    # Simulate batch of 32 images (common batch size)
    batch_size = 32
    images = np.random.randint(0, 255, (batch_size, 224, 224, 3), dtype=np.uint8)
    
    print(f"Input shape: {images.shape}")
    print(f"Input dtype: {images.dtype}")
    print(f"Input range: [{images.min()}, {images.max()}]")
    
    # Process batch
    processed = processor.prepare_batch(images)
    
    print(f"\nOutput shape: {processed.shape}")
    print(f"Output dtype: {processed.dtype}")
    print(f"Output range: [{processed.min():.2f}, {processed.max():.2f}]")
    
    # Extract patches (Vision Transformer style)
    patches = processor.extract_patches(images[0])
    print(f"\nPatches from one image: {patches.shape}")
    print(f"Number of 16x16 patches: {patches.shape[0]}")
    
    # Demo 2: Weight Initialization
    print("\n\n2. NEURAL NETWORK WEIGHT INITIALIZATION")
    print("-" * 40)
    
    # Initialize a simple network: 784 -> 256 -> 128 -> 10
    layer_sizes = [784, 256, 128, 10]
    weights = WeightInitializer.initialize_network(layer_sizes)
    
    print("Network architecture:", " -> ".join(map(str, layer_sizes)))
    print("\nInitialized weights:")
    
    for i, w in enumerate(weights):
        print(f"  Layer {i+1}: {w.shape}, "
              f"mean={w.mean():.4f}, std={w.std():.4f}")
    
    # Demo 3: Performance Benchmark
    print("\n\n3. VECTORIZATION PERFORMANCE")
    print("-" * 40)
    
    results = PerformanceBenchmark.run_benchmark(size=500)
    
    print("\nNormalization (500x500 array):")
    print(f"  Loop:       {results['normalize']['loop_ms']:.2f} ms")
    print(f"  Vectorized: {results['normalize']['vectorized_ms']:.2f} ms")
    print(f"  Speedup:    {results['normalize']['speedup']:.1f}x")
    
    print("\nMatrix multiplication (100x100):")
    print(f"  Loop:       {results['matmul']['loop_ms']:.2f} ms")
    print(f"  Vectorized: {results['matmul']['vectorized_ms']:.2f} ms")
    print(f"  Speedup:    {results['matmul']['speedup']:.1f}x")
    
    # Demo 4: Advanced Indexing
    print("\n\n4. ADVANCED INDEXING OPERATIONS")
    print("-" * 40)
    
    # Top-k selection (like in language model sampling)
    scores = np.random.rand(1000)
    top_5 = AdvancedIndexing.top_k_indices(scores, k=5)
    
    print("\nTop-5 selection from 1000 scores:")
    print(f"  Indices: {top_5}")
    print(f"  Values:  {scores[top_5]}")
    
    # Threshold filtering (like in object detection)
    detections = np.random.rand(100)
    filtered, indices = AdvancedIndexing.filter_by_threshold(detections, 0.9)
    
    print(f"\nFiltering detections > 0.9:")
    print(f"  Found {len(filtered)} of 100 detections")
    
    # Scatter-add (gradient accumulation)
    grad_target = np.zeros(10)
    grad_indices = np.array([1, 3, 3, 5, 5, 5])
    grad_values = np.ones(6)
    
    accumulated = AdvancedIndexing.scatter_add(grad_target, grad_indices, grad_values)
    print(f"\nScatter-add result: {accumulated}")
    
    print("\n" + "=" * 60)
    print("Key Insight: Vectorization isn't premature optimization—")
    print("it's the only way to make AI systems actually work.")
    print("=" * 60)


if __name__ == "__main__":
    main()
