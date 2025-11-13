"""
Day 31: Introduction to NumPy
A production-style data preprocessing pipeline demonstrating NumPy's power
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from memory_profiler import profile

class ImagePreprocessor:
    """
    A production-style image preprocessing pipeline similar to what's used
    at AI companies like Tesla (computer vision) and Meta (image recognition)
    """
    
    def __init__(self, seed=42):
        """Initialize with reproducible random seed"""
        np.random.seed(seed)
        self.preprocessing_stats = {}
    
    def generate_synthetic_images(self, num_images=10000, height=224, width=224):
        """
        Generate synthetic images to simulate real image data
        Shape: (num_images, height, width, 3) for RGB
        
        Real-world context: This mimics loading a batch of images from disk
        In production, you'd use cv2.imread() or PIL, but the array shape is identical
        """
        print(f"Generating {num_images} synthetic images ({height}x{width})...")
        
        # Generate random RGB values (0-255)
        # This creates a 4D array: (images, height, width, color_channels)
        images = np.random.randint(0, 256, size=(num_images, height, width, 3), dtype=np.uint8)
        
        print(f"Array shape: {images.shape}")
        print(f"Memory usage: {images.nbytes / 1024 / 1024:.2f} MB")
        
        return images
    
    def normalize_vectorized(self, images):
        """
        Normalize pixel values from [0, 255] to [0, 1]
        
        Real-world context: Every neural network expects normalized inputs
        This is done at Google, OpenAI, Meta - everywhere in AI
        """
        start_time = time.time()
        
        # Vectorized operation - processes all pixels simultaneously
        # This is 100x faster than looping through individual pixels
        normalized = images.astype(np.float32) / 255.0
        
        elapsed = time.time() - start_time
        self.preprocessing_stats['normalize_time'] = elapsed
        
        print(f"Normalized {images.shape[0]} images in {elapsed:.4f} seconds")
        print(f"Value range: [{normalized.min():.2f}, {normalized.max():.2f}]")
        
        return normalized
    
    def normalize_python_list(self, images):
        """
        Normalize using Python lists (the slow way)
        This demonstrates why AI companies use NumPy
        """
        start_time = time.time()
        
        # Convert to Python list and normalize element by element
        # This is how beginners often try to process data
        height, width, channels = images.shape[1:]
        normalized = []
        
        for img in images[:100]:  # Only process 100 to avoid waiting forever
            img_normalized = []
            for i in range(height):
                row = []
                for j in range(width):
                    pixel = []
                    for c in range(channels):
                        pixel.append(img[i, j, c] / 255.0)
                    row.append(pixel)
                img_normalized.append(row)
            normalized.append(img_normalized)
        
        elapsed = time.time() - start_time
        self.preprocessing_stats['normalize_python_time'] = elapsed
        
        print(f"Normalized 100 images with Python lists in {elapsed:.4f} seconds")
        
        return np.array(normalized)
    
    def apply_standardization(self, images):
        """
        Standardize images: zero mean, unit variance
        
        Real-world context: This is the standard preprocessing at Tesla for Autopilot
        and at most computer vision systems
        """
        start_time = time.time()
        
        # Calculate mean and std across all images
        # axis=(0,1,2) computes statistics across batch, height, width
        mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
        std = np.std(images, axis=(0, 1, 2), keepdims=True)
        
        # Standardize: (x - mean) / std
        standardized = (images - mean) / (std + 1e-7)  # Add epsilon to avoid division by zero
        
        elapsed = time.time() - start_time
        self.preprocessing_stats['standardize_time'] = elapsed
        
        print(f"Standardized images in {elapsed:.4f} seconds")
        print(f"New mean: {np.mean(standardized):.6f} (should be ~0)")
        print(f"New std: {np.std(standardized):.6f} (should be ~1)")
        
        return standardized
    
    def extract_features_vectorized(self, images):
        """
        Extract simple features from images using vectorized operations
        
        Real-world context: Feature extraction is the first step in many CV pipelines
        More complex versions use convolutions, but the principle is the same
        """
        start_time = time.time()
        
        # Extract features using broadcasting and vectorization
        # 1. Grayscale conversion (weighted average of RGB)
        grayscale = np.dot(images[..., :3], [0.299, 0.587, 0.114])
        
        # 2. Mean intensity per image
        mean_intensity = np.mean(grayscale, axis=(1, 2))
        
        # 3. Standard deviation per image (measure of contrast)
        std_intensity = np.std(grayscale, axis=(1, 2))
        
        # 4. Edge detection approximation (gradient magnitude)
        # Horizontal and vertical gradients using array slicing
        grad_x = np.abs(grayscale[:, :, 1:] - grayscale[:, :, :-1])
        grad_y = np.abs(grayscale[:, 1:, :] - grayscale[:, :-1, :])
        edge_strength = np.mean(grad_x, axis=(1, 2)) + np.mean(grad_y, axis=(1, 2))
        
        # Stack features into a feature matrix: (num_images, num_features)
        features = np.column_stack([mean_intensity, std_intensity, edge_strength])
        
        elapsed = time.time() - start_time
        self.preprocessing_stats['feature_extraction_time'] = elapsed
        
        print(f"Extracted features from {images.shape[0]} images in {elapsed:.4f} seconds")
        print(f"Feature matrix shape: {features.shape}")
        
        return features
    
    def batch_process(self, images, batch_size=256):
        """
        Process images in batches (mini-batch processing)
        
        Real-world context: This is how TensorFlow's tf.data.Dataset works
        and how PyTorch's DataLoader works - processing data in batches
        """
        num_images = images.shape[0]
        num_batches = (num_images + batch_size - 1) // batch_size
        
        print(f"\nProcessing {num_images} images in {num_batches} batches of {batch_size}...")
        
        processed_batches = []
        start_time = time.time()
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_images)
            
            batch = images[start_idx:end_idx]
            
            # Normalize batch
            normalized_batch = batch.astype(np.float32) / 255.0
            
            # Apply some augmentation (random flip)
            if np.random.rand() > 0.5:
                normalized_batch = np.flip(normalized_batch, axis=2)
            
            processed_batches.append(normalized_batch)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed batch {i+1}/{num_batches}")
        
        elapsed = time.time() - start_time
        print(f"Batch processing completed in {elapsed:.4f} seconds")
        
        # Concatenate all batches back together
        return np.concatenate(processed_batches, axis=0)
    
    def demonstrate_broadcasting(self):
        """
        Demonstrate NumPy broadcasting - a key concept for AI
        
        Real-world context: Broadcasting is used everywhere in neural networks
        for operations like adding biases, multiplying by learning rates, etc.
        """
        print("\n" + "="*60)
        print("BROADCASTING DEMONSTRATION")
        print("="*60)
        
        # Example 1: Add bias to predictions
        # Simulating adding bias in a neural network layer
        predictions = np.random.randn(4, 3)  # 4 samples, 3 classes
        bias = np.array([0.1, -0.2, 0.15])   # Bias for each class
        
        print("\n1. Adding bias to predictions (common in neural networks):")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Bias shape: {bias.shape}")
        
        # Broadcasting automatically expands bias to match predictions
        predictions_with_bias = predictions + bias
        print(f"Result shape: {predictions_with_bias.shape}")
        print("✓ Bias was broadcast across all samples automatically!")
        
        # Example 2: Normalize each feature independently
        # Simulating feature-wise normalization (like BatchNorm)
        data = np.random.randn(100, 5)  # 100 samples, 5 features
        mean = np.mean(data, axis=0)    # Mean of each feature
        std = np.std(data, axis=0)      # Std of each feature
        
        print("\n2. Feature-wise normalization (like BatchNorm):")
        print(f"Data shape: {data.shape}")
        print(f"Mean shape: {mean.shape}")
        print(f"Std shape: {std.shape}")
        
        normalized = (data - mean) / std
        print(f"Normalized shape: {normalized.shape}")
        print("✓ Mean and std were broadcast across all samples!")
        
        # Example 3: Apply different transformations to different channels
        # Simulating per-channel operations in image processing
        images = np.random.randint(0, 256, size=(10, 64, 64, 3))
        channel_scales = np.array([1.0, 0.9, 1.1])  # Different scale per channel
        
        print("\n3. Per-channel scaling (common in image preprocessing):")
        print(f"Images shape: {images.shape}")
        print(f"Channel scales shape: {channel_scales.shape}")
        
        scaled_images = images * channel_scales
        print(f"Scaled images shape: {scaled_images.shape}")
        print("✓ Scales were broadcast across batch, height, and width!")
    
    def save_preprocessed_data(self, data, filename="preprocessed_data.npy"):
        """
        Save preprocessed data in NumPy's efficient binary format
        
        Real-world context: This is how ML engineers save processed datasets
        .npy files load 10-100x faster than CSV or JSON
        """
        print(f"\nSaving preprocessed data to {filename}...")
        np.save(filename, data)
        
        file_size = np.load(filename, mmap_mode='r').nbytes / 1024 / 1024
        print(f"File size: {file_size:.2f} MB")
        print(f"Can be loaded with: data = np.load('{filename}')")
        
        return filename
    
    def print_performance_report(self):
        """Print a performance report of all operations"""
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        for operation, time_taken in self.preprocessing_stats.items():
            print(f"{operation:30s}: {time_taken:.4f} seconds")
        
        if 'normalize_time' in self.preprocessing_stats and 'normalize_python_time' in self.preprocessing_stats:
            speedup = self.preprocessing_stats['normalize_python_time'] / self.preprocessing_stats['normalize_time']
            print(f"\nNumPy speedup over Python lists: {speedup:.1f}x faster!")


def demonstrate_numpy_basics():
    """Demonstrate fundamental NumPy concepts"""
    print("\n" + "="*60)
    print("NUMPY FUNDAMENTALS FOR AI")
    print("="*60)
    
    # 1. Array creation and properties
    print("\n1. Creating Arrays (the foundation of all AI data)")
    
    # Image data: 3D array (height, width, channels)
    image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    print(f"Image shape: {image.shape} (height, width, channels)")
    print(f"Data type: {image.dtype}")
    print(f"Total elements: {image.size}")
    print(f"Memory: {image.nbytes / 1024:.2f} KB")
    
    # Text embeddings: 2D array (sequence_length, embedding_dim)
    embeddings = np.random.randn(50, 768)  # 50 tokens, 768-dim embeddings (like BERT)
    print(f"\nText embeddings shape: {embeddings.shape}")
    print(f"This is how language models represent text!")
    
    # 2. Vectorized operations
    print("\n2. Vectorized Operations (why NumPy is fast)")
    
    # Create a large array
    large_array = np.random.randn(1000000)
    
    # Time vectorized operation
    start = time.time()
    result = large_array ** 2 + 2 * large_array + 1
    vectorized_time = time.time() - start
    print(f"Vectorized operation on 1M elements: {vectorized_time:.6f} seconds")
    
    # 3. Array indexing and slicing
    print("\n3. Array Slicing (accessing data efficiently)")
    
    data = np.arange(100).reshape(10, 10)
    print(f"Original shape: {data.shape}")
    print(f"First row: {data[0]}")
    print(f"First column: {data[:, 0]}")
    print(f"Center 4x4 block: {data[3:7, 3:7].shape}")
    
    # Boolean indexing (crucial for filtering data)
    mask = data > 50
    filtered = data[mask]
    print(f"Elements > 50: {len(filtered)} elements")
    
    # 4. Array reshaping
    print("\n4. Reshaping Arrays (preparing data for models)")
    
    # Flatten image for classical ML
    flat_image = image.reshape(-1)  # -1 means "figure out this dimension"
    print(f"Flattened image shape: {flat_image.shape}")
    
    # Batch of images for neural network
    batch = np.random.randn(32, 224, 224, 3)  # 32 images
    print(f"Batch shape: {batch.shape} (batch, height, width, channels)")
    
    # 5. Statistical operations
    print("\n5. Statistical Operations (analyzing data)")
    
    data = np.random.randn(1000, 10)  # 1000 samples, 10 features
    print(f"Mean per feature: {np.mean(data, axis=0)[:3]}... (showing first 3)")
    print(f"Std per feature: {np.std(data, axis=0)[:3]}... (showing first 3)")
    print(f"Min value: {np.min(data):.4f}")
    print(f"Max value: {np.max(data):.4f}")


def main():
    """Main execution function"""
    print("="*60)
    print("DAY 31: INTRODUCTION TO NUMPY")
    print("Production-Style AI Data Preprocessing")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(seed=42)
    
    # Demonstrate NumPy fundamentals
    demonstrate_numpy_basics()
    
    # Generate synthetic images (simulating a real dataset)
    print("\n" + "="*60)
    print("BUILDING A DATA PREPROCESSING PIPELINE")
    print("="*60)
    images = preprocessor.generate_synthetic_images(num_images=10000)
    
    # Compare NumPy vs Python performance
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: NumPy vs Pure Python")
    print("="*60)
    
    print("\nNormalizing with NumPy (all 10,000 images):")
    normalized_numpy = preprocessor.normalize_vectorized(images)
    
    print("\nNormalizing with Python lists (only 100 images):")
    normalized_python = preprocessor.normalize_python_list(images)
    
    # Apply standardization
    print("\n" + "="*60)
    print("STANDARDIZATION (Zero Mean, Unit Variance)")
    print("="*60)
    standardized = preprocessor.apply_standardization(normalized_numpy)
    
    # Extract features
    print("\n" + "="*60)
    print("FEATURE EXTRACTION")
    print("="*60)
    features = preprocessor.extract_features_vectorized(images)
    
    # Demonstrate batch processing
    print("\n" + "="*60)
    print("BATCH PROCESSING (Mini-Batch Training)")
    print("="*60)
    batch_processed = preprocessor.batch_process(images[:1000], batch_size=256)
    
    # Demonstrate broadcasting
    preprocessor.demonstrate_broadcasting()
    
    # Save preprocessed data
    print("\n" + "="*60)
    print("SAVING PREPROCESSED DATA")
    print("="*60)
    preprocessor.save_preprocessed_data(features, "features.npy")
    
    # Print performance report
    preprocessor.print_performance_report()
    
    # Final summary
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("✓ NumPy arrays are 10-100x faster than Python lists")
    print("✓ Vectorization processes all data simultaneously")
    print("✓ Broadcasting simplifies operations on different-shaped arrays")
    print("✓ This pipeline mimics real production AI preprocessing")
    print("✓ Every major AI framework (TensorFlow, PyTorch) builds on NumPy")
    print("\nYou've learned the foundation of all AI/ML computation!")


if __name__ == "__main__":
    main()
