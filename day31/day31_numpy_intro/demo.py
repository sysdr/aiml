"""
Demo script for Day 31: NumPy
Runs the preprocessing pipeline and updates dashboard metrics
"""

import json
import time
import requests
from lesson_code import ImagePreprocessor, main
import sys

def run_demo_and_update_dashboard():
    """Run the NumPy demo and update dashboard metrics"""
    print("="*60)
    print("Running NumPy Demo - Updating Dashboard Metrics")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(seed=42)
    
    # Generate synthetic images
    print("\nGenerating synthetic images...")
    images = preprocessor.generate_synthetic_images(num_images=10000, height=224, width=224)
    memory_mb = images.nbytes / 1024 / 1024
    
    # Normalize with NumPy
    print("\nNormalizing with NumPy...")
    normalized_numpy = preprocessor.normalize_vectorized(images)
    
    # Normalize with Python (for comparison)
    print("\nNormalizing with Python lists (for comparison)...")
    normalized_python = preprocessor.normalize_python_list(images)
    
    # Apply standardization
    print("\nApplying standardization...")
    standardized = preprocessor.apply_standardization(normalized_numpy)
    
    # Extract features
    print("\nExtracting features...")
    features = preprocessor.extract_features_vectorized(images)
    features_count = features.shape[0] * features.shape[1] if len(features.shape) >= 2 else features.shape[0]
    
    # Batch processing
    print("\nProcessing in batches...")
    batch_start = time.time()
    batch_processed = preprocessor.batch_process(images[:1000], batch_size=256)
    batch_time = time.time() - batch_start
    preprocessor.preprocessing_stats['batch_processing_time'] = batch_time
    batches_count = (1000 + 256 - 1) // 256
    
    # Calculate speedup
    if 'normalize_time' in preprocessor.preprocessing_stats and 'normalize_python_time' in preprocessor.preprocessing_stats:
        speedup = preprocessor.preprocessing_stats['normalize_python_time'] / preprocessor.preprocessing_stats['normalize_time']
        preprocessor.preprocessing_stats['numpy_speedup'] = speedup
    
    # Print performance report
    preprocessor.print_performance_report()
    
    # Try to update dashboard (if it's running)
    try:
        # Import dashboard module to update metrics directly
        import dashboard
        dashboard.update_metrics_from_stats(
            preprocessor.preprocessing_stats,
            num_images=images.shape[0],
            memory_mb=memory_mb,
            features_count=features_count,
            batches_count=batches_count
        )
        print("\n✓ Dashboard metrics updated successfully!")
    except Exception as e:
        print(f"\n⚠ Dashboard not running or update failed: {e}")
        print("   Start dashboard with: python dashboard.py")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)
    print(f"Processed {images.shape[0]} images")
    print(f"Extracted {features_count} features")
    print(f"Memory used: {memory_mb:.2f} MB")
    if 'numpy_speedup' in preprocessor.preprocessing_stats:
        print(f"NumPy speedup: {preprocessor.preprocessing_stats['numpy_speedup']:.1f}x faster")
    print("="*60)

if __name__ == '__main__':
    run_demo_and_update_dashboard()


