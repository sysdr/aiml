"""
Day 92: PCA for Dimensionality Reduction
Production-ready implementation with comprehensive examples
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
import time
import joblib
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ProductionPCA:
    """
    Production-ready PCA pipeline with preprocessing, evaluation, and persistence.
    Follows patterns used at Netflix, Spotify, Google for dimensionality reduction.
    """
    
    def __init__(self, variance_threshold: float = 0.95, random_state: int = 42):
        """
        Initialize PCA pipeline with variance preservation threshold.
        
        Args:
            variance_threshold: Target cumulative variance to preserve (0-1)
            random_state: Random seed for reproducibility
        """
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.n_components_optimal = None
        self.fit_time = None
        self.transform_time = None
        
    def fit(self, X: np.ndarray) -> 'ProductionPCA':
        """
        Fit PCA transformation on training data.
        
        Pattern: Fit on training data only to prevent data leakage.
        Used in production to learn transformation offline.
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            self for method chaining
        """
        start_time = time.time()
        
        # Step 1: Standardize features (critical for PCA)
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 2: Fit PCA with all components to analyze variance
        pca_full = PCA(random_state=self.random_state)
        pca_full.fit(X_scaled)
        
        # Step 3: Determine optimal components based on variance threshold
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        self.n_components_optimal = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        
        # Step 4: Fit final PCA with optimal components
        self.pca = PCA(n_components=self.n_components_optimal, random_state=self.random_state)
        self.pca.fit(X_scaled)
        
        self.fit_time = time.time() - start_time
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to reduced dimensionality.
        
        Production pattern: Apply learned transformation to new data.
        Used in real-time inference for recommendations, search, etc.
        
        Args:
            X: Data to transform (n_samples, n_features)
            
        Returns:
            Transformed data (n_samples, n_components)
        """
        if self.pca is None:
            raise ValueError("Must fit PCA before transforming. Call fit() first.")
            
        start_time = time.time()
        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)
        self.transform_time = time.time() - start_time
        
        return X_reduced
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step (training data only)."""
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Reconstruct original space from reduced representation.
        
        Production use: Anomaly detection via reconstruction error.
        High error indicates data doesn't fit learned variance patterns.
        
        Args:
            X_reduced: Reduced data (n_samples, n_components)
            
        Returns:
            Reconstructed data (n_samples, n_features)
        """
        if self.pca is None:
            raise ValueError("Must fit PCA before inverse transforming.")
            
        X_scaled_reconstructed = self.pca.inverse_transform(X_reduced)
        X_reconstructed = self.scaler.inverse_transform(X_scaled_reconstructed)
        return X_reconstructed
        
    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """
        Calculate mean squared reconstruction error.
        
        Pattern used for: Anomaly detection, quality validation
        Low error = data fits learned patterns
        High error = anomalous/outlier data
        
        Args:
            X: Original data
            
        Returns:
            Mean squared error of reconstruction
        """
        X_reduced = self.transform(X)
        X_reconstructed = self.inverse_transform(X_reduced)
        mse = np.mean((X - X_reconstructed) ** 2)
        return mse
        
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get variance explained by each component."""
        if self.pca is None:
            raise ValueError("Must fit PCA first.")
        return self.pca.explained_variance_ratio_
        
    def get_cumulative_variance(self) -> np.ndarray:
        """Get cumulative variance explained."""
        return np.cumsum(self.get_explained_variance_ratio())
        
    def get_component_loadings(self, feature_names: Optional[List[str]] = None) -> Dict:
        """
        Get component loadings (eigenvectors) for interpretation.
        
        Production use: Understand what each component represents.
        Example: At Netflix, PC1 might represent "binge-watching tendency"
        
        Returns:
            Dictionary mapping component index to top contributing features
        """
        if self.pca is None:
            raise ValueError("Must fit PCA first.")
            
        loadings = {}
        components = self.pca.components_
        
        for i, component in enumerate(components):
            # Get indices of top contributing features
            top_indices = np.argsort(np.abs(component))[-5:][::-1]
            
            if feature_names:
                top_features = [(feature_names[idx], component[idx]) for idx in top_indices]
            else:
                top_features = [(f"Feature_{idx}", component[idx]) for idx in top_indices]
                
            loadings[f"PC{i+1}"] = top_features
            
        return loadings
        
    def save(self, filepath: str):
        """Save fitted PCA pipeline for production deployment."""
        if self.pca is None:
            raise ValueError("Must fit PCA before saving.")
        joblib.dump({
            'scaler': self.scaler,
            'pca': self.pca,
            'n_components': self.n_components_optimal,
            'variance_threshold': self.variance_threshold
        }, filepath)
        
    @classmethod
    def load(cls, filepath: str) -> 'ProductionPCA':
        """Load fitted PCA pipeline from disk."""
        data = joblib.load(filepath)
        instance = cls(variance_threshold=data['variance_threshold'])
        instance.scaler = data['scaler']
        instance.pca = data['pca']
        instance.n_components_optimal = data['n_components']
        return instance


def demonstrate_basic_pca():
    """
    Demonstrate basic PCA workflow on synthetic high-dimensional data.
    Pattern: What happens at tech companies processing user behavior data.
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 1: Basic PCA on High-Dimensional Synthetic Data")
    print("="*80)
    
    # Generate synthetic data: 1000 samples, 100 features
    # Simulates user behavior data (100 tracked features per user)
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=20,  # Only 20 features truly matter
        n_redundant=30,    # 30 are linear combinations
        n_repeated=10,     # 10 are duplicates
        random_state=42
    )
    
    print(f"\nOriginal data shape: {X.shape}")
    print(f"Contains: {X.shape[1]} features (simulating user behavior tracking)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit PCA pipeline
    pca_pipeline = ProductionPCA(variance_threshold=0.95)
    pca_pipeline.fit(X_train)
    
    print(f"\nPCA Analysis:")
    print(f"  Optimal components: {pca_pipeline.n_components_optimal}")
    print(f"  Dimensionality reduction: {X.shape[1]} → {pca_pipeline.n_components_optimal}")
    print(f"  Compression ratio: {X.shape[1] / pca_pipeline.n_components_optimal:.2f}x")
    print(f"  Variance preserved: {pca_pipeline.get_cumulative_variance()[-1]:.4f}")
    print(f"  Fit time: {pca_pipeline.fit_time:.4f}s")
    
    # Transform data
    X_train_reduced = pca_pipeline.transform(X_train)
    X_test_reduced = pca_pipeline.transform(X_test)
    
    print(f"\nTransformed shapes:")
    print(f"  Training: {X_train.shape} → {X_train_reduced.shape}")
    print(f"  Test: {X_test.shape} → {X_test_reduced.shape}")
    print(f"  Transform time: {pca_pipeline.transform_time:.4f}s")
    
    # Reconstruction error
    train_error = pca_pipeline.get_reconstruction_error(X_train)
    test_error = pca_pipeline.get_reconstruction_error(X_test)
    
    print(f"\nReconstruction Error (MSE):")
    print(f"  Training: {train_error:.6f}")
    print(f"  Test: {test_error:.6f}")
    print(f"  Error increase: {((test_error - train_error) / train_error * 100):.2f}%")
    
    return pca_pipeline, X_train_reduced, X_test_reduced


def demonstrate_mnist_compression():
    """
    Demonstrate PCA on MNIST digits (784 dimensions → compressed).
    Real-world pattern: Image compression for computer vision pipelines.
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 2: PCA on MNIST Digits (Image Compression)")
    print("="*80)
    
    # Load MNIST digits
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"\nMNIST Dataset:")
    print(f"  Images: {X.shape[0]}")
    print(f"  Pixels per image: {X.shape[1]} (8x8 grayscale)")
    print(f"  Classes: {len(np.unique(y))} (digits 0-9)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test different variance thresholds
    thresholds = [0.80, 0.90, 0.95, 0.99]
    results = []
    
    for threshold in thresholds:
        pca = ProductionPCA(variance_threshold=threshold)
        pca.fit(X_train)
        
        compression_ratio = X.shape[1] / pca.n_components_optimal
        reconstruction_error = pca.get_reconstruction_error(X_test)
        
        results.append({
            'threshold': threshold,
            'components': pca.n_components_optimal,
            'compression': compression_ratio,
            'error': reconstruction_error,
            'actual_variance': pca.get_cumulative_variance()[-1]
        })
        
    print(f"\nCompression Analysis (varying variance thresholds):")
    print(f"{'Threshold':<12} {'Components':<12} {'Compression':<15} {'Variance':<12} {'Recon Error':<15}")
    print("-" * 76)
    
    for r in results:
        print(f"{r['threshold']:<12.2f} {r['components']:<12} "
              f"{r['compression']:<15.2f}x {r['actual_variance']:<12.4f} "
              f"{r['error']:<15.6f}")
    
    # Demonstrate 95% threshold (balanced compression)
    print(f"\n{'Production Choice: 95% Variance Threshold'}")
    print(f"  Reduces 64 pixels → {results[2]['components']} components")
    print(f"  {results[2]['compression']:.1f}x compression with minimal information loss")
    print(f"  Used by: Google Photos, Facebook, Tesla vision systems")
    
    return results


def demonstrate_incremental_pca():
    """
    Demonstrate IncrementalPCA for large datasets that don't fit in memory.
    Production pattern: Processing billions of samples at Google, Meta, Netflix.
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 3: Incremental PCA (Large-Scale Processing)")
    print("="*80)
    
    # Simulate large dataset (10,000 samples, 500 features)
    # In production: millions of samples, thousands of features
    n_samples = 10000
    n_features = 500
    n_components = 50
    batch_size = 1000
    
    print(f"\nSimulating Large Dataset:")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Features: {n_features}")
    print(f"  Target components: {n_components}")
    print(f"  Processing batch size: {batch_size}")
    
    # Generate data in batches (simulating streaming data)
    scaler = StandardScaler()
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    start_time = time.time()
    
    # Process in batches
    for i in range(0, n_samples, batch_size):
        # Generate batch
        X_batch, _ = make_classification(
            n_samples=batch_size,
            n_features=n_features,
            n_informative=100,
            random_state=42 + i
        )
        
        # Scale and fit incrementally
        X_batch_scaled = scaler.fit_transform(X_batch) if i == 0 else scaler.transform(X_batch)
        ipca.partial_fit(X_batch_scaled)
        
        if (i + batch_size) % (batch_size * 3) == 0:
            print(f"  Processed {i + batch_size:,} samples...")
    
    total_time = time.time() - start_time
    
    print(f"\nIncremental PCA Results:")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Throughput: {n_samples / total_time:.0f} samples/second")
    print(f"  Explained variance: {np.sum(ipca.explained_variance_ratio_):.4f}")
    print(f"  Memory efficient: Processes unlimited data size")
    
    print(f"\nProduction Use Cases:")
    print(f"  - Netflix: Daily batch processing of viewing history")
    print(f"  - Spotify: Weekly user behavior compression")
    print(f"  - Google: Continuous image embedding reduction")
    
    return ipca


def visualize_pca_results(pca_pipeline: ProductionPCA, X: np.ndarray, y: np.ndarray):
    """
    Create comprehensive PCA visualizations.
    Production pattern: Used in dashboards at every major AI company.
    """
    print("\n" + "="*80)
    print("VISUALIZATION: PCA Analysis Plots")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Scree plot (variance per component)
    explained_var = pca_pipeline.get_explained_variance_ratio()
    axes[0, 0].bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Variance Explained')
    axes[0, 0].set_title('Scree Plot: Variance per Component')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Cumulative variance
    cumulative_var = pca_pipeline.get_cumulative_variance()
    axes[0, 1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                    marker='o', linewidth=2, markersize=4)
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[0, 1].axvline(x=pca_pipeline.n_components_optimal, color='g', 
                       linestyle='--', label=f'Optimal: {pca_pipeline.n_components_optimal}')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Variance Explained')
    axes[0, 1].set_title('Cumulative Variance Explained')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. 2D projection (first two components)
    X_reduced = pca_pipeline.transform(X)
    scatter = axes[1, 0].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                 c=y, cmap='viridis', alpha=0.6, s=20)
    axes[1, 0].set_xlabel(f'PC1 ({explained_var[0]:.1%} var)')
    axes[1, 0].set_ylabel(f'PC2 ({explained_var[1]:.1%} var)')
    axes[1, 0].set_title('2D Projection (First Two Components)')
    plt.colorbar(scatter, ax=axes[1, 0], label='Class')
    
    # 4. Reconstruction error distribution
    reconstruction_errors = []
    for sample in X[:200]:  # Sample for speed
        sample_reduced = pca_pipeline.transform(sample.reshape(1, -1))
        sample_reconstructed = pca_pipeline.inverse_transform(sample_reduced)
        error = np.mean((sample - sample_reconstructed.flatten()) ** 2)
        reconstruction_errors.append(error)
    
    axes[1, 1].hist(reconstruction_errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Reconstruction Error (MSE)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reconstruction Error Distribution')
    axes[1, 1].axvline(x=np.mean(reconstruction_errors), color='r', 
                       linestyle='--', label=f'Mean: {np.mean(reconstruction_errors):.6f}')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: pca_analysis.png")
    print("  Contains: Scree plot, cumulative variance, 2D projection, reconstruction errors")
    
    plt.close()


def run_pca_dimensionality_reduction(n_samples: int = 1000, n_features: int = 100) -> Dict:
    """
    Main execution function with comprehensive metrics.
    
    Returns:
        Dictionary of performance metrics for testing/validation
    """
    # Generate sample data
    # Ensure n_informative doesn't exceed n_features
    n_informative = min(max(2, n_features // 5), n_features - 1)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit PCA
    pca_pipeline = ProductionPCA(variance_threshold=0.95)
    pca_pipeline.fit(X_train)
    
    # Transform
    X_train_reduced = pca_pipeline.transform(X_train)
    X_test_reduced = pca_pipeline.transform(X_test)
    
    # Calculate metrics
    metrics = {
        'original_dims': X.shape[1],
        'reduced_dims': pca_pipeline.n_components_optimal,
        'compression_ratio': X.shape[1] / pca_pipeline.n_components_optimal,
        'variance_preserved': float(pca_pipeline.get_cumulative_variance()[-1]),
        'reconstruction_error_train': pca_pipeline.get_reconstruction_error(X_train),
        'reconstruction_error_test': pca_pipeline.get_reconstruction_error(X_test),
        'fit_time': pca_pipeline.fit_time,
        'transform_time': pca_pipeline.transform_time
    }
    
    return metrics


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DAY 92: PCA FOR DIMENSIONALITY REDUCTION")
    print("Production Implementation with Comprehensive Examples")
    print("="*80)
    
    # Run demonstrations
    pca_pipeline, X_train_reduced, X_test_reduced = demonstrate_basic_pca()
    mnist_results = demonstrate_mnist_compression()
    ipca = demonstrate_incremental_pca()
    
    # Generate sample data for visualization
    X, y = make_classification(n_samples=500, n_features=50, n_informative=10, 
                               n_classes=3, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    viz_pca = ProductionPCA(variance_threshold=0.95)
    viz_pca.fit(X_train)
    visualize_pca_results(viz_pca, X_train, y_train)
    
    # Save model for production
    pca_pipeline.save('production_pca_model.pkl')
    print("\n" + "="*80)
    print("✓ Model saved: production_pca_model.pkl")
    print("  Ready for deployment in production systems")
    print("="*80)
    
    # Final summary
    print("\n" + "="*80)
    print("KEY TAKEAWAYS - PRODUCTION PCA PATTERNS")
    print("="*80)
    print("""
1. Standardization is Critical: Always scale features before PCA
2. Variance Threshold Selection: 95% for accuracy, 85-90% for speed
3. Fit-Transform Pattern: Fit on training only, transform all data
4. Incremental Processing: Use IncrementalPCA for large datasets
5. Reconstruction Error: Monitor for anomaly detection
6. Component Interpretation: Understand what PCs represent
7. Model Persistence: Save fitted transformers for production

Production Applications:
- Recommendation Systems: Netflix, Spotify, Amazon
- Computer Vision: Google Photos, Facebook, Tesla
- Anomaly Detection: Datadog, Google Cloud Monitoring  
- Data Visualization: Tableau, Looker, Analytics Platforms

Next Steps (Days 93-98):
- Review clustering + dimensionality reduction integration
- Hyperparameter optimization for unsupervised learning
- Prepare for reinforcement learning (Days 99+)
    """)
