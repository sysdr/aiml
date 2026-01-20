"""
Day 87: K-Means with Scikit-learn - Production Implementation

This module demonstrates production-ready K-Means clustering using scikit-learn,
including data preprocessing, model training, prediction, and persistence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import joblib
from pathlib import Path
from typing import Tuple, Optional


class CustomerSegmentation:
    """
    Production-ready customer segmentation using K-Means clustering.
    
    This class encapsulates the complete pipeline: data preprocessing,
    model training, prediction, and persistence. Used in production systems
    at companies like Spotify (user segmentation), Amazon (product clustering),
    and Uber (geographic zone creation).
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        init: str = 'k-means++',
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42
    ):
        """
        Initialize customer segmentation model.
        
        Args:
            n_clusters: Number of customer segments
            init: Initialization method ('k-means++' recommended)
            n_init: Number of initializations to try
            max_iter: Maximum iterations per initialization
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None) -> 'CustomerSegmentation':
        """
        Train the segmentation model on customer data.
        
        This method:
        1. Scales features using StandardScaler (critical for K-Means)
        2. Trains K-Means clustering model
        3. Stores model state for production use
        
        Args:
            X: Customer feature matrix (n_samples, n_features)
            feature_names: Optional feature names for interpretability
            
        Returns:
            self for method chaining
        """
        # Validate input
        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset")
        
        if len(X) < self.n_clusters:
            raise ValueError(
                f"Number of samples ({len(X)}) must be >= n_clusters ({self.n_clusters})"
            )
        
        # Scale features - critical step often missed
        # K-Means uses Euclidean distance, so features must be on same scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train clustering model
        self.kmeans.fit(X_scaled)
        
        # Store metadata
        self.is_fitted = True
        self.feature_names = feature_names
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new customers to existing segments.
        
        Production pattern: fit once (nightly), predict many times (real-time).
        This is how Spotify assigns new users to listener segments.
        
        Args:
            X: New customer features (n_samples, n_features)
            
        Returns:
            Cluster assignments (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Apply same scaling transformation as training data
        X_scaled = self.scaler.transform(X)
        
        return self.kmeans.predict(X_scaled)
    
    def fit_predict(self, X: np.ndarray, feature_names: Optional[list] = None) -> np.ndarray:
        """
        Fit model and return cluster assignments.
        
        Convenience method for training pipeline.
        
        Args:
            X: Customer feature matrix
            feature_names: Optional feature names
            
        Returns:
            Cluster assignments
        """
        self.fit(X, feature_names)
        return self.kmeans.labels_
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centroids in original feature space.
        
        Useful for interpreting segments: "Segment 0 has high purchase frequency,
        low average order value, recent last purchase."
        
        Returns:
            Cluster centers (n_clusters, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Transform centroids back to original scale
        return self.scaler.inverse_transform(self.kmeans.cluster_centers_)
    
    def get_metrics(self) -> dict:
        """
        Get clustering quality metrics.
        
        Returns:
            Dictionary containing:
            - inertia: Sum of squared distances to nearest centroid
            - n_iter: Iterations until convergence
            - n_clusters: Number of clusters
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return {
            'inertia': self.kmeans.inertia_,
            'n_iter': self.kmeans.n_iter_,
            'n_clusters': self.n_clusters
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk for production deployment.
        
        Pattern used by Netflix, Spotify, etc. Train offline, deploy online.
        
        Args:
            filepath: Path to save model (e.g., 'models/customer_segments_v1.pkl')
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save entire object (preserves scaler + kmeans)
        joblib.dump(self, filepath)
        
    @staticmethod
    def load(filepath: str) -> 'CustomerSegmentation':
        """
        Load saved model from disk.
        
        Production pattern for serving predictions at scale.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded CustomerSegmentation instance
        """
        return joblib.load(filepath)


def generate_synthetic_customers(
    n_samples: int = 1000,
    n_features: int = 4,
    n_clusters: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Generate synthetic customer data for demonstration.
    
    In production, this would be replaced with actual customer data from
    your data warehouse (e.g., BigQuery, Redshift, Snowflake).
    
    Args:
        n_samples: Number of customers
        n_features: Number of features per customer
        n_clusters: True number of underlying segments
        random_state: Random seed
        
    Returns:
        Tuple of (features, true_labels, feature_names)
    """
    # Generate clustered data
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.5,
        random_state=random_state
    )
    
    # Realistic feature names for customer segmentation
    feature_names = [
        'purchase_frequency',      # How often they buy
        'avg_order_value',         # Average purchase amount
        'days_since_last_purchase', # Recency
        'customer_lifetime_value'  # Total value
    ][:n_features]
    
    return X, y, feature_names


def visualize_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    title: str = "Customer Segments"
) -> None:
    """
    Visualize clusters in 2D using PCA for dimensionality reduction.
    
    Production insight: High-dimensional data (100+ features) is common.
    PCA projects to 2D for human visualization while preserving variance.
    This is exactly how Airbnb visualizes user segments.
    
    Args:
        X: Feature matrix
        labels: Cluster assignments
        centroids: Optional cluster centers
        title: Plot title
    """
    # Reduce to 2D for visualization if needed
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        
        if centroids is not None:
            centroids_2d = pca.transform(centroids)
        
        explained_var = pca.explained_variance_ratio_.sum() * 100
        subtitle = f"(PCA: {explained_var:.1f}% variance explained)"
    else:
        X_2d = X
        centroids_2d = centroids
        subtitle = ""
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot clusters
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.6,
        edgecolors='w',
        linewidth=0.5,
        s=50
    )
    
    # Plot centroids
    if centroids is not None:
        plt.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c='red',
            marker='X',
            s=300,
            edgecolors='black',
            linewidth=2,
            label='Centroids',
            zorder=10
        )
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel('Principal Component 1' if X.shape[1] > 2 else 'Feature 1')
    plt.ylabel('Principal Component 2' if X.shape[1] > 2 else 'Feature 2')
    plt.title(f"{title}\n{subtitle}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('customer_segments.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved to 'customer_segments.png'")


def analyze_segments(
    model: CustomerSegmentation,
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list
) -> pd.DataFrame:
    """
    Analyze segment characteristics for business insights.
    
    Production use: Generate segment profiles for marketing teams.
    "Segment 0 = High-value repeat customers, target with premium offers."
    
    Args:
        model: Fitted segmentation model
        X: Customer features
        labels: Cluster assignments
        feature_names: Feature names
        
    Returns:
        DataFrame with segment statistics
    """
    # Get cluster centers in original scale
    centers = model.get_cluster_centers()
    
    # Create segment profile
    segment_profiles = []
    
    for cluster_id in range(model.n_clusters):
        # Get customers in this segment
        mask = labels == cluster_id
        segment_size = mask.sum()
        
        # Calculate segment statistics
        profile = {
            'Segment': f'Segment {cluster_id}',
            'Size': segment_size,
            'Percentage': f'{(segment_size / len(labels) * 100):.1f}%'
        }
        
        # Add centroid features
        for i, feature in enumerate(feature_names):
            profile[feature] = centers[cluster_id, i]
        
        segment_profiles.append(profile)
    
    # Convert to DataFrame for nice display
    df = pd.DataFrame(segment_profiles)
    
    return df


def main():
    """
    Main execution: Complete K-Means clustering pipeline.
    
    Demonstrates production workflow:
    1. Generate/load data
    2. Train segmentation model
    3. Analyze segments
    4. Save model for deployment
    5. Test prediction on new data
    """
    print("=" * 70)
    print("Day 87: K-Means with Scikit-learn - Production Pipeline")
    print("=" * 70)
    print()
    
    # Step 1: Generate synthetic customer data
    print("Step 1: Generating synthetic customer data...")
    X, y_true, feature_names = generate_synthetic_customers(
        n_samples=1000,
        n_features=4,
        n_clusters=5,
        random_state=42
    )
    print(f"✅ Generated {len(X)} customers with {X.shape[1]} features")
    print(f"   Features: {', '.join(feature_names)}")
    print()
    
    # Step 2: Train customer segmentation model
    print("Step 2: Training customer segmentation model...")
    model = CustomerSegmentation(n_clusters=5, random_state=42)
    labels = model.fit_predict(X, feature_names=feature_names)
    
    metrics = model.get_metrics()
    print(f"✅ Model trained successfully")
    print(f"   Inertia: {metrics['inertia']:.2f}")
    print(f"   Iterations: {metrics['n_iter']}")
    print()
    
    # Step 3: Analyze segments
    print("Step 3: Analyzing customer segments...")
    segment_profiles = analyze_segments(model, X, labels, feature_names)
    print(segment_profiles.to_string(index=False))
    print()
    
    # Step 4: Visualize clusters
    print("Step 4: Visualizing customer segments...")
    centroids = model.get_cluster_centers()
    visualize_clusters(X, labels, centroids, "Customer Segments (K-Means)")
    print()
    
    # Step 5: Save model
    print("Step 5: Saving model for production deployment...")
    model_path = 'models/customer_segmentation_v1.pkl'
    model.save(model_path)
    print(f"✅ Model saved to '{model_path}'")
    print()
    
    # Step 6: Load and test prediction
    print("Step 6: Testing prediction on new customers...")
    loaded_model = CustomerSegmentation.load(model_path)
    
    # Generate new customers
    X_new, _, _ = generate_synthetic_customers(
        n_samples=10,
        n_features=4,
        n_clusters=5,
        random_state=123
    )
    
    predictions = loaded_model.predict(X_new)
    print(f"✅ Predicted segments for {len(X_new)} new customers:")
    for i, segment in enumerate(predictions):
        print(f"   Customer {i+1}: Segment {segment}")
    print()
    
    print("=" * 70)
    print("Pipeline complete! Check 'customer_segments.png' for visualization.")
    print("=" * 70)


if __name__ == "__main__":
    main()
