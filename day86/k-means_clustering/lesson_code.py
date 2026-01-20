"""
Day 86: K-Means Clustering Theory - Core Algorithm Implementation
Demonstrates the theoretical foundations of K-Means clustering
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import seaborn as sns

class KMeansTheory:
    """
    Educational implementation of K-Means algorithm to demonstrate theory.
    This shows the algorithm mechanics step-by-step for learning purposes.
    For production use, utilize scikit-learn's optimized implementation.
    """
    
    def __init__(self, n_clusters: int = 3, max_iterations: int = 100, 
                 tolerance: float = 1e-4, random_state: Optional[int] = None):
        """
        Initialize K-Means clustering algorithm.
        
        Args:
            n_clusters: Number of clusters (K)
            max_iterations: Maximum iterations before stopping
            tolerance: Convergence threshold for centroid movement
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Algorithm state
        self.centroids = None
        self.labels = None
        self.iteration_history = []
        self.inertia_history = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _euclidean_distance(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distance between points and centroids.
        
        This is the core distance metric used in K-Means.
        For point X and centroid C: distance = sqrt(sum((x_i - c_i)^2))
        
        Args:
            X: Data points (n_samples, n_features)
            centroids: Cluster centroids (n_clusters, n_features)
        
        Returns:
            Distance matrix (n_samples, n_clusters)
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for i, centroid in enumerate(centroids):
            # Calculate squared differences for each dimension
            squared_diff = (X - centroid) ** 2
            # Sum across features and take square root
            distances[:, i] = np.sqrt(np.sum(squared_diff, axis=1))
        
        return distances
    
    def _initialize_centroids(self, X: np.ndarray, method: str = 'random') -> np.ndarray:
        """
        Initialize cluster centroids.
        
        Args:
            X: Input data (n_samples, n_features)
            method: Initialization method ('random' or 'kmeans++')
        
        Returns:
            Initial centroids (n_clusters, n_features)
        """
        n_samples, n_features = X.shape
        
        if method == 'random':
            # Simple random initialization: pick K random points
            # If K > n_samples, allow replacement
            replace = self.n_clusters > n_samples
            indices = np.random.choice(n_samples, self.n_clusters, replace=replace)
            return X[indices].copy()
        
        elif method == 'kmeans++':
            # K-Means++ initialization for better convergence
            # If K > n_samples, fall back to random initialization
            if self.n_clusters > n_samples:
                return self._initialize_centroids(X, method='random')
            
            centroids = np.zeros((self.n_clusters, n_features))
            
            # First centroid: random point
            centroids[0] = X[np.random.randint(n_samples)]
            
            # Subsequent centroids: weighted by distance to nearest existing centroid
            for i in range(1, self.n_clusters):
                distances = np.array([
                    min([np.linalg.norm(x - c) ** 2 for c in centroids[:i]])
                    for x in X
                ])
                dist_sum = distances.sum()
                # Handle case where all distances are zero (shouldn't happen with K <= n_samples)
                if dist_sum == 0 or np.isnan(dist_sum):
                    # Fall back to random selection
                    available_indices = [j for j in range(n_samples) 
                                       if not any(np.allclose(X[j], c) for c in centroids[:i])]
                    if len(available_indices) > 0:
                        centroids[i] = X[np.random.choice(available_indices)]
                    else:
                        # All points are already centroids, use random point
                        centroids[i] = X[np.random.randint(n_samples)]
                else:
                    probabilities = distances / dist_sum
                    centroids[i] = X[np.random.choice(n_samples, p=probabilities)]
            
            return centroids
        
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each point to nearest centroid.
        
        This is the "assignment step" of K-Means.
        
        Args:
            X: Data points (n_samples, n_features)
            centroids: Current centroids (n_clusters, n_features)
        
        Returns:
            Cluster labels (n_samples,)
        """
        distances = self._euclidean_distance(X, centroids)
        # Each point joins the cluster with minimum distance
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids to mean of assigned points.
        
        This is the "update step" of K-Means.
        
        Args:
            X: Data points (n_samples, n_features)
            labels: Current cluster assignments (n_samples,)
        
        Returns:
            Updated centroids (n_clusters, n_features)
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        for i in range(self.n_clusters):
            # Find all points in cluster i
            cluster_points = X[labels == i]
            
            if len(cluster_points) > 0:
                # Centroid = mean of all points in cluster
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster: reinitialize to random point
                new_centroids[i] = X[np.random.randint(len(X))]
        
        return new_centroids
    
    def _calculate_inertia(self, X: np.ndarray, centroids: np.ndarray, 
                          labels: np.ndarray) -> float:
        """
        Calculate within-cluster sum of squares (inertia).
        
        This is the objective function K-Means minimizes:
        J = sum of squared distances from points to their centroids
        
        Args:
            X: Data points
            centroids: Cluster centroids
            labels: Cluster assignments
        
        Returns:
            Inertia value
        """
        inertia = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroid = centroids[i]
                inertia += np.sum((cluster_points - centroid) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray, init_method: str = 'kmeans++', 
            verbose: bool = False) -> 'KMeansTheory':
        """
        Fit K-Means clustering to data.
        
        Args:
            X: Training data (n_samples, n_features)
            init_method: Centroid initialization method
            verbose: Print iteration details
        
        Returns:
            Self (fitted model)
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X, method=init_method)
        self.iteration_history = [self.centroids.copy()]
        
        for iteration in range(self.max_iterations):
            # Assignment step: assign points to nearest centroid
            labels = self._assign_clusters(X, self.centroids)
            
            # Update step: recalculate centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Calculate and store inertia
            inertia = self._calculate_inertia(X, new_centroids, labels)
            self.inertia_history.append(inertia)
            
            # Check convergence: has centroid movement stopped?
            centroid_shift = np.max(np.abs(new_centroids - self.centroids))
            
            if verbose:
                print(f"Iteration {iteration + 1}: Inertia = {inertia:.4f}, "
                      f"Centroid shift = {centroid_shift:.6f}")
            
            # Store state
            self.centroids = new_centroids
            self.labels = labels
            self.iteration_history.append(self.centroids.copy())
            
            # Convergence check
            if centroid_shift < self.tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: New data points (n_samples, n_features)
        
        Returns:
            Predicted cluster labels (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self._assign_clusters(X, self.centroids)
    
    def get_inertia(self) -> float:
        """Get final inertia value."""
        return self.inertia_history[-1] if self.inertia_history else None


class KMeansVisualizer:
    """Visualize K-Means algorithm steps for educational purposes."""
    
    @staticmethod
    def plot_algorithm_steps(X: np.ndarray, kmeans: KMeansTheory, 
                           steps_to_show: List[int] = None):
        """
        Visualize K-Means iterations showing centroid movement and assignments.
        
        Args:
            X: Data points (2D for visualization)
            kmeans: Fitted KMeansTheory model
            steps_to_show: Which iterations to display
        """
        if steps_to_show is None:
            # Show initialization, middle iterations, and final state
            n_history = len(kmeans.iteration_history)
            if n_history <= 4:
                steps_to_show = list(range(n_history))
            else:
                steps_to_show = [0, n_history // 3, 2 * n_history // 3, n_history - 1]
        
        n_steps = len(steps_to_show)
        fig, axes = plt.subplots(1, n_steps, figsize=(5 * n_steps, 4))
        
        if n_steps == 1:
            axes = [axes]
        
        colors = sns.color_palette("husl", kmeans.n_clusters)
        
        for idx, step in enumerate(steps_to_show):
            ax = axes[idx]
            centroids = kmeans.iteration_history[step]
            
            # Assign points to nearest centroid at this iteration
            labels = kmeans._assign_clusters(X, centroids)
            
            # Plot points colored by cluster
            for i in range(kmeans.n_clusters):
                cluster_points = X[labels == i]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                          c=[colors[i]], alpha=0.6, s=50, label=f'Cluster {i}')
            
            # Plot centroids
            ax.scatter(centroids[:, 0], centroids[:, 1],
                      c='red', marker='X', s=300, edgecolors='black',
                      linewidths=2, label='Centroids', zorder=5)
            
            title = "Initial State" if step == 0 else f"Iteration {step}"
            if step == len(kmeans.iteration_history) - 1:
                title = "Final Clustering"
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_convergence(kmeans: KMeansTheory):
        """
        Plot inertia over iterations to show convergence.
        
        Args:
            kmeans: Fitted KMeansTheory model
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(len(kmeans.inertia_history))
        ax.plot(iterations, kmeans.inertia_history, 
               marker='o', linewidth=2, markersize=8)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        ax.set_title('K-Means Convergence: Inertia Reduction Over Time', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Annotate final inertia
        final_inertia = kmeans.inertia_history[-1]
        ax.annotate(f'Final Inertia: {final_inertia:.2f}',
                   xy=(len(kmeans.inertia_history) - 1, final_inertia),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        return fig


def generate_sample_data(n_samples: int = 300, n_clusters: int = 3, 
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic clustered data for demonstration.
    
    Args:
        n_samples: Total number of samples
        n_clusters: Number of natural clusters
        random_state: Random seed
    
    Returns:
        X: Data points (n_samples, 2)
        true_labels: Actual cluster labels
    """
    np.random.seed(random_state)
    
    X = []
    true_labels = []
    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples % n_clusters
    
    # Generate clusters at different locations
    for i in range(n_clusters):
        # Random center for each cluster
        center_x = np.random.uniform(-10, 10)
        center_y = np.random.uniform(-10, 10)
        
        # Add remainder samples to the last cluster to ensure exact count
        cluster_size = samples_per_cluster + (1 if i == n_clusters - 1 else 0) * remainder
        
        # Generate points around center with some spread
        cluster_x = np.random.normal(center_x, 1.5, cluster_size)
        cluster_y = np.random.normal(center_y, 1.5, cluster_size)
        
        cluster_points = np.column_stack([cluster_x, cluster_y])
        X.append(cluster_points)
        true_labels.extend([i] * cluster_size)
    
    X = np.vstack(X)
    true_labels = np.array(true_labels)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    true_labels = true_labels[shuffle_idx]
    
    return X, true_labels


def demonstrate_distance_calculation():
    """Demonstrate Euclidean distance calculation."""
    print("=" * 60)
    print("DEMONSTRATION: Euclidean Distance Calculation")
    print("=" * 60)
    
    # Example: 2D points
    point = np.array([3.0, 4.0])
    centroid1 = np.array([0.0, 0.0])
    centroid2 = np.array([6.0, 8.0])
    
    # Calculate distances manually
    dist1 = np.sqrt((point[0] - centroid1[0])**2 + (point[1] - centroid1[1])**2)
    dist2 = np.sqrt((point[0] - centroid2[0])**2 + (point[1] - centroid2[1])**2)
    
    print(f"\nPoint: {point}")
    print(f"Centroid 1: {centroid1}")
    print(f"Centroid 2: {centroid2}")
    print(f"\nDistance to Centroid 1: {dist1:.4f}")
    print(f"Distance to Centroid 2: {dist2:.4f}")
    print(f"\nPoint assigned to: Centroid {1 if dist1 < dist2 else 2}")
    print()


def demonstrate_centroid_update():
    """Demonstrate centroid update calculation."""
    print("=" * 60)
    print("DEMONSTRATION: Centroid Update")
    print("=" * 60)
    
    # Example cluster with 4 points
    cluster_points = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [2.0, 2.0]
    ])
    
    print(f"\nCluster points:")
    for i, point in enumerate(cluster_points):
        print(f"  Point {i+1}: {point}")
    
    # Calculate new centroid (mean)
    new_centroid = cluster_points.mean(axis=0)
    
    print(f"\nNew centroid (mean of all points): {new_centroid}")
    print(f"Calculation: [{cluster_points[:, 0].mean():.2f}, "
          f"{cluster_points[:, 1].mean():.2f}]")
    print()


def main():
    """Main demonstration of K-Means theory."""
    print("\n" + "=" * 60)
    print("Day 86: K-Means Clustering Theory - Implementation Demo")
    print("=" * 60 + "\n")
    
    # Part 1: Distance calculation demo
    demonstrate_distance_calculation()
    
    # Part 2: Centroid update demo
    demonstrate_centroid_update()
    
    # Part 3: Full algorithm demonstration
    print("=" * 60)
    print("FULL ALGORITHM: K-Means on Synthetic Data")
    print("=" * 60 + "\n")
    
    # Generate sample data
    print("Generating synthetic clustered data...")
    X, true_labels = generate_sample_data(n_samples=300, n_clusters=3, random_state=42)
    print(f"Generated {len(X)} points with 3 natural clusters\n")
    
    # Fit K-Means
    print("Fitting K-Means algorithm (K=3)...")
    kmeans = KMeansTheory(n_clusters=3, random_state=42)
    kmeans.fit(X, init_method='kmeans++', verbose=True)
    
    print(f"\nFinal Results:")
    print(f"  Iterations: {len(kmeans.iteration_history) - 1}")
    print(f"  Final Inertia: {kmeans.get_inertia():.4f}")
    
    # Visualize algorithm steps
    print("\nGenerating visualizations...")
    
    fig1 = KMeansVisualizer.plot_algorithm_steps(X, kmeans)
    plt.savefig('kmeans_iterations.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: kmeans_iterations.png")
    
    fig2 = KMeansVisualizer.plot_convergence(kmeans)
    plt.savefig('kmeans_convergence.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: kmeans_convergence.png")
    
    # Prediction on new point
    print("\nPrediction Example:")
    new_point = np.array([[2.0, 3.0]])
    predicted_cluster = kmeans.predict(new_point)
    print(f"  New point {new_point[0]} assigned to cluster {predicted_cluster[0]}")
    
    print("\n" + "=" * 60)
    print("Demo complete! Check the generated PNG files.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
