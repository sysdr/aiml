"""
Day 90: Hierarchical Clustering Implementation
Production-ready hierarchical clustering with multiple linkage methods
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import json


class HierarchicalClusterer:
    """
    Production-ready hierarchical clustering with support for multiple linkage methods.
    
    Used in production systems like Netflix's genre taxonomy and Spotify's music clustering.
    Supports agglomerative clustering with single, complete, average, and Ward linkage.
    """
    
    def __init__(
        self,
        linkage_method: str = 'ward',
        distance_metric: str = 'euclidean',
        distance_threshold: Optional[float] = None,
        n_clusters: Optional[int] = None
    ):
        """
        Initialize hierarchical clusterer.
        
        Args:
            linkage_method: 'single', 'complete', 'average', or 'ward'
            distance_metric: Distance metric for computing pairwise distances
            distance_threshold: Height at which to cut dendrogram (alternative to n_clusters)
            n_clusters: Number of clusters to extract (alternative to distance_threshold)
        """
        valid_methods = ['single', 'complete', 'average', 'ward']
        if linkage_method not in valid_methods:
            raise ValueError(f"linkage_method must be one of {valid_methods}")
        
        if distance_threshold is None and n_clusters is None:
            distance_threshold = 2.5  # Default threshold
        
        if distance_threshold is not None and n_clusters is not None:
            raise ValueError("Specify either distance_threshold or n_clusters, not both")
        
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.distance_threshold = distance_threshold
        self.n_clusters = n_clusters
        self.linkage_matrix_ = None
        self.labels_ = None
        self.n_samples_ = None
        
    def fit(self, X: np.ndarray) -> 'HierarchicalClusterer':
        """
        Compute hierarchical clustering.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            self: Fitted clusterer
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        
        self.n_samples_ = X.shape[0]
        
        # Compute linkage matrix using scipy
        # This performs the agglomerative clustering algorithm
        self.linkage_matrix_ = linkage(X, method=self.linkage_method, metric=self.distance_metric)
        
        return self
    
    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict cluster labels by cutting the dendrogram.
        
        Args:
            X: Not used, present for API consistency
            
        Returns:
            labels: Cluster labels for each sample
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Must call fit() before predict()")
        
        # Cut dendrogram to get flat clusters
        if self.n_clusters is not None:
            self.labels_ = fcluster(
                self.linkage_matrix_,
                self.n_clusters,
                criterion='maxclust'
            )
        else:
            self.labels_ = fcluster(
                self.linkage_matrix_,
                self.distance_threshold,
                criterion='distance'
            )
        
        # Convert to 0-indexed labels
        self.labels_ = self.labels_ - 1
        
        return self.labels_
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and return cluster labels.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            labels: Cluster labels for each sample
        """
        self.fit(X)
        return self.predict()
    
    def get_linkage_matrix(self) -> np.ndarray:
        """
        Get the linkage matrix for custom analysis.
        
        Returns:
            linkage_matrix: Scipy linkage matrix of shape (n_samples-1, 4)
                           Each row contains [cluster_i, cluster_j, distance, n_samples]
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Must call fit() before get_linkage_matrix()")
        
        return self.linkage_matrix_
    
    def plot_dendrogram(
        self,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        show: bool = True
    ) -> None:
        """
        Plot the dendrogram visualization.
        
        Args:
            labels: Optional labels for leaf nodes
            save_path: Path to save the figure
            figsize: Figure size (width, height)
            show: Whether to display the plot
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Must call fit() before plot_dendrogram()")
        
        plt.figure(figsize=figsize)
        
        dendrogram_dict = dendrogram(
            self.linkage_matrix_,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=10,
            color_threshold=self.distance_threshold if self.distance_threshold else None
        )
        
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage_method.capitalize()} Linkage)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index or Label', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        
        if self.distance_threshold:
            plt.axhline(y=self.distance_threshold, color='r', linestyle='--', 
                       label=f'Cut threshold: {self.distance_threshold}')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dendrogram saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_cluster_sizes(self) -> dict:
        """
        Get the size of each cluster.
        
        Returns:
            cluster_sizes: Dictionary mapping cluster_id to size
        """
        if self.labels_ is None:
            raise ValueError("Must call predict() before get_cluster_sizes()")
        
        unique, counts = np.unique(self.labels_, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


class ContentTaxonomyBuilder:
    """
    Production system for building hierarchical content taxonomies.
    
    Similar to Netflix's genre taxonomy or Spotify's music clustering.
    Processes content embeddings and generates multi-level hierarchies.
    """
    
    def __init__(self, linkage_method: str = 'ward'):
        """
        Initialize taxonomy builder.
        
        Args:
            linkage_method: Clustering linkage method
        """
        self.linkage_method = linkage_method
        self.clusterer = None
        self.taxonomy_tree = None
        
    def build_taxonomy(
        self,
        embeddings: np.ndarray,
        item_ids: List[str],
        n_levels: int = 3
    ) -> dict:
        """
        Build multi-level content taxonomy from embeddings.
        
        Args:
            embeddings: Content embeddings of shape (n_items, embedding_dim)
            item_ids: List of item identifiers
            n_levels: Number of hierarchy levels to generate
            
        Returns:
            taxonomy: Nested dictionary representing the taxonomy tree
        """
        taxonomy = {'root': {'children': {}, 'items': item_ids.copy()}}
        
        # Build hierarchies at different granularities
        for level in range(1, n_levels + 1):
            n_clusters = 2 ** level  # Exponential growth: 2, 4, 8 clusters
            
            clusterer = HierarchicalClusterer(
                linkage_method=self.linkage_method,
                n_clusters=n_clusters
            )
            
            labels = clusterer.fit_predict(embeddings)
            
            # Store this level's clustering
            level_key = f'level_{level}'
            taxonomy[level_key] = {}
            
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                cluster_items = [item_ids[i] for i in np.where(mask)[0]]
                taxonomy[level_key][f'cluster_{cluster_id}'] = cluster_items
        
        self.taxonomy_tree = taxonomy
        return taxonomy
    
    def save_taxonomy(self, filepath: str) -> None:
        """
        Save taxonomy to JSON file.
        
        Args:
            filepath: Path to save the taxonomy
        """
        if self.taxonomy_tree is None:
            raise ValueError("Must call build_taxonomy() before save_taxonomy()")
        
        with open(filepath, 'w') as f:
            json.dump(self.taxonomy_tree, f, indent=2)
        
        print(f"Taxonomy saved to {filepath}")
    
    def get_cluster_at_level(self, level: int, cluster_id: int) -> List[str]:
        """
        Get items in a specific cluster at a specific level.
        
        Args:
            level: Hierarchy level (1, 2, 3, ...)
            cluster_id: Cluster identifier at that level
            
        Returns:
            items: List of item IDs in the cluster
        """
        if self.taxonomy_tree is None:
            raise ValueError("Must call build_taxonomy() before get_cluster_at_level()")
        
        level_key = f'level_{level}'
        cluster_key = f'cluster_{cluster_id}'
        
        return self.taxonomy_tree.get(level_key, {}).get(cluster_key, [])


def demonstrate_linkage_methods():
    """
    Demonstrate different linkage methods on synthetic data.
    Shows how linkage choice affects cluster structure.
    """
    # Generate synthetic data with clear cluster structure
    np.random.seed(42)
    
    # Three clusters with different characteristics
    cluster1 = np.random.randn(30, 2) + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) + np.array([5, 5])
    cluster3 = np.random.randn(30, 2) + np.array([10, 0])
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    linkage_methods = ['single', 'complete', 'average', 'ward']
    
    print("Comparing Linkage Methods:")
    print("-" * 60)
    
    for method in linkage_methods:
        clusterer = HierarchicalClusterer(
            linkage_method=method,
            n_clusters=3
        )
        
        labels = clusterer.fit_predict(X)
        cluster_sizes = clusterer.get_cluster_sizes()
        
        print(f"\n{method.upper()} Linkage:")
        print(f"  Cluster sizes: {sorted(cluster_sizes.values())}")
        print(f"  Number of clusters: {len(cluster_sizes)}")


def run_content_taxonomy_example():
    """
    Example: Build a movie taxonomy similar to Netflix's genre system.
    """
    print("\n" + "=" * 60)
    print("Content Taxonomy Example: Movie Genre Clustering")
    print("=" * 60)
    
    # Simulate movie embeddings (in production, these come from neural networks)
    np.random.seed(42)
    n_movies = 100
    embedding_dim = 50
    
    # Generate synthetic movie embeddings
    movie_embeddings = np.random.randn(n_movies, embedding_dim)
    movie_ids = [f"movie_{i:03d}" for i in range(n_movies)]
    
    print(f"\nProcessing {n_movies} movies with {embedding_dim}-dimensional embeddings...")
    
    # Build hierarchical taxonomy
    builder = ContentTaxonomyBuilder(linkage_method='ward')
    taxonomy = builder.build_taxonomy(
        embeddings=movie_embeddings,
        item_ids=movie_ids,
        n_levels=3
    )
    
    # Display taxonomy structure
    print("\nTaxonomy Structure:")
    for level in range(1, 4):
        level_key = f'level_{level}'
        n_clusters = len(taxonomy[level_key])
        print(f"  Level {level}: {n_clusters} clusters")
        
        # Show cluster sizes
        for cluster_id in range(n_clusters):
            cluster_items = builder.get_cluster_at_level(level, cluster_id)
            print(f"    Cluster {cluster_id}: {len(cluster_items)} movies")
    
    # Save taxonomy
    builder.save_taxonomy('movie_taxonomy.json')
    
    # Visualize dendrogram for top-level clustering
    print("\nGenerating dendrogram visualization...")
    clusterer = HierarchicalClusterer(linkage_method='ward', n_clusters=8)
    clusterer.fit_predict(movie_embeddings)
    clusterer.plot_dendrogram(
        save_path='movie_dendrogram.png',
        show=False
    )


if __name__ == "__main__":
    # Demonstrate different linkage methods
    demonstrate_linkage_methods()
    
    # Run content taxonomy example
    run_content_taxonomy_example()
    
    print("\n" + "=" * 60)
    print("Day 90 Complete! Check the generated files:")
    print("  - movie_taxonomy.json: Multi-level taxonomy structure")
    print("  - movie_dendrogram.png: Hierarchical clustering visualization")
    print("=" * 60)
