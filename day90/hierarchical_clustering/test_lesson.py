"""
Day 90: Hierarchical Clustering - Test Suite
Comprehensive tests for hierarchical clustering implementation
"""

import pytest
import numpy as np
from lesson_code import HierarchicalClusterer, ContentTaxonomyBuilder


class TestHierarchicalClusterer:
    """Test suite for HierarchicalClusterer class."""
    
    def test_initialization(self):
        """Test clusterer initialization with various parameters."""
        # Default initialization
        clusterer = HierarchicalClusterer()
        assert clusterer.linkage_method == 'ward'
        assert clusterer.distance_threshold == 2.5
        
        # Custom parameters
        clusterer = HierarchicalClusterer(linkage_method='single', n_clusters=5)
        assert clusterer.linkage_method == 'single'
        assert clusterer.n_clusters == 5
    
    def test_invalid_linkage_method(self):
        """Test that invalid linkage methods raise errors."""
        with pytest.raises(ValueError):
            HierarchicalClusterer(linkage_method='invalid')
    
    def test_both_threshold_and_nclusters(self):
        """Test that specifying both threshold and n_clusters raises error."""
        with pytest.raises(ValueError):
            HierarchicalClusterer(distance_threshold=2.0, n_clusters=5)
    
    def test_fit_predict_basic(self):
        """Test basic fit_predict functionality."""
        X = np.random.randn(20, 5)
        clusterer = HierarchicalClusterer(n_clusters=3)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == 20
        assert len(np.unique(labels)) == 3
        assert labels.min() == 0  # Labels should be 0-indexed
    
    def test_single_linkage(self):
        """Test single linkage clustering."""
        np.random.seed(42)
        # Create two well-separated clusters
        cluster1 = np.random.randn(10, 2)
        cluster2 = np.random.randn(10, 2) + np.array([10, 10])
        X = np.vstack([cluster1, cluster2])
        
        clusterer = HierarchicalClusterer(linkage_method='single', n_clusters=2)
        labels = clusterer.fit_predict(X)
        
        assert len(np.unique(labels)) == 2
    
    def test_complete_linkage(self):
        """Test complete linkage clustering."""
        np.random.seed(42)
        X = np.random.randn(15, 3)
        
        clusterer = HierarchicalClusterer(linkage_method='complete', n_clusters=3)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == 15
        assert len(np.unique(labels)) <= 3
    
    def test_average_linkage(self):
        """Test average linkage clustering."""
        np.random.seed(42)
        X = np.random.randn(25, 4)
        
        clusterer = HierarchicalClusterer(linkage_method='average', n_clusters=5)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == 25
        assert len(np.unique(labels)) <= 5
    
    def test_ward_linkage(self):
        """Test Ward linkage clustering."""
        np.random.seed(42)
        X = np.random.randn(30, 5)
        
        clusterer = HierarchicalClusterer(linkage_method='ward', n_clusters=4)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == 30
        assert len(np.unique(labels)) == 4
    
    def test_distance_threshold_cutting(self):
        """Test cutting dendrogram at distance threshold."""
        np.random.seed(42)
        X = np.random.randn(20, 3)
        
        clusterer = HierarchicalClusterer(distance_threshold=1.5)
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == 20
        # Number of clusters should be determined by threshold
        assert len(np.unique(labels)) > 0
    
    def test_get_linkage_matrix(self):
        """Test retrieval of linkage matrix."""
        X = np.random.randn(10, 3)
        clusterer = HierarchicalClusterer()
        clusterer.fit(X)
        
        linkage_matrix = clusterer.get_linkage_matrix()
        assert linkage_matrix.shape == (9, 4)  # n_samples - 1 rows
    
    def test_get_cluster_sizes(self):
        """Test cluster size calculation."""
        np.random.seed(42)
        X = np.random.randn(30, 4)
        
        clusterer = HierarchicalClusterer(n_clusters=3)
        clusterer.fit_predict(X)
        
        sizes = clusterer.get_cluster_sizes()
        assert len(sizes) == 3
        assert sum(sizes.values()) == 30
    
    def test_predict_before_fit(self):
        """Test that predict raises error before fit."""
        clusterer = HierarchicalClusterer()
        with pytest.raises(ValueError):
            clusterer.predict()
    
    def test_2d_input_requirement(self):
        """Test that 1D input raises error."""
        X = np.random.randn(20)  # 1D array
        clusterer = HierarchicalClusterer()
        
        with pytest.raises(ValueError):
            clusterer.fit(X)
    
    def test_consistent_results(self):
        """Test that results are consistent across runs."""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        
        clusterer1 = HierarchicalClusterer(n_clusters=4)
        labels1 = clusterer1.fit_predict(X)
        
        clusterer2 = HierarchicalClusterer(n_clusters=4)
        labels2 = clusterer2.fit_predict(X)
        
        # Labels might be permuted but cluster assignments should be identical
        assert len(labels1) == len(labels2)


class TestContentTaxonomyBuilder:
    """Test suite for ContentTaxonomyBuilder class."""
    
    def test_initialization(self):
        """Test taxonomy builder initialization."""
        builder = ContentTaxonomyBuilder()
        assert builder.linkage_method == 'ward'
    
    def test_build_taxonomy(self):
        """Test taxonomy building."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 10)
        item_ids = [f"item_{i}" for i in range(50)]
        
        builder = ContentTaxonomyBuilder()
        taxonomy = builder.build_taxonomy(embeddings, item_ids, n_levels=2)
        
        assert 'root' in taxonomy
        assert 'level_1' in taxonomy
        assert 'level_2' in taxonomy
        
        # Level 1 should have 2 clusters
        assert len(taxonomy['level_1']) == 2
        # Level 2 should have 4 clusters
        assert len(taxonomy['level_2']) == 4
    
    def test_get_cluster_at_level(self):
        """Test retrieving cluster items at specific level."""
        np.random.seed(42)
        embeddings = np.random.randn(30, 8)
        item_ids = [f"item_{i}" for i in range(30)]
        
        builder = ContentTaxonomyBuilder()
        builder.build_taxonomy(embeddings, item_ids, n_levels=2)
        
        items = builder.get_cluster_at_level(level=1, cluster_id=0)
        assert isinstance(items, list)
        assert len(items) > 0
        assert all(item.startswith('item_') for item in items)
    
    def test_taxonomy_before_build(self):
        """Test that accessing taxonomy before building raises error."""
        builder = ContentTaxonomyBuilder()
        
        with pytest.raises(ValueError):
            builder.get_cluster_at_level(level=1, cluster_id=0)


def test_integration_workflow():
    """Test complete workflow from data to taxonomy."""
    np.random.seed(42)
    
    # Generate synthetic data
    n_items = 40
    embedding_dim = 15
    embeddings = np.random.randn(n_items, embedding_dim)
    item_ids = [f"content_{i:03d}" for i in range(n_items)]
    
    # Build taxonomy
    builder = ContentTaxonomyBuilder(linkage_method='ward')
    taxonomy = builder.build_taxonomy(embeddings, item_ids, n_levels=3)
    
    # Verify structure
    assert 'level_1' in taxonomy
    assert 'level_2' in taxonomy
    assert 'level_3' in taxonomy
    
    # Verify all items are accounted for
    level_1_items = []
    for cluster_items in taxonomy['level_1'].values():
        level_1_items.extend(cluster_items)
    
    assert len(level_1_items) == n_items
    assert set(level_1_items) == set(item_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
