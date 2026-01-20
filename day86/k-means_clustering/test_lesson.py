"""
Test suite for Day 86: K-Means Clustering Theory
"""

import pytest
import numpy as np
from lesson_code import (
    KMeansTheory, 
    KMeansVisualizer, 
    generate_sample_data,
    demonstrate_distance_calculation,
    demonstrate_centroid_update
)


class TestKMeansTheory:
    """Test KMeansTheory implementation."""
    
    def test_initialization(self):
        """Test model initialization."""
        kmeans = KMeansTheory(n_clusters=3, max_iterations=100, tolerance=1e-4)
        assert kmeans.n_clusters == 3
        assert kmeans.max_iterations == 100
        assert kmeans.tolerance == 1e-4
        assert kmeans.centroids is None
        assert kmeans.labels is None
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        kmeans = KMeansTheory(n_clusters=2)
        
        X = np.array([[0, 0], [3, 4]])
        centroids = np.array([[0, 0], [6, 8]])
        
        distances = kmeans._euclidean_distance(X, centroids)
        
        # Point [0,0] should have distance 0 to centroid [0,0]
        assert np.isclose(distances[0, 0], 0.0)
        # Point [3,4] should have distance 5 to centroid [0,0]
        assert np.isclose(distances[1, 0], 5.0)
        # Point [3,4] should have distance 5 to centroid [6,8]
        assert np.isclose(distances[1, 1], 5.0)
    
    def test_initialize_centroids_random(self):
        """Test random centroid initialization."""
        X = np.random.randn(100, 2)
        kmeans = KMeansTheory(n_clusters=3, random_state=42)
        
        centroids = kmeans._initialize_centroids(X, method='random')
        
        assert centroids.shape == (3, 2)
        # Centroids should be from the dataset
        for centroid in centroids:
            assert any(np.allclose(centroid, x) for x in X)
    
    def test_initialize_centroids_kmeanspp(self):
        """Test K-Means++ initialization."""
        X = np.random.randn(100, 2)
        kmeans = KMeansTheory(n_clusters=3, random_state=42)
        
        centroids = kmeans._initialize_centroids(X, method='kmeans++')
        
        assert centroids.shape == (3, 2)
        # K-Means++ centroids should be spread apart
        # Check that centroids are not too close to each other
        min_distance = np.min([
            np.linalg.norm(centroids[i] - centroids[j])
            for i in range(3) for j in range(i+1, 3)
        ])
        assert min_distance > 0.1  # Should have some separation
    
    def test_assign_clusters(self):
        """Test cluster assignment."""
        kmeans = KMeansTheory(n_clusters=2)
        
        # Simple 2D data
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        centroids = np.array([[0.5, 0.5], [10.5, 10.5]])
        
        labels = kmeans._assign_clusters(X, centroids)
        
        # First two points should be in cluster 0
        assert labels[0] == 0
        assert labels[1] == 0
        # Last two points should be in cluster 1
        assert labels[2] == 1
        assert labels[3] == 1
    
    def test_update_centroids(self):
        """Test centroid update."""
        kmeans = KMeansTheory(n_clusters=2)
        
        X = np.array([[0, 0], [2, 2], [10, 10], [12, 12]])
        labels = np.array([0, 0, 1, 1])
        
        new_centroids = kmeans._update_centroids(X, labels)
        
        # Cluster 0 centroid should be mean of [0,0] and [2,2]
        assert np.allclose(new_centroids[0], [1, 1])
        # Cluster 1 centroid should be mean of [10,10] and [12,12]
        assert np.allclose(new_centroids[1], [11, 11])
    
    def test_calculate_inertia(self):
        """Test inertia calculation."""
        kmeans = KMeansTheory(n_clusters=2)
        
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        centroids = np.array([[0.5, 0.5], [10.5, 10.5]])
        labels = np.array([0, 0, 1, 1])
        
        inertia = kmeans._calculate_inertia(X, centroids, labels)
        
        # Inertia should be sum of squared distances
        expected_inertia = (
            (0 - 0.5)**2 + (0 - 0.5)**2 +  # Point [0,0] to centroid [0.5,0.5]
            (1 - 0.5)**2 + (1 - 0.5)**2 +  # Point [1,1] to centroid [0.5,0.5]
            (10 - 10.5)**2 + (10 - 10.5)**2 +  # Point [10,10] to centroid [10.5,10.5]
            (11 - 10.5)**2 + (11 - 10.5)**2    # Point [11,11] to centroid [10.5,10.5]
        )
        assert np.isclose(inertia, expected_inertia)
    
    def test_fit(self):
        """Test complete fitting process."""
        X, _ = generate_sample_data(n_samples=100, n_clusters=3, random_state=42)
        
        kmeans = KMeansTheory(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        # Model should be fitted
        assert kmeans.centroids is not None
        assert kmeans.labels is not None
        assert kmeans.centroids.shape == (3, 2)
        assert kmeans.labels.shape == (100,)
        assert len(kmeans.iteration_history) > 0
        assert len(kmeans.inertia_history) > 0
        
        # All labels should be in valid range
        assert all(0 <= label < 3 for label in kmeans.labels)
    
    def test_predict(self):
        """Test prediction on new data."""
        X, _ = generate_sample_data(n_samples=100, n_clusters=3, random_state=42)
        
        kmeans = KMeansTheory(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        # Predict on new points
        new_points = np.array([[0, 0], [5, 5], [10, 10]])
        predictions = kmeans.predict(new_points)
        
        assert predictions.shape == (3,)
        assert all(0 <= pred < 3 for pred in predictions)
    
    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error if not fitted."""
        kmeans = KMeansTheory(n_clusters=3)
        X = np.array([[0, 0], [1, 1]])
        
        with pytest.raises(ValueError, match="Model not fitted"):
            kmeans.predict(X)
    
    def test_convergence(self):
        """Test that algorithm converges."""
        X, _ = generate_sample_data(n_samples=100, n_clusters=3, random_state=42)
        
        kmeans = KMeansTheory(n_clusters=3, tolerance=1e-4, random_state=42)
        kmeans.fit(X)
        
        # Inertia should decrease over iterations
        inertia_values = kmeans.inertia_history
        assert all(inertia_values[i] >= inertia_values[i+1] 
                  for i in range(len(inertia_values)-1))
    
    def test_get_inertia(self):
        """Test getting final inertia value."""
        X, _ = generate_sample_data(n_samples=100, n_clusters=3, random_state=42)
        
        kmeans = KMeansTheory(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        inertia = kmeans.get_inertia()
        assert inertia is not None
        assert inertia > 0
        assert inertia == kmeans.inertia_history[-1]


class TestKMeansVisualizer:
    """Test visualization functions."""
    
    def test_plot_algorithm_steps(self):
        """Test algorithm steps visualization."""
        X, _ = generate_sample_data(n_samples=50, n_clusters=2, random_state=42)
        kmeans = KMeansTheory(n_clusters=2, random_state=42)
        kmeans.fit(X)
        
        # Should create figure without errors
        fig = KMeansVisualizer.plot_algorithm_steps(X, kmeans)
        assert fig is not None
    
    def test_plot_convergence(self):
        """Test convergence plot."""
        X, _ = generate_sample_data(n_samples=50, n_clusters=2, random_state=42)
        kmeans = KMeansTheory(n_clusters=2, random_state=42)
        kmeans.fit(X)
        
        # Should create figure without errors
        fig = KMeansVisualizer.plot_convergence(kmeans)
        assert fig is not None


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        X, labels = generate_sample_data(n_samples=90, n_clusters=3, random_state=42)
        
        assert X.shape[0] == 90
        assert X.shape[1] == 2
        assert labels.shape == (90,)
        assert set(labels) == {0, 1, 2}
        
        # Each cluster should have approximately 30 points
        for i in range(3):
            cluster_count = np.sum(labels == i)
            assert 25 <= cluster_count <= 35  # Allow some variation
    
    def test_demonstrate_distance_calculation(self):
        """Test distance calculation demo runs without error."""
        # Should run without raising exceptions
        demonstrate_distance_calculation()
    
    def test_demonstrate_centroid_update(self):
        """Test centroid update demo runs without error."""
        # Should run without raising exceptions
        demonstrate_centroid_update()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_cluster_handling(self):
        """Test handling of empty clusters during update."""
        kmeans = KMeansTheory(n_clusters=3, random_state=42)
        
        # Create scenario where one cluster is empty
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 0, 0])  # All points in cluster 0
        
        # Should handle empty clusters without error
        new_centroids = kmeans._update_centroids(X, labels)
        assert new_centroids.shape == (3, 2)
    
    def test_single_cluster(self):
        """Test K-Means with K=1."""
        X, _ = generate_sample_data(n_samples=50, n_clusters=2, random_state=42)
        
        kmeans = KMeansTheory(n_clusters=1, random_state=42)
        kmeans.fit(X)
        
        # All points should be in cluster 0
        assert all(label == 0 for label in kmeans.labels)
        assert kmeans.centroids.shape == (1, 2)
    
    def test_more_clusters_than_points(self):
        """Test K > number of data points."""
        X = np.array([[0, 0], [1, 1]])
        
        kmeans = KMeansTheory(n_clusters=5, random_state=42)
        kmeans.fit(X)
        
        # Should still run (though not meaningful)
        assert kmeans.centroids is not None


def test_integration():
    """Integration test: complete workflow."""
    # Generate data
    X, true_labels = generate_sample_data(n_samples=150, n_clusters=3, random_state=42)
    
    # Fit model
    kmeans = KMeansTheory(n_clusters=3, random_state=42)
    kmeans.fit(X, init_method='kmeans++', verbose=False)
    
    # Make predictions
    predictions = kmeans.predict(X[:10])
    
    # Verify results
    assert kmeans.centroids.shape == (3, 2)
    assert kmeans.labels.shape == (150,)
    assert predictions.shape == (10,)
    assert kmeans.get_inertia() > 0
    
    # Check convergence
    assert len(kmeans.inertia_history) > 0
    final_inertia = kmeans.inertia_history[-1]
    initial_inertia = kmeans.inertia_history[0]
    assert final_inertia < initial_inertia  # Should improve


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
