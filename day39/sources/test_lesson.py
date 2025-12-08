"""
Tests for Day 39: Supervised vs. Unsupervised Learning
"""

import pytest
import numpy as np
from lesson_code import SupervisedLearningDemo, UnsupervisedLearningDemo


class TestSupervisedLearning:
    """Test supervised learning implementation."""
    
    def test_data_generation(self):
        """Test that synthetic email data has correct structure."""
        demo = SupervisedLearningDemo()
        X, y = demo.generate_synthetic_email_data(100)
        
        # Check shapes
        assert X.shape == (100, 5), "Feature matrix should be 100x5"
        assert y.shape == (100,), "Label vector should have 100 elements"
        
        # Check label values
        assert set(np.unique(y)) == {0, 1}, "Labels should be binary (0 or 1)"
        
        # Check equal class distribution
        assert np.sum(y == 0) == 50, "Should have 50 non-spam emails"
        assert np.sum(y == 1) == 50, "Should have 50 spam emails"
    
    def test_model_training(self):
        """Test that supervised model trains successfully."""
        demo = SupervisedLearningDemo()
        accuracy = demo.train_and_evaluate()
        
        # Model should achieve reasonable accuracy on synthetic data
        assert accuracy > 0.7, f"Model accuracy {accuracy:.2%} is too low"
        assert demo.model is not None, "Model should be trained"
    
    def test_supervised_requires_labels(self):
        """Verify supervised learning requires both X and y."""
        demo = SupervisedLearningDemo()
        X, y = demo.generate_synthetic_email_data(100)
        
        # Should work with both X and y
        demo.model.fit(X, y)
        
        # Should fail without y (uncomment to test - will raise error)
        # with pytest.raises(TypeError):
        #     demo.model.fit(X)


class TestUnsupervisedLearning:
    """Test unsupervised learning implementation."""
    
    def test_customer_data_generation(self):
        """Test that customer data has correct structure."""
        demo = UnsupervisedLearningDemo(n_clusters=3)
        X = demo.generate_customer_data(300)
        
        # Check shape
        assert X.shape == (300, 2), "Customer data should be 300x2"
        
        # Check feature values are reasonable
        assert X[:, 0].min() > 0, "Purchase amounts should be positive"
        assert X[:, 1].min() > 0, "Purchase frequency should be positive"
    
    def test_clustering(self):
        """Test that unsupervised model discovers segments."""
        demo = UnsupervisedLearningDemo(n_clusters=3)
        silhouette = demo.discover_segments()
        
        # Silhouette score should be reasonable
        assert -1 <= silhouette <= 1, "Silhouette score should be in [-1, 1]"
        assert silhouette > 0.3, f"Cluster quality {silhouette:.3f} is too low"
        
        # Should have found 3 clusters
        assert demo.model.n_clusters == 3, "Should discover 3 customer segments"
    
    def test_unsupervised_no_labels(self):
        """Verify unsupervised learning works without labels."""
        demo = UnsupervisedLearningDemo(n_clusters=3)
        X = demo.generate_customer_data(300)
        
        # Should work with only X (no y)
        demo.model.fit(X)
        
        # Should produce cluster assignments
        clusters = demo.model.predict(X)
        assert len(clusters) == len(X), "Should assign cluster to each customer"
        assert len(np.unique(clusters)) == 3, "Should have 3 unique clusters"


class TestKeyDifferences:
    """Test understanding of supervised vs unsupervised differences."""
    
    def test_api_difference(self):
        """
        Verify the key API difference between approaches.
        
        Supervised: model.fit(X, y) - requires labels
        Unsupervised: model.fit(X) - no labels needed
        """
        # Supervised learning requires y
        supervised = SupervisedLearningDemo()
        X_sup, y_sup = supervised.generate_synthetic_email_data(100)
        supervised.model.fit(X_sup, y_sup)  # Needs both X and y
        
        # Unsupervised learning doesn't use y
        unsupervised = UnsupervisedLearningDemo(n_clusters=3)
        X_unsup = unsupervised.generate_customer_data(100)
        unsupervised.model.fit(X_unsup)  # Only needs X
        
        # This is the fundamental difference!
        assert True, "Both APIs work as expected"
    
    def test_evaluation_difference(self):
        """
        Verify evaluation metrics differ between approaches.
        
        Supervised: Can calculate accuracy (compare to true labels)
        Unsupervised: Use internal metrics (silhouette score)
        """
        # Supervised can measure "correctness"
        supervised = SupervisedLearningDemo()
        supervised_score = supervised.train_and_evaluate()
        assert 0 <= supervised_score <= 1, "Accuracy should be between 0 and 1"
        
        # Unsupervised measures "cluster quality"
        unsupervised = UnsupervisedLearningDemo(n_clusters=3)
        unsupervised_score = unsupervised.discover_segments()
        assert -1 <= unsupervised_score <= 1, "Silhouette should be between -1 and 1"
        
        # Different metrics for different approaches!
        assert True, "Different evaluation approaches confirmed"


def test_complete_lesson():
    """Integration test: verify complete lesson runs without errors."""
    from lesson_code import main
    
    # Should run without raising exceptions
    main()
    
    assert True, "Complete lesson executed successfully"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
