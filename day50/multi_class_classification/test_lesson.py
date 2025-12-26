"""
Tests for Day 50: Multi-Class Classification
Verifies both OvR and Softmax implementations
"""

import pytest
import numpy as np
from lesson_code import (
    NewsCategorizerOvR,
    NewsCategorizerSoftmax,
    generate_sample_data
)


class TestMultiClassClassification:
    """Test multi-class classification implementations"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        articles, labels = generate_sample_data(n_samples=200)
        return articles, labels
    
    def test_data_generation(self, sample_data):
        """Test synthetic data generation"""
        articles, labels = sample_data
        
        assert len(articles) == 200
        assert len(labels) == 200
        assert all(isinstance(article, str) for article in articles)
        assert all(label in [0, 1, 2, 3] for label in labels)
        assert len(set(labels)) == 4  # All 4 categories present
    
    def test_ovr_classifier(self, sample_data):
        """Test One-vs-Rest classifier"""
        articles, labels = sample_data
        
        # Split data
        split_idx = int(len(articles) * 0.8)
        X_train, X_test = articles[:split_idx], articles[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        
        # Train classifier
        classifier = NewsCategorizerOvR(max_iter=500)
        classifier.fit(X_train, y_train)
        
        # Make predictions
        predictions = classifier.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1, 2, 3] for pred in predictions)
        
        # Check accuracy is reasonable
        accuracy = np.mean(predictions == y_test)
        assert accuracy > 0.5  # Should do better than random (0.25)
    
    def test_softmax_classifier(self, sample_data):
        """Test Softmax classifier"""
        articles, labels = sample_data
        
        # Split data
        split_idx = int(len(articles) * 0.8)
        X_train, X_test = articles[:split_idx], articles[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        
        # Train classifier
        classifier = NewsCategorizerSoftmax(max_iter=500)
        classifier.fit(X_train, y_train)
        
        # Make predictions
        predictions = classifier.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1, 2, 3] for pred in predictions)
        
        # Check accuracy is reasonable
        accuracy = np.mean(predictions == y_test)
        assert accuracy > 0.5  # Should do better than random (0.25)
    
    def test_probability_outputs(self, sample_data):
        """Test probability outputs for both classifiers"""
        articles, labels = sample_data
        
        # Ensure we use enough data to include all 4 categories
        # Use first 160 samples (40 per category) to ensure all categories are present
        train_size = 160
        assert len(set(labels[:train_size])) == 4, "Training data must include all 4 categories"
        
        # Train both classifiers
        ovr = NewsCategorizerOvR(max_iter=500)
        ovr.fit(articles[:train_size], labels[:train_size])
        
        softmax = NewsCategorizerSoftmax(max_iter=500)
        softmax.fit(articles[:train_size], labels[:train_size])
        
        # Get probability predictions
        test_article = [articles[0]]
        
        probs_ovr = ovr.predict_proba(test_article)
        probs_softmax = softmax.predict_proba(test_article)
        
        # Check shapes
        assert probs_ovr.shape == (1, 4)
        assert probs_softmax.shape == (1, 4)
        
        # Softmax probabilities should sum to 1.0
        assert np.abs(probs_softmax.sum() - 1.0) < 0.001
        
        # All probabilities should be between 0 and 1
        assert np.all(probs_ovr >= 0) and np.all(probs_ovr <= 1)
        assert np.all(probs_softmax >= 0) and np.all(probs_softmax <= 1)
    
    def test_top_features_extraction(self, sample_data):
        """Test feature importance extraction"""
        articles, labels = sample_data
        
        classifier = NewsCategorizerSoftmax(max_iter=500)
        classifier.fit(articles, labels)
        
        # Get top features for each category
        for category_idx in range(4):
            top_features = classifier.get_top_features(category_idx, top_n=5)
            
            assert len(top_features) == 5
            assert all(isinstance(feature, str) for feature, _ in top_features)
            assert all(isinstance(score, (int, float)) for _, score in top_features)
    
    def test_prediction_consistency(self, sample_data):
        """Test that predictions are deterministic"""
        articles, labels = sample_data
        
        classifier = NewsCategorizerSoftmax(max_iter=500)
        classifier.fit(articles[:150], labels[:150])
        
        test_article = [articles[160]]
        
        # Make multiple predictions
        pred1 = classifier.predict(test_article)
        pred2 = classifier.predict(test_article)
        pred3 = classifier.predict(test_article)
        
        assert pred1[0] == pred2[0] == pred3[0]


class TestProductionPatterns:
    """Test production-ready patterns"""
    
    def test_batch_prediction(self):
        """Test efficient batch prediction"""
        articles, labels = generate_sample_data(n_samples=100)
        
        classifier = NewsCategorizerSoftmax(max_iter=500)
        classifier.fit(articles[:80], labels[:80])
        
        # Batch prediction
        test_batch = articles[80:]
        predictions = classifier.predict(test_batch)
        
        assert len(predictions) == len(test_batch)
    
    def test_category_mapping(self):
        """Test category name mapping"""
        classifier = NewsCategorizerOvR()
        
        assert 0 in classifier.categories
        assert 1 in classifier.categories
        assert 2 in classifier.categories
        assert 3 in classifier.categories
        
        assert classifier.categories[0] == 'Technology'
        assert classifier.categories[1] == 'Sports'
        assert classifier.categories[2] == 'Politics'
        assert classifier.categories[3] == 'Entertainment'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
