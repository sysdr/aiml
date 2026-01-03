"""
Test suite for Day 62: KNN Theory
Tests core KNN functionality
"""

import pytest
import numpy as np
from lesson_code import KNNClassifier

@pytest.fixture
def simple_dataset():
    """Create simple 2D dataset for testing"""
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [10, 10],
        [11, 11],
        [12, 12]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_knn_initialization():
    """Test KNN classifier initialization"""
    knn = KNNClassifier(k=3)
    assert knn.k == 3
    assert knn.distance_metric == 'euclidean'


def test_knn_fit(simple_dataset):
    """Test KNN fit method"""
    X, y = simple_dataset
    knn = KNNClassifier(k=3)
    knn.fit(X, y)
    
    assert knn.X_train is not None
    assert knn.y_train is not None
    assert len(knn.X_train) == 6


def test_euclidean_distance():
    """Test Euclidean distance calculation"""
    knn = KNNClassifier(k=3, distance_metric='euclidean')
    knn.X_train = np.array([[0, 0]])
    knn.y_train = np.array([0])
    
    dist = knn._calculate_distance(np.array([0, 0]), np.array([3, 4]))
    assert np.isclose(dist, 5.0)


def test_manhattan_distance():
    """Test Manhattan distance calculation"""
    knn = KNNClassifier(k=3, distance_metric='manhattan')
    knn.X_train = np.array([[0, 0]])
    knn.y_train = np.array([0])
    
    dist = knn._calculate_distance(np.array([0, 0]), np.array([3, 4]))
    assert np.isclose(dist, 7.0)


def test_knn_prediction(simple_dataset):
    """Test KNN prediction"""
    X, y = simple_dataset
    knn = KNNClassifier(k=3)
    knn.fit(X, y)
    
    # Test point close to class 0
    pred = knn.predict_single(np.array([2.5, 2.5]))
    assert pred == 0
    
    # Test point close to class 1
    pred = knn.predict_single(np.array([11, 11]))
    assert pred == 1


def test_knn_batch_prediction(simple_dataset):
    """Test batch prediction"""
    X, y = simple_dataset
    knn = KNNClassifier(k=3)
    knn.fit(X, y)
    
    X_test = np.array([[2, 2], [11, 11]])
    predictions = knn.predict(X_test)
    
    assert len(predictions) == 2
    assert predictions[0] == 0
    assert predictions[1] == 1


def test_knn_score(simple_dataset):
    """Test KNN scoring"""
    X, y = simple_dataset
    knn = KNNClassifier(k=3)
    knn.fit(X, y)
    
    accuracy = knn.score(X, y)
    assert 0.0 <= accuracy <= 1.0
    assert accuracy > 0.8  # Should get high accuracy on simple dataset


def test_different_k_values(simple_dataset):
    """Test KNN with different K values"""
    X, y = simple_dataset
    
    for k in [1, 3, 5]:
        knn = KNNClassifier(k=k)
        knn.fit(X, y)
        accuracy = knn.score(X, y)
        assert accuracy > 0.5


def test_k_equals_one(simple_dataset):
    """Test K=1 (nearest neighbor only)"""
    X, y = simple_dataset
    knn = KNNClassifier(k=1)
    knn.fit(X, y)
    
    # Should perfectly predict training data with K=1
    accuracy = knn.score(X, y)
    assert accuracy == 1.0


def test_get_neighbors(simple_dataset):
    """Test neighbor finding"""
    X, y = simple_dataset
    knn = KNNClassifier(k=3)
    knn.fit(X, y)
    
    neighbors = knn._get_neighbors(np.array([2, 2]))
    assert len(neighbors) == 3
    assert all(isinstance(idx, (int, np.integer)) for idx in neighbors)


def test_invalid_distance_metric():
    """Test invalid distance metric"""
    knn = KNNClassifier(k=3, distance_metric='invalid')
    knn.X_train = np.array([[0, 0]])
    knn.y_train = np.array([0])
    
    with pytest.raises(ValueError):
        knn._calculate_distance(np.array([0, 0]), np.array([1, 1]))


def test_knn_with_multiclass():
    """Test KNN with 3 classes"""
    X = np.array([
        [1, 1], [2, 2],
        [5, 5], [6, 6],
        [10, 10], [11, 11]
    ])
    y = np.array([0, 0, 1, 1, 2, 2])
    
    knn = KNNClassifier(k=3)
    knn.fit(X, y)
    
    # Test predictions for each class
    assert knn.predict_single(np.array([1.5, 1.5])) == 0
    assert knn.predict_single(np.array([5.5, 5.5])) == 1
    assert knn.predict_single(np.array([10.5, 10.5])) == 2


def test_knn_higher_dimensions():
    """Test KNN with more than 2 features"""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    knn = KNNClassifier(k=5)
    knn.fit(X, y)
    
    predictions = knn.predict(X[:10])
    assert len(predictions) == 10


def test_consistency():
    """Test prediction consistency"""
    X = np.array([[1, 1], [2, 2], [10, 10], [11, 11]])
    y = np.array([0, 0, 1, 1])
    
    knn = KNNClassifier(k=3)
    knn.fit(X, y)
    
    test_point = np.array([1.5, 1.5])
    
    # Same prediction multiple times
    pred1 = knn.predict_single(test_point)
    pred2 = knn.predict_single(test_point)
    pred3 = knn.predict_single(test_point)
    
    assert pred1 == pred2 == pred3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
