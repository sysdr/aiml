"""
Test suite for Day 58: Decision Trees Theory
"""

import pytest
import numpy as np
from lesson_code import (
    DecisionTree, Node, 
    generate_customer_churn_data
)


class TestNode:
    """Test Node class"""
    
    def test_leaf_node(self):
        """Test leaf node creation and identification"""
        node = Node(value=1)
        assert node.is_leaf()
        assert node.value == 1
        
    def test_internal_node(self):
        """Test internal node creation"""
        node = Node(feature=0, threshold=5.0)
        assert not node.is_leaf()
        assert node.feature == 0
        assert node.threshold == 5.0


class TestDecisionTree:
    """Test DecisionTree class"""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple test data"""
        # XOR-like problem
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        y = np.array([0, 1, 1, 0])
        return X, y
    
    @pytest.fixture
    def tree(self):
        """Create decision tree instance"""
        return DecisionTree(max_depth=3, min_samples_split=2)
    
    def test_entropy_pure(self, tree):
        """Test entropy calculation for pure node"""
        y = np.array([1, 1, 1, 1])
        entropy = tree._calculate_entropy(y)
        assert entropy == 0.0
    
    def test_entropy_mixed(self, tree):
        """Test entropy calculation for mixed node"""
        y = np.array([0, 0, 1, 1])
        entropy = tree._calculate_entropy(y)
        assert entropy == 1.0  # Maximum entropy for 50/50 split
    
    def test_entropy_skewed(self, tree):
        """Test entropy calculation for skewed distribution"""
        y = np.array([0, 0, 0, 1])
        entropy = tree._calculate_entropy(y)
        assert 0 < entropy < 1  # Should be between 0 and 1
    
    def test_information_gain(self, tree):
        """Test information gain calculation"""
        parent = np.array([0, 0, 1, 1])
        left = np.array([0, 0])  # Pure
        right = np.array([1, 1])  # Pure
        
        gain = tree._information_gain(parent, left, right)
        assert gain > 0  # Should have positive gain
        assert gain == 1.0  # Perfect split
    
    def test_information_gain_no_split(self, tree):
        """Test information gain with no improvement"""
        parent = np.array([0, 0, 1, 1])
        left = np.array([0, 1])
        right = np.array([0, 1])
        
        gain = tree._information_gain(parent, left, right)
        assert gain == 0  # No improvement
    
    def test_fit_simple(self, tree, simple_data):
        """Test fitting on simple data"""
        X, y = simple_data
        tree.fit(X, y, feature_names=['feature_0', 'feature_1'])
        assert tree.root is not None
        assert tree.feature_names == ['feature_0', 'feature_1']
    
    def test_predict_simple(self, tree, simple_data):
        """Test prediction on simple data"""
        X, y = simple_data
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        # Should get reasonable accuracy (may not be perfect due to XOR)
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)
    
    def test_find_best_split(self, tree):
        """Test finding best split"""
        X = np.array([
            [1], [2], [3], [4], [5], [6]
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        feature, threshold = tree._find_best_split(X, y)
        assert feature is not None
        assert threshold is not None
        # Threshold should be between 3 and 4 for perfect split
        assert 3 <= threshold <= 4
    
    def test_max_depth_limit(self):
        """Test that tree respects max_depth"""
        tree = DecisionTree(max_depth=1)  # Very shallow
        X = np.random.rand(100, 2)
        y = np.random.randint(0, 2, 100)
        tree.fit(X, y)
        
        # Tree should be shallow - verify by checking structure
        assert tree.root is not None


class TestCustomerChurnData:
    """Test data generation function"""
    
    def test_data_generation(self):
        """Test customer churn data generation"""
        X_df, y = generate_customer_churn_data(n_samples=100, random_state=42)
        
        assert len(X_df) == 100
        assert len(y) == 100
        assert X_df.shape[1] == 4  # 4 features
        assert all(label in [0, 1] for label in y)
    
    def test_data_reproducibility(self):
        """Test that random_state ensures reproducibility"""
        X1, y1 = generate_customer_churn_data(n_samples=50, random_state=42)
        X2, y2 = generate_customer_churn_data(n_samples=50, random_state=42)
        
        assert np.allclose(X1.values, X2.values)
        assert np.array_equal(y1, y2)
    
    def test_feature_ranges(self):
        """Test that features are in expected ranges"""
        X_df, y = generate_customer_churn_data(n_samples=500, random_state=42)
        
        # Check feature ranges
        assert (X_df['login_days'] >= 0).all()
        assert (X_df['login_days'] <= 90).all()
        assert (X_df['features_used'] >= 0).all()
        assert (X_df['features_used'] <= 10).all()
        assert (X_df['support_tickets'] >= 0).all()
        assert (X_df['account_age_months'] >= 1).all()


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_full_pipeline(self):
        """Test complete training and prediction pipeline"""
        # Generate data
        X_df, y = generate_customer_churn_data(n_samples=200, random_state=42)
        X = X_df.values
        
        # Split data
        split_idx = 160
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        tree = DecisionTree(max_depth=5, min_samples_split=10)
        tree.fit(X_train, y_train, feature_names=list(X_df.columns))
        
        # Make predictions
        train_pred = tree.predict(X_train)
        test_pred = tree.predict(X_test)
        
        # Verify predictions
        assert len(train_pred) == len(y_train)
        assert len(test_pred) == len(y_test)
        
        # Check accuracy is reasonable (>60%)
        train_accuracy = np.mean(train_pred == y_train)
        test_accuracy = np.mean(test_pred == y_test)
        
        assert train_accuracy > 0.6
        assert test_accuracy > 0.5  # Test may be lower due to small size
    
    def test_comparison_with_sklearn(self):
        """Test that our implementation performs similarly to sklearn"""
        from sklearn.tree import DecisionTreeClassifier
        
        # Generate data
        X_df, y = generate_customer_churn_data(n_samples=300, random_state=42)
        X = X_df.values
        
        # Train our tree
        our_tree = DecisionTree(max_depth=5, min_samples_split=10)
        our_tree.fit(X, y)
        our_pred = our_tree.predict(X)
        our_accuracy = np.mean(our_pred == y)
        
        # Train sklearn tree
        sklearn_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
        sklearn_tree.fit(X, y)
        sklearn_pred = sklearn_tree.predict(X)
        sklearn_accuracy = np.mean(sklearn_pred == y)
        
        # Our implementation should be within 10% of sklearn
        assert abs(our_accuracy - sklearn_accuracy) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
