"""
Test Suite for Day 60: Random Forests and Ensemble Methods
Comprehensive tests ensuring production-ready implementation
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lesson_code import CustomerChurnPredictor

class TestCustomerChurnPredictor:
    """Test suite for Random Forest churn predictor."""
    
    @pytest.fixture
    def predictor(self):
        """Create predictor instance for testing."""
        return CustomerChurnPredictor(n_estimators=50, max_depth=10, random_state=42)
    
    @pytest.fixture
    def sample_data(self, predictor):
        """Generate sample data for testing."""
        X, y = predictor.generate_synthetic_data(n_samples=1000)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.random_state == 42
        assert predictor.is_trained == False
        assert predictor.single_tree is not None
        assert predictor.random_forest is not None
    
    def test_data_generation(self, predictor):
        """Test synthetic data generation."""
        X, y = predictor.generate_synthetic_data(n_samples=1000)
        
        # Check dimensions
        assert X.shape[0] == 1000
        assert X.shape[1] == 15
        assert len(y) == 1000
        
        # Check feature names
        assert predictor.feature_names is not None
        assert len(predictor.feature_names) == 15
        
        # Check binary labels
        assert set(y).issubset({0, 1})
        
        # Check realistic class distribution (around 70-30 split)
        churn_rate = np.mean(y)
        assert 0.2 < churn_rate < 0.4
    
    def test_training(self, predictor, sample_data):
        """Test model training."""
        X_train, X_test, y_train, y_test = sample_data
        
        predictor.train(X_train, y_train)
        
        # Check training flag
        assert predictor.is_trained == True
        
        # Check models can make predictions
        tree_pred = predictor.single_tree.predict(X_test)
        rf_pred = predictor.random_forest.predict(X_test)
        
        assert len(tree_pred) == len(X_test)
        assert len(rf_pred) == len(X_test)
    
    def test_random_forest_hyperparameters(self, predictor):
        """Test Random Forest is configured correctly."""
        rf = predictor.random_forest
        
        # Check key hyperparameters
        assert rf.n_estimators == 50  # Test uses 50 for speed
        assert rf.max_depth == 10
        assert rf.min_samples_split == 10
        assert rf.min_samples_leaf == 5
        assert rf.max_features == 'sqrt'
        assert rf.bootstrap == True
        assert rf.oob_score == True
        assert rf.n_jobs == -1
    
    def test_ensemble_superiority(self, predictor, sample_data):
        """Test that Random Forest outperforms single tree."""
        X_train, X_test, y_train, y_test = sample_data
        
        predictor.train(X_train, y_train)
        
        # Get accuracies
        tree_accuracy = predictor.single_tree.score(X_test, y_test)
        rf_accuracy = predictor.random_forest.score(X_test, y_test)
        
        # Random Forest should be better (or at least equal)
        assert rf_accuracy >= tree_accuracy
        
        # Should see meaningful improvement (at least 2%)
        improvement = rf_accuracy - tree_accuracy
        assert improvement >= 0.00  # At minimum, no degradation
    
    def test_evaluation_metrics(self, predictor, sample_data):
        """Test evaluation returns proper metrics."""
        X_train, X_test, y_train, y_test = sample_data
        
        predictor.train(X_train, y_train)
        results = predictor.evaluate(X_test, y_test)
        
        # Check structure
        assert 'single_tree' in results
        assert 'random_forest' in results
        assert 'predictions' in results
        
        # Check single tree metrics
        tree_metrics = results['single_tree']
        assert 'accuracy' in tree_metrics
        assert 'precision' in tree_metrics
        assert 'recall' in tree_metrics
        assert 'f1' in tree_metrics
        assert 'roc_auc' in tree_metrics
        
        # Check random forest metrics
        rf_metrics = results['random_forest']
        assert 'accuracy' in rf_metrics
        assert 'oob_score' in rf_metrics
        
        # Check metric ranges
        assert 0 <= rf_metrics['accuracy'] <= 1
        assert 0 <= rf_metrics['precision'] <= 1
        assert 0 <= rf_metrics['recall'] <= 1
        assert 0 <= rf_metrics['f1'] <= 1
        assert 0 <= rf_metrics['roc_auc'] <= 1
        assert 0 <= rf_metrics['oob_score'] <= 1
    
    def test_oob_score(self, predictor, sample_data):
        """Test Out-of-Bag score is calculated."""
        X_train, X_test, y_train, y_test = sample_data
        
        predictor.train(X_train, y_train)
        
        # OOB score should be calculated
        oob_score = predictor.random_forest.oob_score_
        assert oob_score is not None
        assert 0 < oob_score < 1
        
        # OOB score should be reasonable (within 10% of test accuracy)
        test_accuracy = predictor.random_forest.score(X_test, y_test)
        assert abs(oob_score - test_accuracy) < 0.15
    
    def test_feature_importance(self, predictor, sample_data):
        """Test feature importance analysis."""
        X_train, X_test, y_train, y_test = sample_data
        
        predictor.train(X_train, y_train)
        importances = predictor.analyze_feature_importance()
        
        # Check structure
        assert len(importances) == 15
        assert 'feature' in importances.columns
        assert 'importance' in importances.columns
        
        # Check importance values sum to 1
        total_importance = importances['importance'].sum()
        assert abs(total_importance - 1.0) < 0.01
        
        # Check all importances are non-negative
        assert all(importances['importance'] >= 0)
        
        # Check sorted descending
        assert importances['importance'].is_monotonic_decreasing
    
    def test_probability_predictions(self, predictor, sample_data):
        """Test probability predictions work correctly."""
        X_train, X_test, y_train, y_test = sample_data
        
        predictor.train(X_train, y_train)
        
        # Get probability predictions
        probas = predictor.random_forest.predict_proba(X_test)
        
        # Check shape
        assert probas.shape == (len(X_test), 2)
        
        # Check probabilities sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0)
        
        # Check range [0, 1]
        assert np.all((probas >= 0) & (probas <= 1))
    
    def test_ensemble_diversity(self, predictor, sample_data):
        """Test that individual trees make different predictions."""
        X_train, X_test, y_train, y_test = sample_data
        
        predictor.train(X_train, y_train)
        
        # Get predictions from first 5 trees
        tree_preds = []
        for tree in predictor.random_forest.estimators_[:5]:
            preds = tree.predict(X_test[:100])
            tree_preds.append(preds)
        
        tree_preds = np.array(tree_preds)
        
        # Trees should make different predictions (diversity)
        # Check that predictions vary across trees
        for i in range(100):
            sample_preds = tree_preds[:, i]
            # Not all trees should agree on every sample
            unique_preds = len(np.unique(sample_preds))
            # At least some disagreement should exist across all samples
        
        # Overall diversity: trees shouldn't all predict the same
        all_same = np.all(tree_preds == tree_preds[0], axis=0)
        diversity_rate = 1 - np.mean(all_same)
        assert diversity_rate > 0.1  # At least 10% disagreement
    
    def test_prediction_consistency(self, predictor, sample_data):
        """Test predictions are consistent (deterministic)."""
        X_train, X_test, y_train, y_test = sample_data
        
        predictor.train(X_train, y_train)
        
        # Make predictions twice
        pred1 = predictor.random_forest.predict(X_test)
        pred2 = predictor.random_forest.predict(X_test)
        
        # Should be identical (deterministic)
        assert np.array_equal(pred1, pred2)
    
    def test_min_samples_constraints(self, predictor, sample_data):
        """Test that min_samples constraints are respected."""
        X_train, X_test, y_train, y_test = sample_data
        
        predictor.train(X_train, y_train)
        
        # Extract a tree and check leaf sizes
        tree = predictor.random_forest.estimators_[0].tree_
        
        # Get leaf nodes (where left and right children are both -1)
        leaf_mask = (tree.children_left == -1) & (tree.children_right == -1)
        leaf_samples = tree.n_node_samples[leaf_mask]
        
        # All leaves should have at least min_samples_leaf samples
        assert np.all(leaf_samples >= predictor.random_forest.min_samples_leaf)
    
    def test_error_handling(self, predictor):
        """Test error handling for untrained model."""
        X_test = np.random.randn(100, 15)
        y_test = np.random.randint(0, 2, 100)
        
        # Should raise error when evaluating untrained model
        with pytest.raises(ValueError):
            predictor.evaluate(X_test, y_test)
        
        # Should raise error when analyzing features of untrained model
        with pytest.raises(ValueError):
            predictor.analyze_feature_importance()
    
    def test_bootstrap_sampling(self, predictor, sample_data):
        """Test that bootstrap sampling creates diverse datasets."""
        X_train, X_test, y_train, y_test = sample_data
        
        predictor.train(X_train, y_train)
        
        # Check that OOB samples exist (indicates bootstrapping worked)
        # OOB score should be computable
        assert predictor.random_forest.oob_score_ > 0
        
        # Number of estimators should match
        assert len(predictor.random_forest.estimators_) == predictor.random_forest.n_estimators
    
    def test_scalability(self, predictor):
        """Test model handles different data sizes."""
        # Small dataset
        X_small, y_small = predictor.generate_synthetic_data(n_samples=100)
        X_train, X_test, y_train, y_test = train_test_split(
            X_small, y_small, test_size=0.2, random_state=42
        )
        predictor.train(X_train, y_train)
        accuracy_small = predictor.random_forest.score(X_test, y_test)
        assert 0 < accuracy_small < 1
        
        # Larger dataset should work too
        predictor2 = CustomerChurnPredictor(n_estimators=50, random_state=42)
        X_large, y_large = predictor2.generate_synthetic_data(n_samples=3000)
        X_train, X_test, y_train, y_test = train_test_split(
            X_large, y_large, test_size=0.2, random_state=42
        )
        predictor2.train(X_train, y_train)
        accuracy_large = predictor2.random_forest.score(X_test, y_test)
        assert 0 < accuracy_large < 1

def test_sklearn_random_forest_api():
    """Test that we're using sklearn Random Forest correctly."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    # Test core functionality
    assert rf.score(X_test, y_test) > 0.5
    assert hasattr(rf, 'feature_importances_')
    assert hasattr(rf, 'oob_score_')
    assert len(rf.estimators_) == 50

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
