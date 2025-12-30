"""
Day 59: Decision Trees with Scikit-learn - Test Suite

Comprehensive tests validating decision tree implementation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys
import os

# Import lesson code
from lesson_code import CustomerChurnPredictor, compare_with_baseline


class TestCustomerChurnPredictor:
    """Test suite for churn prediction system."""
    
    @pytest.fixture
    def predictor(self):
        """Create predictor instance for testing."""
        return CustomerChurnPredictor(max_depth=10, min_samples_split=50)
    
    @pytest.fixture
    def sample_data(self, predictor):
        """Generate sample data for testing."""
        X, y = predictor.generate_synthetic_data(n_samples=1000)
        return X, y
    
    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.max_depth == 10
        assert predictor.min_samples_split == 50
        assert predictor.model is None
        
    def test_data_generation(self, predictor):
        """Test synthetic data generation."""
        X, y = predictor.generate_synthetic_data(n_samples=1000)
        
        # Check dimensions
        assert X.shape[0] == 1000
        assert X.shape[1] == 10  # 10 features
        assert len(y) == 1000
        
        # Check feature names
        expected_features = [
            'monthly_hours', 'login_frequency', 'content_diversity',
            'completion_rate', 'support_tickets', 'account_age_months',
            'subscription_tier', 'payment_failures', 'days_since_last_login',
            'device_count'
        ]
        assert predictor.feature_names == expected_features
        
        # Check label distribution (binary)
        assert set(y.unique()) == {0, 1}
        
        # Check for imbalance (churn should be minority class)
        churn_rate = y.mean()
        assert 0.1 <= churn_rate <= 0.3  # Realistic churn rate
        
    def test_data_quality(self, sample_data):
        """Test data quality and distributions."""
        X, y = sample_data
        
        # Check for missing values
        assert not X.isnull().any().any()
        assert not y.isnull().any()
        
        # Check feature ranges
        assert (X['monthly_hours'] >= 0).all()
        assert (X['login_frequency'] >= 0).all()
        assert (X['completion_rate'] >= 0).all() and (X['completion_rate'] <= 100).all()
        assert (X['subscription_tier'].isin([1, 2, 3])).all()
        
    def test_training(self, predictor, sample_data):
        """Test model training."""
        X, y = sample_data
        metrics = predictor.train(X, y, use_grid_search=False)
        
        # Check model is trained
        assert predictor.model is not None
        assert isinstance(predictor.model, DecisionTreeClassifier)
        
        # Check metrics exist
        assert 'cv_mean' in metrics
        assert 'cv_std' in metrics
        assert 'test_accuracy' in metrics
        assert 'test_roc_auc' in metrics
        
        # Check performance is reasonable
        assert metrics['cv_mean'] > 0.5  # Better than random
        assert metrics['test_roc_auc'] > 0.5
        
    def test_model_parameters(self, predictor, sample_data):
        """Test model uses correct parameters."""
        X, y = sample_data
        predictor.train(X, y, use_grid_search=False)
        
        assert predictor.model.max_depth == 10
        assert predictor.model.min_samples_split == 50
        assert predictor.model.class_weight == 'balanced'
        
    def test_feature_importance(self, predictor, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        predictor.train(X, y)
        
        importance_df = predictor.get_feature_importance()
        
        # Check structure
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == 10  # All features
        
        # Check importances sum to 1
        assert abs(importance_df['importance'].sum() - 1.0) < 0.01
        
        # Check sorted descending
        assert (importance_df['importance'].diff().dropna() <= 0).all()
        
    def test_feature_importance_without_training(self, predictor):
        """Test that feature importance fails before training."""
        with pytest.raises(ValueError, match="Model not trained yet"):
            predictor.get_feature_importance()
            
    def test_predictions(self, predictor, sample_data):
        """Test model predictions."""
        X, y = sample_data
        predictor.train(X, y)
        
        # Test predictions on new data
        X_new, _ = predictor.generate_synthetic_data(n_samples=100)
        predictions = predictor.model.predict(X_new)
        
        # Check predictions are binary
        assert set(predictions) <= {0, 1}
        assert len(predictions) == 100
        
        # Check prediction probabilities
        probabilities = predictor.model.predict_proba(X_new)
        assert probabilities.shape == (100, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
    def test_cross_validation(self, predictor, sample_data):
        """Test cross-validation metrics."""
        X, y = sample_data
        metrics = predictor.train(X, y)
        
        # Check CV scores are reasonable
        assert 0.5 <= metrics['cv_mean'] <= 1.0
        assert metrics['cv_std'] >= 0
        assert metrics['cv_std'] < 0.5  # Not too variable
        
    def test_imbalanced_handling(self, predictor, sample_data):
        """Test handling of imbalanced dataset."""
        X, y = sample_data
        metrics = predictor.train(X, y)
        
        # Check that model doesn't just predict majority class
        y_pred = metrics['y_pred']
        unique_predictions = set(y_pred)
        assert len(unique_predictions) == 2  # Predicts both classes
        
        # Check ROC-AUC (better metric for imbalanced data)
        assert metrics['test_roc_auc'] > 0.6  # Meaningful performance
        
    def test_confusion_matrix(self, predictor, sample_data):
        """Test confusion matrix generation."""
        X, y = sample_data
        metrics = predictor.train(X, y)
        
        cm = metrics['confusion_matrix']
        
        # Check shape (2x2 for binary classification)
        assert cm.shape == (2, 2)
        
        # Check all values are non-negative
        assert (cm >= 0).all()
        
        # Check sum equals test set size
        assert cm.sum() == len(metrics['y_test'])
        
    def test_stratified_split(self, predictor, sample_data):
        """Test that train/test split maintains class distribution."""
        X, y = sample_data
        metrics = predictor.train(X, y)
        
        # Get churn rates
        overall_churn = y.mean()
        test_churn = metrics['y_test'].mean()
        
        # Check they're similar (stratified split)
        assert abs(overall_churn - test_churn) < 0.05
        
    def test_grid_search(self, predictor, sample_data):
        """Test grid search hyperparameter tuning."""
        X, y = sample_data
        metrics = predictor.train(X, y, use_grid_search=True)
        
        # Check best parameters were found
        assert predictor.best_params is not None
        assert 'max_depth' in predictor.best_params
        assert 'min_samples_split' in predictor.best_params
        
    def test_reproducibility(self, predictor):
        """Test that results are reproducible with same random seed."""
        X1, y1 = predictor.generate_synthetic_data(n_samples=500)
        
        predictor2 = CustomerChurnPredictor()
        X2, y2 = predictor2.generate_synthetic_data(n_samples=500)
        
        # Check data is identical
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)


class TestBaselineComparison:
    """Test baseline model comparison."""
    
    def test_baseline_comparison(self):
        """Test comparison function runs without errors."""
        predictor = CustomerChurnPredictor()
        X, y = predictor.generate_synthetic_data(n_samples=500)
        
        # Should run without errors
        compare_with_baseline(X, y)
        
    def test_decision_tree_outperforms_random(self):
        """Test that decision tree outperforms random baseline."""
        from sklearn.dummy import DummyClassifier
        
        predictor = CustomerChurnPredictor()
        X, y = predictor.generate_synthetic_data(n_samples=1000)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train random baseline
        baseline = DummyClassifier(strategy='stratified', random_state=42)
        baseline.fit(X_train, y_train)
        baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
        
        # Train decision tree
        dt = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=50,
            class_weight='balanced',
            random_state=42
        )
        dt.fit(X_train, y_train)
        dt_auc = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
        
        # Decision tree should significantly outperform random
        assert dt_auc > baseline_auc + 0.1


def test_scikit_learn_integration():
    """Test integration with scikit-learn API."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    predictor = CustomerChurnPredictor()
    X, y = predictor.generate_synthetic_data(n_samples=500)
    
    # Test pipeline integration
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    
    assert score > 0.5  # Better than random


def test_production_metrics():
    """Test that all production metrics are calculated."""
    predictor = CustomerChurnPredictor()
    X, y = predictor.generate_synthetic_data(n_samples=500)
    metrics = predictor.train(X, y)
    
    required_metrics = [
        'cv_mean', 'cv_std', 'test_accuracy', 'test_roc_auc',
        'confusion_matrix', 'classification_report'
    ]
    
    for metric in required_metrics:
        assert metric in metrics, f"Missing required metric: {metric}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
