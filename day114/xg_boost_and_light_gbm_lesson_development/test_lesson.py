"""
Day 114: XGBoost and LightGBM - Comprehensive Test Suite
Tests for production fraud detection implementation
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import os


@pytest.fixture
def sample_data():
    """Generate sample fraud detection data"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def xgb_model(sample_data):
    """Train XGBoost model"""
    X_train, X_test, y_train, y_test = sample_data
    
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=50,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def lgb_model(sample_data):
    """Train LightGBM model"""
    X_train, X_test, y_train, y_test = sample_data
    
    model = lgb.LGBMClassifier(
        num_leaves=15,
        learning_rate=0.1,
        n_estimators=50,
        is_unbalance=True,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    return model


class TestXGBoost:
    """Test suite for XGBoost implementation"""
    
    def test_model_creation(self):
        """Test XGBoost model can be created"""
        model = xgb.XGBClassifier()
        assert model is not None
    
    def test_model_training(self, sample_data):
        """Test XGBoost model can be trained"""
        X_train, X_test, y_train, y_test = sample_data
        
        model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X_train.shape[1]
    
    def test_predictions(self, xgb_model, sample_data):
        """Test XGBoost predictions"""
        _, X_test, _, y_test = sample_data
        
        predictions = xgb_model.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
    
    def test_probability_predictions(self, xgb_model, sample_data):
        """Test XGBoost probability predictions"""
        _, X_test, _, _ = sample_data
        
        probas = xgb_model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert (probas >= 0).all() and (probas <= 1).all()
    
    def test_feature_importance(self, xgb_model, sample_data):
        """Test feature importance extraction"""
        X_train, _, _, _ = sample_data
        
        importances = xgb_model.feature_importances_
        
        assert len(importances) == X_train.shape[1]
        assert (importances >= 0).all()
        assert importances.sum() > 0
    
    def test_scale_pos_weight(self, sample_data):
        """Test class imbalance handling"""
        X_train, X_test, y_train, y_test = sample_data
        
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Model should predict some positive cases despite imbalance
        predictions = model.predict(X_test)
        assert predictions.sum() > 0
    
    def test_early_stopping(self, sample_data):
        """Test early stopping functionality"""
        X_train, X_test, y_train, y_test = sample_data
        
        model = xgb.XGBClassifier(
            n_estimators=1000,
            early_stopping_rounds=10,
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Should stop before 1000 iterations
        assert model.best_iteration < 1000
    
    def test_tree_method_hist(self, sample_data):
        """Test histogram-based tree method"""
        X_train, _, y_train, _ = sample_data
        
        model = xgb.XGBClassifier(
            tree_method='hist',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')


class TestLightGBM:
    """Test suite for LightGBM implementation"""
    
    def test_model_creation(self):
        """Test LightGBM model can be created"""
        model = lgb.LGBMClassifier(verbose=-1)
        assert model is not None
    
    def test_model_training(self, sample_data):
        """Test LightGBM model can be trained"""
        X_train, X_test, y_train, y_test = sample_data
        
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X_train.shape[1]
    
    def test_predictions(self, lgb_model, sample_data):
        """Test LightGBM predictions"""
        _, X_test, _, y_test = sample_data
        
        predictions = lgb_model.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
    
    def test_probability_predictions(self, lgb_model, sample_data):
        """Test LightGBM probability predictions"""
        _, X_test, _, _ = sample_data
        
        probas = lgb_model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert (probas >= 0).all() and (probas <= 1).all()
    
    def test_feature_importance(self, lgb_model, sample_data):
        """Test feature importance extraction"""
        X_train, _, _, _ = sample_data
        
        importances = lgb_model.feature_importances_
        
        assert len(importances) == X_train.shape[1]
        assert (importances >= 0).all()
        assert importances.sum() > 0
    
    def test_is_unbalance(self, sample_data):
        """Test class imbalance handling"""
        X_train, X_test, y_train, y_test = sample_data
        
        model = lgb.LGBMClassifier(
            is_unbalance=True,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Model should predict some positive cases despite imbalance
        predictions = model.predict(X_test)
        assert predictions.sum() > 0
    
    def test_leaf_wise_growth(self, sample_data):
        """Test leaf-wise tree growth"""
        X_train, _, y_train, _ = sample_data
        
        model = lgb.LGBMClassifier(
            num_leaves=31,
            boosting_type='gbdt',
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')
    
    def test_categorical_features(self, sample_data):
        """Test categorical feature handling"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Add a categorical feature
        X_train_cat = np.column_stack([
            X_train,
            np.random.randint(0, 5, len(X_train))
        ])
        X_test_cat = np.column_stack([
            X_test,
            np.random.randint(0, 5, len(X_test))
        ])
        
        model = lgb.LGBMClassifier(
            categorical_feature=[X_train.shape[1]],
            random_state=42,
            verbose=-1
        )
        model.fit(X_train_cat, y_train)
        predictions = model.predict(X_test_cat)
        
        assert len(predictions) == len(y_test)


class TestModelComparison:
    """Test suite for comparing XGBoost and LightGBM"""
    
    def test_similar_performance(self, xgb_model, lgb_model, sample_data):
        """Test both models achieve similar performance"""
        _, X_test, _, y_test = sample_data
        
        xgb_score = xgb_model.score(X_test, y_test)
        lgb_score = lgb_model.score(X_test, y_test)
        
        # Both should achieve reasonable accuracy
        assert xgb_score > 0.7
        assert lgb_score > 0.7
        
        # Scores should be similar (within 10%)
        assert abs(xgb_score - lgb_score) < 0.1
    
    def test_inference_speed(self, xgb_model, lgb_model, sample_data):
        """Test inference speed comparison"""
        _, X_test, _, _ = sample_data
        
        import time
        
        # Warmup
        xgb_model.predict(X_test[:10])
        lgb_model.predict(X_test[:10])
        
        # Benchmark XGBoost
        start = time.perf_counter()
        for _ in range(10):
            xgb_model.predict(X_test)
        xgb_time = time.perf_counter() - start
        
        # Benchmark LightGBM
        start = time.perf_counter()
        for _ in range(10):
            lgb_model.predict(X_test)
        lgb_time = time.perf_counter() - start
        
        # Both should complete quickly
        assert xgb_time < 1.0
        assert lgb_time < 1.0
    
    def test_feature_importance_correlation(self, xgb_model, lgb_model):
        """Test feature importance correlation between models"""
        xgb_importance = xgb_model.feature_importances_
        lgb_importance = lgb_model.feature_importances_
        
        # Normalize importances
        xgb_norm = xgb_importance / xgb_importance.sum()
        lgb_norm = lgb_importance / lgb_importance.sum()
        
        # Calculate correlation
        correlation = np.corrcoef(xgb_norm, lgb_norm)[0, 1]
        
        # Should have positive correlation
        assert correlation > 0.5


class TestProductionReadiness:
    """Test suite for production deployment readiness"""
    
    def test_batch_prediction(self, xgb_model, sample_data):
        """Test batch prediction capability"""
        _, X_test, _, _ = sample_data
        
        # Test different batch sizes
        for batch_size in [1, 10, 100]:
            predictions = xgb_model.predict(X_test[:batch_size])
            assert len(predictions) == batch_size
    
    def test_missing_value_handling(self, sample_data):
        """Test handling of missing values"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Introduce missing values
        X_train_missing = X_train.copy()
        X_test_missing = X_test.copy()
        
        mask_train = np.random.random(X_train.shape) < 0.1
        mask_test = np.random.random(X_test.shape) < 0.1
        
        X_train_missing[mask_train] = np.nan
        X_test_missing[mask_test] = np.nan
        
        # XGBoost should handle missing values natively
        model = xgb.XGBClassifier(random_state=42)
        model.fit(X_train_missing, y_train)
        predictions = model.predict(X_test_missing)
        
        assert len(predictions) == len(y_test)
        assert not np.isnan(predictions).any()
    
    def test_model_serialization(self, xgb_model, lgb_model, tmp_path):
        """Test model saving and loading"""
        import joblib
        
        # Save models
        xgb_path = tmp_path / "xgb_model.pkl"
        lgb_path = tmp_path / "lgb_model.pkl"
        
        joblib.dump(xgb_model, xgb_path)
        joblib.dump(lgb_model, lgb_path)
        
        # Load models
        xgb_loaded = joblib.load(xgb_path)
        lgb_loaded = joblib.load(lgb_path)
        
        assert xgb_loaded is not None
        assert lgb_loaded is not None
    
    def test_consistent_predictions(self, xgb_model, sample_data):
        """Test prediction consistency"""
        _, X_test, _, _ = sample_data
        
        # Make predictions multiple times
        pred1 = xgb_model.predict(X_test)
        pred2 = xgb_model.predict(X_test)
        pred3 = xgb_model.predict(X_test)
        
        # Predictions should be identical
        assert np.array_equal(pred1, pred2)
        assert np.array_equal(pred2, pred3)
    
    def test_memory_efficiency(self, sample_data):
        """Test memory-efficient training"""
        X_train, _, y_train, _ = sample_data
        
        # Train with subsample to reduce memory
        model = lgb.LGBMClassifier(
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')


def test_data_generation():
    """Test synthetic fraud data generation"""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42
    )
    
    assert X.shape == (1000, 20)
    assert len(y) == 1000
    assert y.sum() > 0  # Has some positive cases
    assert y.sum() < len(y) * 0.1  # Imbalanced


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
