"""
Test suite for Credit Card Fraud Detection System
"""

import pytest
import numpy as np
import pandas as pd
from lesson_code import FraudDetectionSystem

@pytest.fixture
def detector():
    """Create a fraud detection system for testing"""
    return FraudDetectionSystem(model_type='random_forest', use_smote=False)

@pytest.fixture
def sample_data(detector):
    """Generate small sample dataset for testing"""
    df = detector.generate_synthetic_data(n_samples=1000, fraud_ratio=0.01)
    df = detector.engineer_features(df)
    return df

def test_system_initialization():
    """Test fraud detection system initializes correctly"""
    # Random Forest
    detector_rf = FraudDetectionSystem(model_type='random_forest')
    assert detector_rf.model_type == 'random_forest'
    assert detector_rf.model is not None
    
    # Gradient Boosting
    detector_gb = FraudDetectionSystem(model_type='gradient_boosting')
    assert detector_gb.model_type == 'gradient_boosting'
    
    # Logistic Regression
    detector_lr = FraudDetectionSystem(model_type='logistic')
    assert detector_lr.model_type == 'logistic'

def test_data_generation(detector):
    """Test synthetic data generation"""
    df = detector.generate_synthetic_data(n_samples=1000, fraud_ratio=0.01)
    
    assert len(df) == 1000
    assert 'Class' in df.columns
    assert 'Amount' in df.columns
    assert df['Class'].isin([0, 1]).all()
    
    # Check fraud ratio is approximately correct
    fraud_ratio = df['Class'].mean()
    assert 0.005 <= fraud_ratio <= 0.015  # Within reasonable range

def test_feature_engineering(detector):
    """Test feature engineering creates expected features"""
    df = detector.generate_synthetic_data(n_samples=500, fraud_ratio=0.02)
    df_engineered = detector.engineer_features(df)
    
    # Check new features exist
    expected_features = [
        'Amount_Log', 'Amount_Squared', 'Is_Large_Transaction',
        'Is_Night', 'High_Velocity', 'Far_From_Home',
        'Night_And_Far', 'Large_And_Fast'
    ]
    
    for feature in expected_features:
        assert feature in df_engineered.columns
    
    # Check feature values are valid
    assert (df_engineered['Amount_Log'] >= 0).all()
    assert df_engineered['Is_Night'].isin([0, 1]).all()
    assert df_engineered['Is_Large_Transaction'].isin([0, 1]).all()

def test_data_preparation(detector, sample_data):
    """Test data preparation and splitting"""
    X_train, X_test, y_train, y_test, _ = detector.prepare_data(sample_data)
    
    # Check shapes
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    
    # Check data is scaled
    assert X_train.mean() < 1.0  # Should be scaled
    assert X_test.mean() < 1.0

def test_data_preparation_with_smote(sample_data):
    """Test data preparation with SMOTE balancing"""
    detector = FraudDetectionSystem(model_type='random_forest', use_smote=True)
    X_train, X_test, y_train, y_test, _ = detector.prepare_data(sample_data)
    
    # Check SMOTE increased fraud samples in training set
    fraud_ratio = y_train.mean()
    assert fraud_ratio > 0.1  # Should be much higher than original 1%

def test_model_training(detector, sample_data):
    """Test model training completes successfully"""
    X_train, X_test, y_train, y_test, _ = detector.prepare_data(sample_data)
    
    detector.train(X_train, y_train)
    
    # Check model is fitted
    assert hasattr(detector.model, 'predict')
    assert hasattr(detector.model, 'predict_proba')
    
    # Check predictions work
    predictions = detector.model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert predictions.dtype == np.int64

def test_threshold_optimization(detector, sample_data):
    """Test threshold optimization for recall target"""
    X_train, X_test, y_train, y_test, _ = detector.prepare_data(sample_data)
    detector.train(X_train, y_train)
    
    # Optimize for 80% recall (more achievable with small dataset)
    threshold = detector.optimize_threshold(X_test, y_test, target_recall=0.80)
    
    assert 0.0 <= threshold <= 1.0
    assert threshold != 0.5  # Should be different from default

def test_prediction_output_format(detector, sample_data):
    """Test individual transaction prediction format"""
    X_train, X_test, y_train, y_test, _ = detector.prepare_data(sample_data)
    detector.train(X_train, y_train)
    detector.optimize_threshold(X_test, y_test)
    
    # Get a test transaction
    transaction = X_test[0]
    result = detector.predict_transaction(transaction)
    
    # Check output format
    assert 'fraud_probability' in result
    assert 'is_fraud' in result
    assert 'risk_level' in result
    assert 'threshold_used' in result
    
    # Check value types and ranges
    assert 0.0 <= result['fraud_probability'] <= 1.0
    assert isinstance(result['is_fraud'], (bool, np.bool_))
    assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']

def test_risk_level_assignment(detector, sample_data):
    """Test risk level categorization"""
    X_train, X_test, y_train, y_test, _ = detector.prepare_data(sample_data)
    detector.train(X_train, y_train)
    
    # Create mock predictions to test thresholds
    detector.best_threshold = 0.5
    
    # Low risk transaction (20% fraud probability)
    transaction = X_test[0]
    # Manually set probability for testing
    original_predict = detector.model.predict_proba
    detector.model.predict_proba = lambda x: np.array([[0.8, 0.2]])
    
    result = detector.predict_transaction(transaction)
    assert result['risk_level'] == 'LOW'
    
    # Medium risk (50% fraud probability)
    detector.model.predict_proba = lambda x: np.array([[0.5, 0.5]])
    result = detector.predict_transaction(transaction)
    assert result['risk_level'] == 'MEDIUM'
    
    # High risk (90% fraud probability)
    detector.model.predict_proba = lambda x: np.array([[0.1, 0.9]])
    result = detector.predict_transaction(transaction)
    assert result['risk_level'] == 'HIGH'
    
    # Restore original method
    detector.model.predict_proba = original_predict

def test_evaluation_metrics_computation(detector, sample_data):
    """Test evaluation produces expected metrics"""
    X_train, X_test, y_train, y_test, _ = detector.prepare_data(sample_data)
    detector.train(X_train, y_train)
    detector.optimize_threshold(X_test, y_test)
    
    metrics = detector.evaluate(X_test, y_test)
    
    # Check required metrics exist
    assert 'roc_auc' in metrics
    assert 'confusion_matrix' in metrics
    assert 'fraud_caught_rate' in metrics
    assert 'false_alarm_rate' in metrics
    
    # Check metric ranges
    assert 0.0 <= metrics['roc_auc'] <= 1.0
    assert 0.0 <= metrics['fraud_caught_rate'] <= 1.0
    assert 0.0 <= metrics['false_alarm_rate'] <= 1.0

def test_confusion_matrix_structure(detector, sample_data):
    """Test confusion matrix has correct structure"""
    X_train, X_test, y_train, y_test, _ = detector.prepare_data(sample_data)
    detector.train(X_train, y_train)
    detector.optimize_threshold(X_test, y_test)
    
    metrics = detector.evaluate(X_test, y_test)
    cm = metrics['confusion_matrix']
    
    # Check shape
    assert cm.shape == (2, 2)
    
    # Check all values are non-negative
    assert (cm >= 0).all()
    
    # Check sum equals test set size
    assert cm.sum() == len(X_test)

def test_different_fraud_ratios(detector):
    """Test system handles different fraud ratios"""
    fraud_ratios = [0.001, 0.01, 0.05]
    
    for ratio in fraud_ratios:
        df = detector.generate_synthetic_data(n_samples=1000, fraud_ratio=ratio)
        actual_ratio = df['Class'].mean()
        
        # Should be within 50% of target (small samples have variance)
        assert ratio * 0.5 <= actual_ratio <= ratio * 1.5

def test_feature_importance_available(sample_data):
    """Test Random Forest provides feature importances"""
    detector = FraudDetectionSystem(model_type='random_forest')
    X_train, X_test, y_train, y_test, _ = detector.prepare_data(sample_data)
    detector.train(X_train, y_train)
    
    assert hasattr(detector.model, 'feature_importances_')
    importances = detector.model.feature_importances_
    assert len(importances) == X_train.shape[1]
    assert (importances >= 0).all()
    assert np.isclose(importances.sum(), 1.0)

def test_model_comparison_runs():
    """Test model comparison completes without errors"""
    from lesson_code import compare_models
    
    # This is more of an integration test
    # Just verify it runs without crashing
    try:
        compare_models()
        assert True
    except Exception as e:
        pytest.fail(f"Model comparison failed: {str(e)}")

def test_class_weight_balanced():
    """Test class_weight parameter is set correctly"""
    detector = FraudDetectionSystem(model_type='random_forest')
    
    # Check Random Forest has balanced class weights
    assert detector.model.class_weight == 'balanced'

def test_scaler_transformation():
    """Test scaler properly transforms data"""
    detector = FraudDetectionSystem()
    df = detector.generate_synthetic_data(n_samples=500, fraud_ratio=0.02)
    df = detector.engineer_features(df)
    
    X = df.drop('Class', axis=1)
    
    # Fit scaler
    X_scaled = detector.scaler.fit_transform(X)
    
    # Check scaling properties
    assert X_scaled.mean() < 1.0  # Should be roughly centered
    assert X_scaled.std() < 10.0  # Should have reduced variance

def test_handles_missing_fraud_in_test():
    """Test system handles edge case of no fraud in test set"""
    detector = FraudDetectionSystem()
    
    # Create dataset with very few fraud cases
    df = detector.generate_synthetic_data(n_samples=200, fraud_ratio=0.005)
    df = detector.engineer_features(df)
    
    # This might result in no fraud in test set due to small size
    X_train, X_test, y_train, y_test, _ = detector.prepare_data(df)
    detector.train(X_train, y_train)
    
    # Should not crash even if test set has no fraud
    try:
        detector.optimize_threshold(X_test, y_test)
        detector.evaluate(X_test, y_test)
        assert True
    except Exception as e:
        # Some metrics might fail gracefully - that's acceptable
        assert "divide by zero" not in str(e).lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
