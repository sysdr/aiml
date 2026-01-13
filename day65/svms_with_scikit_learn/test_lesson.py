"""
Tests for Day 65: SVMs with Scikit-learn
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os


def test_imports():
    """Test that all required libraries are available"""
    import sklearn
    import numpy
    import pandas
    import matplotlib
    import seaborn
    assert True


def test_dataset_generation():
    """Test fraud dataset generation"""
    from lesson_code import generate_fraud_dataset
    
    df = generate_fraud_dataset(n_samples=1000, fraud_ratio=0.1)
    
    # Check shape
    assert len(df) == 1000
    assert len(df.columns) == 5
    
    # Check fraud ratio
    fraud_ratio = df['is_fraud'].mean()
    assert 0.08 <= fraud_ratio <= 0.12  # Allow some variance
    
    # Check feature ranges
    assert df['amount'].min() >= 0
    assert 0 <= df['hour'].min() and df['hour'].max() < 24
    assert df['distance_from_last'].min() >= 0
    assert 0 <= df['merchant_risk_score'].min() <= 1


def test_svm_without_scaling():
    """Test SVM training without scaling (should work but perform poorly)"""
    from lesson_code import generate_fraud_dataset
    
    df = generate_fraud_dataset(n_samples=500)
    X = df[['amount', 'hour', 'distance_from_last', 'merchant_risk_score']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    
    # Should complete without error
    y_pred = svm.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    
    # Score might be low, but should be > 0.5
    assert score > 0.5


def test_svm_with_scaling():
    """Test SVM training with scaling pipeline"""
    from lesson_code import generate_fraud_dataset
    
    df = generate_fraud_dataset(n_samples=500)
    X = df[['amount', 'hour', 'distance_from_last', 'merchant_risk_score']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # With scaling, should perform significantly better
    assert score > 0.85
    assert f1 > 0.3  # F1 should be reasonable given class imbalance


def test_scaling_improves_performance():
    """Verify that scaling improves SVM performance"""
    from lesson_code import generate_fraud_dataset
    
    df = generate_fraud_dataset(n_samples=500)
    X = df[['amount', 'hour', 'distance_from_last', 'merchant_risk_score']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Without scaling
    svm_no_scale = SVC(kernel='rbf', random_state=42)
    svm_no_scale.fit(X_train, y_train)
    score_no_scale = svm_no_scale.score(X_test, y_test)
    
    # With scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    score_with_scale = pipeline.score(X_test, y_test)
    
    # Scaling should improve or maintain performance (not degrade)
    assert score_with_scale >= score_no_scale


def test_different_kernels():
    """Test that different kernels can be used"""
    from lesson_code import generate_fraud_dataset
    
    df = generate_fraud_dataset(n_samples=300)
    X = df[['amount', 'hour', 'distance_from_last', 'merchant_risk_score']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    for kernel in ['linear', 'rbf', 'poly']:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        
        # All kernels should work
        assert score > 0.5


def test_probability_predictions():
    """Test that probability predictions work"""
    from lesson_code import generate_fraud_dataset
    
    df = generate_fraud_dataset(n_samples=300)
    X = df[['amount', 'hour', 'distance_from_last', 'merchant_risk_score']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)
    
    # Check probability shape
    assert probabilities.shape == (len(X_test), 2)
    
    # Probabilities should sum to 1
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    # Probabilities should be between 0 and 1
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


def test_model_persistence():
    """Test saving and loading model"""
    from lesson_code import generate_fraud_dataset
    
    df = generate_fraud_dataset(n_samples=300)
    X = df[['amount', 'hour', 'distance_from_last', 'merchant_risk_score']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Train model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    # Save model
    filename = 'test_fraud_svm.pkl'
    joblib.dump(pipeline, filename)
    
    # Load model
    loaded_model = joblib.load(filename)
    
    # Make predictions with both
    pred_original = pipeline.predict(X_test)
    pred_loaded = loaded_model.predict(X_test)
    
    # Predictions should be identical
    assert np.array_equal(pred_original, pred_loaded)
    
    # Cleanup
    os.remove(filename)


def test_class_weight_balanced():
    """Test that class_weight='balanced' handles imbalanced data"""
    from lesson_code import generate_fraud_dataset
    
    # Create highly imbalanced dataset
    df = generate_fraud_dataset(n_samples=500, fraud_ratio=0.02)
    X = df[['amount', 'hour', 'distance_from_last', 'merchant_risk_score']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Train with balanced class weights
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', class_weight='balanced', random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Should catch at least some fraud cases
    fraud_caught = np.sum((y_test == 1) & (y_pred == 1))
    assert fraud_caught > 0


def test_hyperparameter_ranges():
    """Test that different hyperparameters can be set"""
    from lesson_code import generate_fraud_dataset
    
    df = generate_fraud_dataset(n_samples=200)
    X = df[['amount', 'hour', 'distance_from_last', 'merchant_risk_score']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Test different C values
    for C in [0.1, 1, 10]:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=C, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        assert pipeline.score(X_test, y_test) > 0.5
    
    # Test different gamma values
    for gamma in [0.001, 0.01, 0.1]:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', gamma=gamma, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        assert pipeline.score(X_test, y_test) > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
