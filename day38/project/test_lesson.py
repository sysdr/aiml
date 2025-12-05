"""
Tests for Day 38: Machine Learning Workflow
Verify understanding of ML pipeline stages
"""

import pytest
import numpy as np
import pandas as pd
from lesson_code import MLWorkflowPipeline


@pytest.fixture
def pipeline():
    """Create a fresh pipeline for each test"""
    return MLWorkflowPipeline()


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    reviews = [
        "Great product! Highly recommend.",
        "Terrible quality. Very disappointed.",
        "Amazing! Best purchase ever.",
        "Awful. Complete waste of money.",
        "Love it! Perfect for my needs.",
        "Bad product. Does not work."
    ]
    sentiments = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    return pd.DataFrame({'review_text': reviews, 'sentiment': sentiments})


def test_problem_definition(pipeline):
    """Test Stage 1: Problem Definition"""
    problem = pipeline.define_problem()
    
    assert problem['task'] == 'binary_classification'
    assert problem['target_variable'] == 'sentiment'
    assert problem['success_threshold'] == 0.80
    assert len(pipeline.logs) > 0
    print("✓ Problem definition stage works correctly")


def test_data_collection(pipeline):
    """Test Stage 2: Data Collection"""
    df = pipeline.collect_data()
    
    # Check data structure
    assert isinstance(df, pd.DataFrame)
    assert 'review_text' in df.columns
    assert 'sentiment' in df.columns
    assert len(df) > 0
    
    # Check sentiment distribution
    assert df['sentiment'].isin([0, 1]).all()
    print(f"✓ Data collection works - collected {len(df)} reviews")


def test_data_preparation(pipeline, sample_data):
    """Test Stage 3: Data Preparation"""
    X_train, X_test, y_train, y_test = pipeline.prepare_data(sample_data)
    
    # Check splits exist
    assert X_train is not None
    assert X_test is not None
    assert len(y_train) > 0
    assert len(y_test) > 0
    
    # Check vectorizer was created
    assert pipeline.vectorizer is not None
    
    print(f"✓ Data preparation works - train: {len(y_train)}, test: {len(y_test)}")


def test_model_training(pipeline):
    """Test Stage 4: Model Training"""
    # Get data
    df = pipeline.collect_data()
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    
    # Train model
    model = pipeline.train_model(X_train, y_train)
    
    # Check model exists and is trained
    assert pipeline.model is not None
    assert hasattr(model, 'coef_')  # Model has been fitted
    assert model.coef_.shape[1] > 0  # Has features
    
    print("✓ Model training works correctly")


def test_model_evaluation(pipeline):
    """Test Stage 5: Model Evaluation"""
    # Full workflow up to evaluation
    df = pipeline.collect_data()
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    pipeline.train_model(X_train, y_train)
    
    # Evaluate
    metrics = pipeline.evaluate_model(X_test, y_test)
    
    # Check metrics exist and are valid
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    
    # Check metrics are in valid range [0, 1]
    for metric_name, value in metrics.items():
        assert 0 <= value <= 1, f"{metric_name} should be between 0 and 1"
    
    print(f"✓ Model evaluation works - F1 Score: {metrics['f1_score']:.3f}")


def test_model_deployment(pipeline, tmp_path):
    """Test Stage 6: Deployment"""
    # Train a model first
    df = pipeline.collect_data()
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    pipeline.train_model(X_train, y_train)
    
    # Deploy to temporary directory
    model_dir = str(tmp_path / "models")
    model_path = pipeline.deploy_model(model_dir)
    
    # Check files were created
    assert model_path is not None
    import os
    assert os.path.exists(model_path)
    
    print(f"✓ Model deployment works - saved to {model_dir}")


def test_prediction_monitoring(pipeline):
    """Test Stage 7: Monitoring & Prediction"""
    # Train model
    df = pipeline.collect_data()
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    pipeline.train_model(X_train, y_train)
    
    # Make predictions
    new_reviews = [
        "Excellent product! Very happy.",
        "Terrible. Would not buy again."
    ]
    
    results = pipeline.predict(new_reviews)
    
    # Check predictions
    assert len(results) == 2
    assert all('sentiment' in r for r in results)
    assert all('confidence' in r for r in results)
    
    # Check sentiment predictions make sense
    assert results[0]['sentiment'] == 'Positive'  # "Excellent" should be positive
    assert results[1]['sentiment'] == 'Negative'  # "Terrible" should be negative
    
    print("✓ Prediction and monitoring work correctly")


def test_end_to_end_workflow():
    """Test complete workflow from start to finish"""
    pipeline = MLWorkflowPipeline()
    
    # Run all stages
    pipeline.define_problem()
    df = pipeline.collect_data()
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    pipeline.train_model(X_train, y_train)
    metrics = pipeline.evaluate_model(X_test, y_test)
    pipeline.deploy_model()
    
    # Make a prediction
    results = pipeline.predict(["Great product!"])
    
    # Check everything worked
    assert len(pipeline.logs) > 0  # Pipeline logged events
    assert metrics['f1_score'] > 0.5  # Model performs reasonably
    assert results[0]['sentiment'] in ['Positive', 'Negative']
    
    print("✓ Complete end-to-end workflow works!")
    print(f"  Final F1 Score: {metrics['f1_score']:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
