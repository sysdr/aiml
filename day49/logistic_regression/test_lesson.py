"""
Test Suite for Day 49: Logistic Regression for Binary Classification
Comprehensive tests ensuring the classifier works correctly
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lesson_code import SpamClassifier, ModelEvaluator, create_sample_dataset


class TestSpamClassifier:
    """Test the SpamClassifier class"""
    
    def test_classifier_initialization(self):
        """Test that classifier initializes correctly"""
        classifier = SpamClassifier(max_features=500)
        assert classifier.is_trained == False
        assert classifier.vectorizer.max_features == 500
        
    def test_prepare_data(self):
        """Test data preparation"""
        classifier = SpamClassifier()
        texts = ["hello world", "spam message"]
        labels = [0, 1]
        
        features, y = classifier.prepare_data(texts, labels)
        assert features.shape[0] == 2
        assert len(y) == 2
        assert y[0] == 0 and y[1] == 1
        
    def test_training(self):
        """Test model training"""
        classifier = SpamClassifier()
        
        # Create simple dataset
        texts = ["legitimate email message"] * 50 + ["spam winner prize"] * 50
        labels = [0] * 50 + [1] * 50
        
        X, y = classifier.prepare_data(texts, labels)
        classifier.train(X, y)
        
        assert classifier.is_trained == True
        
    def test_prediction_before_training(self):
        """Test that prediction fails before training"""
        classifier = SpamClassifier()
        X = np.random.rand(10, 100)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            classifier.predict(X)
            
    def test_prediction_after_training(self):
        """Test predictions after training"""
        classifier = SpamClassifier()
        
        # Train on simple data
        texts = ["legitimate email"] * 50 + ["spam prize winner"] * 50
        labels = [0] * 50 + [1] * 50
        
        X, y = classifier.prepare_data(texts, labels)
        classifier.train(X, y)
        
        # Make predictions
        predictions = classifier.predict(X)
        assert len(predictions) == 100
        assert all(p in [0, 1] for p in predictions)
        
    def test_predict_proba(self):
        """Test probability predictions"""
        classifier = SpamClassifier()
        
        texts = ["good email"] * 50 + ["spam"] * 50
        labels = [0] * 50 + [1] * 50
        
        X, y = classifier.prepare_data(texts, labels)
        classifier.train(X, y)
        
        proba = classifier.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        
    def test_predict_text(self):
        """Test text prediction interface"""
        classifier = SpamClassifier()
        
        texts = ["hello friend"] * 50 + ["winner prize"] * 50
        labels = [0] * 50 + [1] * 50
        
        X, y = classifier.prepare_data(texts, labels)
        classifier.train(X, y)
        
        new_texts = ["normal email", "spam winner"]
        results = classifier.predict_text(new_texts)
        
        assert len(results) == 2
        assert all('prediction' in r for r in results)
        assert all('spam_probability' in r for r in results)
        assert all('confidence' in r for r in results)


class TestModelEvaluator:
    """Test the ModelEvaluator class"""
    
    def test_evaluate_model_basic(self):
        """Test basic metrics evaluation"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        
        metrics = evaluator.evaluate_model(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        
    def test_evaluate_model_with_proba(self):
        """Test evaluation with probabilities"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9]
        ])
        
        metrics = evaluator.evaluate_model(y_true, y_pred, y_proba)
        
        assert 'roc_auc' in metrics
        assert metrics['roc_auc'] == 1.0
        
    def test_metrics_range(self):
        """Test that all metrics are in valid range"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        metrics = evaluator.evaluate_model(y_true, y_pred)
        
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} out of range: {value}"


class TestDatasetCreation:
    """Test dataset creation functionality"""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation"""
        df = create_sample_dataset()
        
        assert 'text' in df.columns
        assert 'label' in df.columns
        assert len(df) > 0
        assert set(df['label'].unique()) == {0, 1}
        
    def test_dataset_balance(self):
        """Test that dataset is balanced"""
        df = create_sample_dataset()
        
        spam_count = sum(df['label'] == 1)
        ham_count = sum(df['label'] == 0)
        
        # Should be roughly balanced
        ratio = spam_count / ham_count
        assert 0.8 < ratio < 1.2
        
    def test_dataset_has_text(self):
        """Test that all entries have text"""
        df = create_sample_dataset()
        
        assert all(len(str(text)) > 0 for text in df['text'])


class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def test_full_pipeline(self):
        """Test the complete classification pipeline"""
        # Create dataset
        df = create_sample_dataset()
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df['text'], df['label'],
            test_size=0.2,
            random_state=42
        )
        
        # Initialize classifier
        classifier = SpamClassifier()
        
        # Prepare data
        X_train, _ = classifier.prepare_data(X_train_text, y_train)
        X_test = classifier.vectorizer.transform(X_test_text)
        
        # Train
        classifier.train(X_train, y_train)
        
        # Predict
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(y_test, y_pred, y_proba)
        
        # Check that model performs reasonably well
        assert metrics['accuracy'] > 0.7
        assert metrics['f1_score'] > 0.7
        
    def test_real_time_prediction(self):
        """Test real-time prediction on new data"""
        # Train classifier on sample data
        df = create_sample_dataset()
        classifier = SpamClassifier()
        
        X, y = classifier.prepare_data(df['text'], df['label'])
        classifier.train(X, y)
        
        # Test with new emails
        new_emails = [
            "Let's schedule a meeting for Monday",
            "WIN FREE MONEY NOW!!!"
        ]
        
        results = classifier.predict_text(new_emails)
        
        assert len(results) == 2
        # First should likely be ham, second spam
        assert results[0]['prediction'] == 'HAM'
        assert results[1]['prediction'] == 'SPAM'


def test_sklearn_logistic_regression():
    """Test that sklearn's LogisticRegression is available and working"""
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression()
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert len(predictions) == 100


def test_evaluation_metrics_available():
    """Test that all required metrics are available"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score
    )
    
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    
    assert accuracy_score(y_true, y_pred) == 1.0
    assert precision_score(y_true, y_pred) == 1.0
    assert recall_score(y_true, y_pred) == 1.0
    assert f1_score(y_true, y_pred) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
