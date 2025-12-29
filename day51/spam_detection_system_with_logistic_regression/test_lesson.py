"""
Day 51: Spam Detection - Test Suite

Comprehensive tests validating the spam detection system.
Production systems include thousands of tests - these are the essentials.
"""

import pytest
import numpy as np
import pandas as pd
from lesson_code import SpamDetector
import os


class TestSpamDetector:
    """Test suite for spam detection system."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for tests."""
        return SpamDetector(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Create minimal sample dataset for testing."""
        # Create synthetic data: 10 features, 100 samples
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate labels (30% spam)
        y = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        
        # Create DataFrame
        columns = [f'feature_{i}' for i in range(n_features)] + ['is_spam']
        data = pd.DataFrame(
            np.column_stack([X, y]), 
            columns=columns
        )
        
        return data
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.random_state == 42
        assert detector.model is None
        assert detector.feature_names is None
    
    def test_prepare_features(self, detector, sample_data):
        """Test feature preparation."""
        X, y = detector.prepare_features(sample_data)
        
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] == len(sample_data.columns) - 1
        assert y.shape[0] == len(sample_data)
        assert len(detector.feature_names) == X.shape[1]
    
    def test_split_data(self, detector, sample_data):
        """Test data splitting."""
        X, y = detector.prepare_features(sample_data)
        detector.split_data(X, y, test_size=0.2)
        
        # Check split sizes
        assert len(detector.X_train) == int(len(X) * 0.8)
        assert len(detector.X_test) == len(X) - len(detector.X_train)
        assert len(detector.y_train) == len(detector.X_train)
        assert len(detector.y_test) == len(detector.X_test)
        
        # Check stratification maintains class distribution
        train_spam_ratio = detector.y_train.mean()
        test_spam_ratio = detector.y_test.mean()
        overall_spam_ratio = y.mean()
        
        assert abs(train_spam_ratio - overall_spam_ratio) < 0.1
        assert abs(test_spam_ratio - overall_spam_ratio) < 0.1
    
    def test_model_training(self, detector, sample_data):
        """Test model training."""
        X, y = detector.prepare_features(sample_data)
        detector.split_data(X, y)
        model = detector.train_model()
        
        assert detector.model is not None
        assert hasattr(detector.model, 'coef_')
        assert hasattr(detector.model, 'intercept_')
        
        # Check model can make predictions
        predictions = detector.model.predict(detector.X_test)
        assert len(predictions) == len(detector.X_test)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_evaluation(self, detector, sample_data):
        """Test model evaluation metrics."""
        X, y = detector.prepare_features(sample_data)
        detector.split_data(X, y)
        detector.train_model()
        
        eval_metrics = detector.evaluate_model()
        
        assert 'confusion_matrix' in eval_metrics
        assert 'fpr' in eval_metrics
        assert 'tpr' in eval_metrics
        assert 'roc_auc' in eval_metrics
        
        # ROC-AUC should be between 0 and 1
        assert 0 <= eval_metrics['roc_auc'] <= 1
        
        # Confusion matrix should be 2x2
        assert eval_metrics['confusion_matrix'].shape == (2, 2)
    
    def test_feature_analysis(self, detector, sample_data):
        """Test feature importance analysis."""
        X, y = detector.prepare_features(sample_data)
        detector.split_data(X, y)
        detector.train_model()
        
        feature_importance = detector.analyze_features(top_n=5)
        
        assert len(feature_importance) == len(detector.feature_names)
        assert 'feature' in feature_importance.columns
        assert 'coefficient' in feature_importance.columns
        assert 'abs_coefficient' in feature_importance.columns
    
    def test_model_persistence(self, detector, sample_data, tmp_path):
        """Test model save and load."""
        X, y = detector.prepare_features(sample_data)
        detector.split_data(X, y)
        detector.train_model()
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        detector.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = SpamDetector.load_model(str(model_path))
        
        # Verify loaded model works
        predictions = loaded_model.predict(detector.X_test)
        assert len(predictions) == len(detector.X_test)
    
    def test_production_simulation(self, detector, sample_data):
        """Test production inference simulation."""
        X, y = detector.prepare_features(sample_data)
        detector.split_data(X, y)
        detector.train_model()
        
        # Should complete without errors
        detector.simulate_production_inference(num_emails=50)
    
    def test_class_balance_handling(self, detector):
        """Test model handles class imbalance."""
        # Create highly imbalanced dataset (95% ham, 5% spam)
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
        
        # Create DataFrame
        columns = [f'feature_{i}' for i in range(n_features)] + ['is_spam']
        data = pd.DataFrame(
            np.column_stack([X, y]), 
            columns=columns
        )
        
        X, y = detector.prepare_features(data)
        detector.split_data(X, y)
        detector.train_model()
        
        # Model should still predict some spam (not all ham)
        predictions = detector.model.predict(detector.X_test)
        assert predictions.sum() > 0  # At least some spam predicted


class TestDataLoading:
    """Tests for data loading functionality."""
    
    def test_spambase_data_exists(self):
        """Test spambase dataset is available."""
        # This test will pass after setup.sh downloads the data
        if os.path.exists('spambase.data'):
            assert os.path.getsize('spambase.data') > 0


class TestModelPerformance:
    """Tests for model performance benchmarks."""
    
    @pytest.fixture
    def trained_detector(self):
        """Create and train detector on full dataset."""
        if not os.path.exists('spambase.data'):
            pytest.skip("spambase.data not found - run setup.sh first")
        
        detector = SpamDetector(random_state=42)
        data = detector.load_data()
        X, y = detector.prepare_features(data)
        detector.split_data(X, y)
        detector.train_model()
        
        return detector
    
    def test_minimum_accuracy(self, trained_detector):
        """Test model achieves minimum accuracy threshold."""
        accuracy = trained_detector.model.score(
            trained_detector.X_test, 
            trained_detector.y_test
        )
        
        # Production systems typically achieve >90% accuracy
        # We set a conservative threshold of 85%
        assert accuracy >= 0.85, f"Accuracy {accuracy:.2%} below threshold"
    
    def test_minimum_roc_auc(self, trained_detector):
        """Test model achieves minimum ROC-AUC score."""
        eval_metrics = trained_detector.evaluate_model()
        roc_auc = eval_metrics['roc_auc']
        
        # ROC-AUC should be significantly better than random (0.5)
        assert roc_auc >= 0.90, f"ROC-AUC {roc_auc:.4f} below threshold"
    
    def test_inference_speed(self, trained_detector):
        """Test inference meets performance requirements."""
        import time
        
        # Sample 1000 emails
        sample_size = min(1000, len(trained_detector.X_test))
        X_sample = trained_detector.X_test[:sample_size]
        
        # Measure inference time
        start = time.time()
        predictions = trained_detector.model.predict(X_sample)
        elapsed = time.time() - start
        
        # Should process at least 50 emails/second
        emails_per_second = sample_size / elapsed
        assert emails_per_second >= 50, \
            f"Inference too slow: {emails_per_second:.0f} emails/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
