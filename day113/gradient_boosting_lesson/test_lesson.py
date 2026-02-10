"""
Comprehensive test suite for Gradient Boosting implementation.
Validates correctness, edge cases, and production readiness.
"""

import numpy as np
import pytest
from lesson_code import GradientBoostingClassifier, FraudDetectionSystem


class TestGradientBoostingClassifier:
    """Test custom GBM implementation."""
    
    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = GradientBoostingClassifier()
        
        assert model.n_estimators == 100
        assert model.learning_rate == 0.1
        assert model.max_depth == 3
        assert len(model.trees) == 0
        assert len(model.training_losses) == 0
    
    def test_custom_parameters(self):
        """Test model with custom parameters."""
        model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=5
        )
        
        assert model.n_estimators == 50
        assert model.learning_rate == 0.05
        assert model.max_depth == 5
    
    def test_sigmoid_function(self):
        """Test sigmoid activation correctness."""
        model = GradientBoostingClassifier()
        
        # Test known values
        assert abs(model._sigmoid(0) - 0.5) < 1e-6
        assert abs(model._sigmoid(10) - 1.0) < 1e-4  # More lenient tolerance for large values
        assert abs(model._sigmoid(-10) - 0.0) < 1e-4  # More lenient tolerance for large values
        
        # Test array input
        x = np.array([-2, 0, 2])
        result = model._sigmoid(x)
        assert result.shape == (3,)
        assert np.all((result >= 0) & (result <= 1))
    
    def test_log_loss_function(self):
        """Test log loss calculation."""
        model = GradientBoostingClassifier()
        
        # Perfect predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.01, 0.99, 0.01, 0.99])
        loss = model._log_loss(y_true, y_pred)
        
        assert loss < 0.05  # Should be very low
        
        # Random predictions (50% probability)
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        loss = model._log_loss(y_true, y_pred)
        
        assert abs(loss - 0.693) < 0.01  # Should be close to log(2)
    
    def test_simple_classification(self):
        """Test basic binary classification."""
        # Create linearly separable data
        np.random.seed(42)
        X = np.random.randn(200, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Train model
        model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1)
        model.fit(X, y)
        
        # Check training occurred
        assert len(model.trees) == 20
        assert len(model.training_losses) == 20
        
        # Check loss decreased
        assert model.training_losses[-1] < model.training_losses[0]
        
        # Check predictions
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        
        assert accuracy > 0.85  # Should achieve high accuracy on training data
    
    def test_predict_proba_shape(self):
        """Test probability prediction output shape."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        
        model = GradientBoostingClassifier(n_estimators=10)
        model.fit(X, y)
        
        proba = model.predict_proba(X)
        
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all((proba >= 0) & (proba <= 1))  # Valid probabilities
    
    def test_predict_consistency(self):
        """Test that predict() matches predict_proba() threshold."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        
        model = GradientBoostingClassifier(n_estimators=10)
        model.fit(X, y)
        
        predictions = model.predict(X)
        proba = model.predict_proba(X)
        predictions_from_proba = (proba[:, 1] >= 0.5).astype(int)
        
        assert np.array_equal(predictions, predictions_from_proba)
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        np.random.seed(42)
        X = np.random.randn(200, 4)
        # Make feature 0 and 1 highly predictive
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = GradientBoostingClassifier(n_estimators=30)
        model.fit(X, y)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == 4
        assert abs(sum(importance.values()) - 1.0) < 1e-6  # Sum to 1
        
        # Features 0 and 1 should be most important
        important_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        assert important_features[0][0] in ['feature_0', 'feature_1']
    
    def test_learning_rate_effect(self):
        """Test that learning rate affects training."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        
        # High learning rate
        model_high = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5)
        model_high.fit(X, y)
        
        # Low learning rate
        model_low = GradientBoostingClassifier(n_estimators=20, learning_rate=0.05)
        model_low.fit(X, y)
        
        # Low learning rate should have slower convergence (higher final loss)
        # for same number of iterations
        assert model_low.training_losses[-1] > model_high.training_losses[-1]
    
    def test_subsample_parameter(self):
        """Test that subsample parameter works."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        
        # With subsampling
        model_subsample = GradientBoostingClassifier(
            n_estimators=20,
            subsample=0.7
        )
        model_subsample.fit(X, y)
        
        # Full sample
        model_full = GradientBoostingClassifier(
            n_estimators=20,
            subsample=1.0
        )
        model_full.fit(X, y)
        
        # Both should train successfully
        assert len(model_subsample.trees) == 20
        assert len(model_full.trees) == 20
    
    def test_edge_case_single_sample(self):
        """Test with single training sample."""
        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([1])
        
        model = GradientBoostingClassifier(n_estimators=5)
        model.fit(X, y)
        
        prediction = model.predict(X)
        assert prediction.shape == (1,)
    
    def test_edge_case_all_same_label(self):
        """Test with all samples having same label."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.ones(50)  # All positive class
        
        model = GradientBoostingClassifier(n_estimators=10)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert np.all(predictions == 1)
    
    def test_edge_case_binary_features(self):
        """Test with binary features only."""
        np.random.seed(42)
        X = np.random.randint(0, 2, (100, 5))
        y = (X[:, 0] & X[:, 1]).astype(int)  # AND logic
        
        model = GradientBoostingClassifier(n_estimators=30)
        model.fit(X, y)
        
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        
        assert accuracy > 0.8


class TestFraudDetectionSystem:
    """Test fraud detection pipeline."""
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        system = FraudDetectionSystem()
        X, y = system.generate_transaction_data(n_samples=1000, fraud_ratio=0.1)
        
        assert X.shape == (1000, 5)
        assert y.shape == (1000,)
        assert np.sum(y) == 100  # 10% fraud
        assert len(system.feature_names) == 5
    
    def test_fraud_ratio_variation(self):
        """Test different fraud ratios."""
        system = FraudDetectionSystem()
        
        for ratio in [0.05, 0.15, 0.25]:
            X, y = system.generate_transaction_data(n_samples=1000, fraud_ratio=ratio)
            actual_ratio = np.sum(y) / len(y)
            assert abs(actual_ratio - ratio) < 0.01
    
    def test_feature_distributions(self):
        """Test that features have reasonable distributions."""
        system = FraudDetectionSystem()
        X, y = system.generate_transaction_data(n_samples=2000)
        
        # Transaction amounts should be positive
        assert np.all(X[:, 0] > 0)
        
        # Time of day should be 0-24
        assert np.all(X[:, 1] >= 0) and np.all(X[:, 1] < 24)
        
        # Location distance should be positive
        assert np.all(X[:, 2] >= 0)
        
        # Merchant trust should be 0-1
        assert np.all(X[:, 4] >= 0) and np.all(X[:, 4] <= 1)
    
    def test_training_execution(self):
        """Test that training executes without errors."""
        system = FraudDetectionSystem(n_estimators=20)
        X, y = system.generate_transaction_data(n_samples=500)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        system.train(X_train, y_train)
        
        assert len(system.model.trees) == 20
        assert system.training_time > 0
    
    def test_evaluation_metrics(self):
        """Test evaluation returns all expected metrics."""
        system = FraudDetectionSystem(n_estimators=20)
        X, y = system.generate_transaction_data(n_samples=500)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        system.train(X_train, y_train)
        metrics = system.evaluate(X_test, y_test)
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_minimum_performance(self):
        """Test that model achieves minimum acceptable performance."""
        system = FraudDetectionSystem(n_estimators=50)
        X, y = system.generate_transaction_data(n_samples=2000)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        system.train(X_train, y_train)
        metrics = system.evaluate(X_test, y_test)
        
        # Model should beat random guessing significantly
        assert metrics['accuracy'] > 0.75
        assert metrics['roc_auc'] > 0.75


class TestProductionReadiness:
    """Test production deployment considerations."""
    
    def test_prediction_speed(self):
        """Test prediction latency for production requirements."""
        import time
        
        system = FraudDetectionSystem(n_estimators=50)
        X, y = system.generate_transaction_data(n_samples=1000)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        system.train(X_train, y_train)
        
        # Measure prediction time for single transaction
        single_transaction = X_test[0:1]
        
        start_time = time.time()
        for _ in range(100):
            _ = system.model.predict(single_transaction)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / 100 * 1000
        
        # Should predict in under 10ms (production requirement)
        assert avg_time_ms < 10, f"Prediction too slow: {avg_time_ms:.2f}ms"
    
    def test_batch_prediction(self):
        """Test batch prediction capability."""
        system = FraudDetectionSystem(n_estimators=30)
        X, y = system.generate_transaction_data(n_samples=500)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        system.train(X_train, y_train)
        
        # Predict on large batch
        batch_size = 1000
        X_batch = np.random.randn(batch_size, 5)
        predictions = system.model.predict(X_batch)
        
        assert predictions.shape == (batch_size,)
        assert np.all((predictions == 0) | (predictions == 1))
    
    def test_model_serialization(self):
        """Test model can be saved and loaded."""
        import joblib
        import tempfile
        import os
        
        system = FraudDetectionSystem(n_estimators=20)
        X, y = system.generate_transaction_data(n_samples=300)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        system.train(X_train, y_train)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            model_path = f.name
            joblib.dump(system.model, model_path)
        
        # Load model
        loaded_model = joblib.load(model_path)
        
        # Compare predictions
        original_pred = system.model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        
        assert np.array_equal(original_pred, loaded_pred)
        
        # Cleanup
        os.unlink(model_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
