"""
Test suite for Day 48: Logistic Regression Theory
"""

import pytest
import numpy as np
from lesson_code import (
    SigmoidFunction,
    LogLossCalculator,
    DecisionBoundaryAnalyzer,
    LogisticRegressionDemo
)


class TestSigmoidFunction:
    """Test sigmoid function implementation"""
    
    def test_sigmoid_at_zero(self):
        """Sigmoid(0) should equal 0.5"""
        result = SigmoidFunction.compute(np.array([0]))
        assert np.isclose(result[0], 0.5, atol=1e-6)
    
    def test_sigmoid_positive_values(self):
        """Large positive values should approach 1"""
        result = SigmoidFunction.compute(np.array([10, 100]))
        assert result[0] > 0.99
        assert result[1] > 0.99
    
    def test_sigmoid_negative_values(self):
        """Large negative values should approach 0"""
        result = SigmoidFunction.compute(np.array([-10, -100]))
        assert result[0] < 0.01
        assert result[1] < 0.01
    
    def test_sigmoid_range(self):
        """Output should always be between 0 and 1"""
        z = np.linspace(-100, 100, 1000)
        result = SigmoidFunction.compute(z)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_sigmoid_symmetry(self):
        """σ(z) + σ(-z) should equal 1"""
        z = np.array([1, 2, 3, 5])
        pos = SigmoidFunction.compute(z)
        neg = SigmoidFunction.compute(-z)
        assert np.allclose(pos + neg, 1.0, atol=1e-6)
    
    def test_sigmoid_derivative_at_zero(self):
        """Derivative at z=0 should be 0.25"""
        result = SigmoidFunction.derivative(np.array([0]))
        assert np.isclose(result[0], 0.25, atol=1e-6)
    
    def test_sigmoid_derivative_range(self):
        """Derivative should always be positive and <= 0.25"""
        z = np.linspace(-10, 10, 100)
        result = SigmoidFunction.derivative(z)
        assert np.all(result >= 0)
        assert np.all(result <= 0.25)


class TestLogLossCalculator:
    """Test log-loss calculation"""
    
    def test_perfect_predictions(self):
        """Perfect predictions should have near-zero loss"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9999, 0.0001, 0.9999, 0.0001])
        loss = LogLossCalculator.compute(y_true, y_pred)
        assert loss < 0.01
    
    def test_worst_predictions(self):
        """Completely wrong predictions should have high loss"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.01, 0.99, 0.01, 0.99])
        loss = LogLossCalculator.compute(y_true, y_pred)
        assert loss > 2.0
    
    def test_random_predictions(self):
        """Random guessing (0.5) should have loss ≈ 0.693"""
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        loss = LogLossCalculator.compute(y_true, y_pred)
        assert np.isclose(loss, np.log(2), atol=0.01)  # log(2) ≈ 0.693
    
    def test_loss_increases_with_confidence_error(self):
        """More confident wrong predictions should have higher loss"""
        y_true = 1
        loss_90 = LogLossCalculator.compute_per_example(y_true, 0.9)
        loss_50 = LogLossCalculator.compute_per_example(y_true, 0.5)
        loss_10 = LogLossCalculator.compute_per_example(y_true, 0.1)
        
        assert loss_90 < loss_50 < loss_10
    
    def test_loss_symmetry(self):
        """Loss should be symmetric for complementary predictions"""
        loss_pos = LogLossCalculator.compute_per_example(1, 0.8)
        loss_neg = LogLossCalculator.compute_per_example(0, 0.2)
        assert np.isclose(loss_pos, loss_neg, atol=1e-6)


class TestDecisionBoundaryAnalyzer:
    """Test decision boundary and threshold analysis"""
    
    def test_default_threshold(self):
        """Default threshold should be 0.5"""
        analyzer = DecisionBoundaryAnalyzer()
        assert analyzer.threshold == 0.5
    
    def test_predictions_at_threshold(self):
        """Probabilities >= threshold should predict 1"""
        analyzer = DecisionBoundaryAnalyzer(threshold=0.5)
        probs = np.array([0.3, 0.5, 0.7, 0.9])
        preds = analyzer.predict_class(probs)
        expected = np.array([0, 1, 1, 1])
        assert np.array_equal(preds, expected)
    
    def test_custom_threshold(self):
        """Custom threshold should change predictions"""
        analyzer_low = DecisionBoundaryAnalyzer(threshold=0.3)
        analyzer_high = DecisionBoundaryAnalyzer(threshold=0.8)
        probs = np.array([0.4, 0.6])
        
        preds_low = analyzer_low.predict_class(probs)
        preds_high = analyzer_high.predict_class(probs)
        
        assert np.array_equal(preds_low, np.array([1, 1]))
        assert np.array_equal(preds_high, np.array([0, 0]))
    
    def test_threshold_extremes(self):
        """Test behavior at extreme thresholds"""
        probs = np.array([0.1, 0.5, 0.9])
        
        # Threshold 0.0: Everything predicts positive
        analyzer = DecisionBoundaryAnalyzer(threshold=0.0)
        assert np.all(analyzer.predict_class(probs) == 1)
        
        # Threshold 1.0: Everything predicts negative
        analyzer = DecisionBoundaryAnalyzer(threshold=1.0)
        assert np.all(analyzer.predict_class(probs) == 0)


class TestLogisticRegressionDemo:
    """Test the complete demo functionality"""
    
    def test_demo_initialization(self):
        """Demo should initialize all components"""
        demo = LogisticRegressionDemo()
        assert demo.sigmoid is not None
        assert demo.log_loss is not None
        assert demo.decision_analyzer is not None
    
    def test_demo_runs_without_errors(self):
        """Complete demo should run without exceptions"""
        demo = LogisticRegressionDemo()
        try:
            demo.run_complete_demo()
            assert True
        except Exception as e:
            pytest.fail(f"Demo failed with exception: {e}")


class TestMathematicalProperties:
    """Test important mathematical properties"""
    
    def test_sigmoid_inverse_relationship(self):
        """σ(z) = 1 - σ(-z)"""
        z = np.array([0.5, 1.0, 2.0, 3.0])
        pos = SigmoidFunction.compute(z)
        neg = SigmoidFunction.compute(-z)
        assert np.allclose(pos, 1 - neg, atol=1e-6)
    
    def test_derivative_formula(self):
        """Verify σ'(z) = σ(z)(1 - σ(z))"""
        z = np.array([0, 1, 2])
        sig = SigmoidFunction.compute(z)
        deriv = SigmoidFunction.derivative(z)
        expected = sig * (1 - sig)
        assert np.allclose(deriv, expected, atol=1e-6)
    
    def test_log_loss_bounds(self):
        """Log-loss should never be negative"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.random(100)
        loss = LogLossCalculator.compute(y_true, y_pred)
        assert loss >= 0


def test_production_scenarios():
    """Test scenarios mimicking production use cases"""
    
    # Scenario 1: High-confidence correct predictions (good model)
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0.95, 0.92, 0.88, 0.08, 0.12, 0.15])
    loss = LogLossCalculator.compute(y_true, y_pred)
    assert loss < 0.2  # Low loss for good predictions
    
    # Scenario 2: Poor model (random guessing)
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    loss = LogLossCalculator.compute(y_true, y_pred)
    assert 0.6 < loss < 0.8  # ~log(2) for random guessing
    
    # Scenario 3: Confident but wrong (bad model)
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.92])
    loss = LogLossCalculator.compute(y_true, y_pred)
    assert loss > 2.0  # High loss for confident wrong predictions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
