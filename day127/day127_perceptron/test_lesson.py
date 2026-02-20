"""
Day 127 Test Suite — 20 tests covering perceptron correctness,
convergence guarantees, and the XOR failure case.
Run: pytest test_lesson.py -v
"""

import numpy as np
import pytest
from lesson_code import Perceptron, get_and_data, get_xor_data


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def trained_and_perceptron():
    X, y = get_and_data()
    p = Perceptron(n_inputs=2, learning_rate=0.1, max_epochs=100)
    p.train(X, y, verbose=False)
    return p, X, y

@pytest.fixture
def trained_xor_perceptron():
    X, y = get_xor_data()
    p = Perceptron(n_inputs=2, learning_rate=0.1, max_epochs=100)
    p.train(X, y, verbose=False)
    return p, X, y


# =============================================================================
# Initialization Tests
# =============================================================================
class TestPerceptronInit:
    def test_weights_initialized_to_zero(self):
        p = Perceptron(n_inputs=3)
        assert np.all(p.weights == 0)

    def test_bias_initialized_to_zero(self):
        p = Perceptron(n_inputs=2)
        assert p.bias == 0.0

    def test_correct_weight_shape(self):
        p = Perceptron(n_inputs=5)
        assert p.weights.shape == (5,)

    def test_learning_rate_stored(self):
        p = Perceptron(n_inputs=2, learning_rate=0.05)
        assert p.lr == 0.05


# =============================================================================
# Activation Function Tests
# =============================================================================
class TestStepActivation:
    def test_step_fires_at_zero(self):
        p = Perceptron(n_inputs=1)
        assert p._step(0) == 1

    def test_step_fires_positive(self):
        p = Perceptron(n_inputs=1)
        assert p._step(1.5) == 1

    def test_step_silent_negative(self):
        p = Perceptron(n_inputs=1)
        assert p._step(-0.001) == 0

    def test_step_output_is_binary(self):
        p = Perceptron(n_inputs=1)
        for z in [-10, -1, -0.001, 0, 0.001, 1, 10]:
            assert p._step(z) in [0, 1]


# =============================================================================
# Forward Pass Tests
# =============================================================================
class TestForwardPass:
    def test_predict_returns_binary(self):
        p = Perceptron(n_inputs=2)
        result = p.predict(np.array([1.0, 1.0]))
        assert result in [0, 1]

    def test_predict_batch_shape(self):
        p = Perceptron(n_inputs=2)
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        preds = p.predict_batch(X)
        assert preds.shape == (4,)

    def test_zeroed_weights_always_fires_at_zero(self):
        # weights=0, bias=0 => z=0 => step(0)=1 for any input
        p = Perceptron(n_inputs=2)
        result = p.predict(np.array([0.5, 0.5]))
        assert result == 1


# =============================================================================
# AND Gate Convergence Tests
# =============================================================================
class TestANDGate:
    def test_and_achieves_100_accuracy(self, trained_and_perceptron):
        p, X, y = trained_and_perceptron
        preds = p.predict_batch(X)
        assert np.all(preds == y)

    def test_and_converged(self, trained_and_perceptron):
        p, X, y = trained_and_perceptron
        assert p.training_errors[-1] == 0

    def test_and_specific_predictions(self, trained_and_perceptron):
        p, X, y = trained_and_perceptron
        assert p.predict(np.array([0.0, 0.0])) == 0  # 0 AND 0 = 0
        assert p.predict(np.array([0.0, 1.0])) == 0  # 0 AND 1 = 0
        assert p.predict(np.array([1.0, 0.0])) == 0  # 1 AND 0 = 0
        assert p.predict(np.array([1.0, 1.0])) == 1  # 1 AND 1 = 1

    def test_and_training_errors_logged(self, trained_and_perceptron):
        p, X, y = trained_and_perceptron
        assert len(p.training_errors) > 0


# =============================================================================
# XOR Failure Tests — the core lesson
# =============================================================================
class TestXORFailure:
    def test_xor_never_fully_converges(self, trained_xor_perceptron):
        """XOR is not linearly separable: at least 1 error must remain."""
        p, X, y = trained_xor_perceptron
        preds = p.predict_batch(X)
        accuracy = np.mean(preds == y)
        # A single perceptron can get at most 3/4 correct on XOR
        assert accuracy < 1.0, "Perceptron cannot solve XOR — if this passes, something's wrong"

    def test_xor_max_accuracy_is_75_percent(self, trained_xor_perceptron):
        p, X, y = trained_xor_perceptron
        preds = p.predict_batch(X)
        n_correct = np.sum(preds == y)
        assert n_correct <= 3, "Maximum 3/4 correct is the theoretical limit for XOR with 1 neuron"

    def test_xor_data_shape(self):
        X, y = get_xor_data()
        assert X.shape == (4, 2)
        assert y.shape == (4,)
        assert set(y.tolist()) == {0, 1}


# =============================================================================
# Dataset Integrity Tests
# =============================================================================
class TestDatasets:
    def test_and_truth_table(self):
        X, y = get_and_data()
        expected = {(0,0):0, (0,1):0, (1,0):0, (1,1):1}
        for xi, yi in zip(X.astype(int), y):
            assert expected[tuple(xi)] == yi

    def test_xor_truth_table(self):
        X, y = get_xor_data()
        expected = {(0,0):0, (0,1):1, (1,0):1, (1,1):0}
        for xi, yi in zip(X.astype(int), y):
            assert expected[tuple(xi)] == yi
