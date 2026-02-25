"""
Day 131 Test Suite — 20 tests covering data pipeline and forward pass.
Run: pytest tests/test_lesson.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from activations import relu, relu_derivative, softmax
from data_loader import one_hot, get_batch
from network import MNISTNetwork


# ── Activation Tests ─────────────────────────────────────────────────────────

class TestReLU:
    def test_relu_kills_negatives(self):
        z = np.array([-3.0, -0.1, 0.0, 0.1, 5.0])
        result = relu(z)
        assert np.all(result >= 0)

    def test_relu_passes_positives_unchanged(self):
        z = np.array([1.0, 2.5, 100.0])
        assert np.allclose(relu(z), z)

    def test_relu_zero_boundary(self):
        assert relu(np.array([0.0]))[0] == 0.0

    def test_relu_derivative_positive(self):
        z = np.array([1.0, 2.0, 3.0])
        assert np.all(relu_derivative(z) == 1.0)

    def test_relu_derivative_negative(self):
        z = np.array([-1.0, -0.5])
        assert np.all(relu_derivative(z) == 0.0)


class TestSoftmax:
    def test_softmax_sums_to_one(self):
        z = np.random.randn(10, 10)
        probs = softmax(z)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_softmax_all_positive(self):
        z = np.random.randn(5, 10)
        assert np.all(softmax(z) > 0)

    def test_softmax_large_input_stable(self):
        # Should not overflow with large values
        z = np.array([[1000.0, 1001.0, 1002.0]])
        probs = softmax(z)
        assert np.all(np.isfinite(probs))
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)

    def test_softmax_argmax_correct(self):
        z = np.array([[0.0, 0.0, 10.0, 0.0]])
        probs = softmax(z)
        assert np.argmax(probs) == 2


# ── Data Pipeline Tests ───────────────────────────────────────────────────────

class TestOneHot:
    def test_shape(self):
        labels = np.array([0, 1, 5, 9])
        encoded = one_hot(labels)
        assert encoded.shape == (4, 10)

    def test_correct_class_hot(self):
        encoded = one_hot(np.array([3]))
        assert encoded[0, 3] == 1.0

    def test_all_other_cold(self):
        encoded = one_hot(np.array([7]))
        row = encoded[0]
        assert row[7] == 1.0
        assert row.sum() == 1.0

    def test_batch_encoding(self):
        labels = np.arange(10)
        encoded = one_hot(labels)
        # Diagonal should all be 1.0
        assert np.allclose(np.diag(encoded), 1.0)

    def test_row_sums_one(self):
        labels = np.array([0, 3, 7, 9])
        encoded = one_hot(labels)
        assert np.allclose(encoded.sum(axis=1), 1.0)


# ── Network Tests ─────────────────────────────────────────────────────────────

class TestMNISTNetwork:
    def setup_method(self):
        self.net = MNISTNetwork(seed=0)
        self.X = np.random.randn(32, 784)

    def test_output_shape(self):
        probs = self.net.forward(self.X)
        assert probs.shape == (32, 10)

    def test_output_sums_to_one(self):
        probs = self.net.forward(self.X)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_output_all_positive(self):
        probs = self.net.forward(self.X)
        assert np.all(probs > 0)

    def test_cache_populated(self):
        self.net.forward(self.X)
        for key in ['X', 'z1', 'a1', 'z2', 'a2', 'z3', 'a3']:
            assert key in self.net.cache

    def test_predict_returns_integers(self):
        preds = self.net.predict(self.X)
        assert preds.dtype in [np.int32, np.int64, int]
        assert preds.shape == (32,)

    def test_predict_valid_classes(self):
        preds = self.net.predict(self.X)
        assert np.all(preds >= 0) and np.all(preds <= 9)

    def test_param_count(self):
        # 784*128 + 128 + 128*64 + 64 + 64*10 + 10
        expected = 784*128 + 128 + 128*64 + 64 + 64*10 + 10
        assert self.net.param_count() == expected
