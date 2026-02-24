"""
Day 130 — Test Suite
Tests cover forward pass shapes, loss computation, gradient correctness,
weight update mechanics, and training convergence.
"""

import numpy as np
import pytest
from lesson_code import (
    NeuralNetwork, relu, relu_derivative, sigmoid, sigmoid_derivative,
    mse_loss, mse_loss_derivative, gradient_check
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def xor_data():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    return X, y

@pytest.fixture
def small_net():
    return NeuralNetwork([2, 3, 1], activation='relu', seed=0)

@pytest.fixture
def sigmoid_net():
    return NeuralNetwork([2, 3, 1], activation='sigmoid', seed=0)


# ── Activation functions ──────────────────────────────────────────────────────

def test_relu_positive():
    assert relu(np.array([2.0])) == 2.0

def test_relu_negative():
    assert relu(np.array([-3.0])) == 0.0

def test_relu_derivative_positive():
    assert relu_derivative(np.array([1.0])) == 1.0

def test_relu_derivative_negative():
    assert relu_derivative(np.array([-1.0])) == 0.0

def test_sigmoid_range():
    z = np.linspace(-10, 10, 100)
    s = sigmoid(z)
    assert np.all(s > 0) and np.all(s < 1)

def test_sigmoid_at_zero():
    assert abs(sigmoid(np.array([0.0])) - 0.5) < 1e-9


# ── Loss function ─────────────────────────────────────────────────────────────

def test_mse_loss_perfect():
    y = np.array([[1.0, 0.0]])
    assert mse_loss(y, y) == 0.0

def test_mse_loss_positive():
    pred = np.array([[1.0]])
    true = np.array([[0.0]])
    assert mse_loss(pred, true) > 0

def test_mse_loss_derivative_shape():
    pred = np.ones((4, 1))
    true = np.zeros((4, 1))
    assert mse_loss_derivative(pred, true).shape == (4, 1)


# ── Forward pass ──────────────────────────────────────────────────────────────

def test_forward_output_shape(small_net, xor_data):
    X, _ = xor_data
    out = small_net.forward(X)
    assert out.shape == (4, 1)

def test_forward_populates_cache(small_net, xor_data):
    X, _ = xor_data
    small_net.forward(X)
    assert 'a0' in small_net.cache
    assert 'z1' in small_net.cache

def test_forward_deterministic(small_net, xor_data):
    X, _ = xor_data
    out1 = small_net.forward(X)
    out2 = small_net.forward(X)
    np.testing.assert_array_equal(out1, out2)


# ── Gradient check ────────────────────────────────────────────────────────────

def test_gradient_check_relu(small_net, xor_data):
    X, y = xor_data
    rel_err, _, _ = gradient_check(small_net, X, y, layer_idx=0)
    assert rel_err < 1e-4, f"Gradient check failed: rel_error={rel_err:.2e}"

def test_gradient_check_sigmoid(sigmoid_net, xor_data):
    X, y = xor_data
    rel_err, _, _ = gradient_check(sigmoid_net, X, y, layer_idx=0)
    assert rel_err < 1e-4, f"Gradient check failed: rel_error={rel_err:.2e}"


# ── Weight update ─────────────────────────────────────────────────────────────

def test_weights_change_after_update(small_net, xor_data):
    X, y = xor_data
    W_before = small_net.weights[0].copy()
    small_net.forward(X)
    grads = small_net.backward(y)
    small_net.update_weights(grads, lr=0.1)
    assert not np.allclose(W_before, small_net.weights[0])

def test_loss_decreases_after_one_step(small_net, xor_data):
    X, y = xor_data
    pred1 = small_net.forward(X)
    loss1 = mse_loss(pred1, y)
    grads = small_net.backward(y)
    small_net.update_weights(grads, lr=0.05)
    pred2 = small_net.forward(X)
    loss2 = mse_loss(pred2, y)
    assert loss2 < loss1 or loss2 < 1e-8, "Loss did not decrease after gradient step"


# ── Training convergence ──────────────────────────────────────────────────────

def test_training_reduces_loss(xor_data):
    X, y = xor_data
    net = NeuralNetwork([2, 8, 8, 1], activation='relu', seed=42)
    net.train(X, y, epochs=2000, lr=0.05, verbose=False)
    assert net.loss_history[-1] < net.loss_history[0] * 0.5

def test_loss_history_length(xor_data):
    X, y = xor_data
    net = NeuralNetwork([2, 4, 1], activation='relu', seed=0)
    net.train(X, y, epochs=50, lr=0.01, verbose=False)
    assert len(net.loss_history) == 50

def test_deeper_network_trains(xor_data):
    X, y = xor_data
    net = NeuralNetwork([2, 8, 8, 8, 1], activation='relu', seed=1)
    net.train(X, y, epochs=500, lr=0.03, verbose=False)
    assert net.loss_history[-1] < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
