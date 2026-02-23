"""
Day 129 Test Suite: Multi-Layer Perceptrons
Tests verify mathematical correctness of the MLP implementation.
Run: pytest test_lesson.py -v
"""

import pytest
import numpy as np
from lesson_code import MLP, relu, sigmoid, softmax


# ─── Activation Function Tests ────────────────────────────────────────────────

class TestActivationFunctions:

    def test_relu_positive_passthrough(self):
        z = np.array([1.0, 2.0, 3.0])
        assert np.allclose(relu(z), z)

    def test_relu_negative_zeroed(self):
        z = np.array([-1.0, -0.5, 0.0])
        assert np.allclose(relu(z), [0.0, 0.0, 0.0])

    def test_relu_mixed(self):
        z = np.array([-2.0, 0.0, 3.0])
        assert np.allclose(relu(z), [0.0, 0.0, 3.0])

    def test_sigmoid_zero_input(self):
        assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)

    def test_sigmoid_large_positive(self):
        assert sigmoid(np.array([100.0]))[0] > 0.999

    def test_sigmoid_large_negative(self):
        assert sigmoid(np.array([-100.0]))[0] < 0.001

    def test_sigmoid_output_range(self):
        z = np.linspace(-10, 10, 100)
        out = sigmoid(z)
        assert out.min() > 0.0 and out.max() < 1.0

    def test_softmax_sums_to_one(self):
        z = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 0.5]])
        result = softmax(z)
        assert np.allclose(result.sum(axis=1), [1.0, 1.0])

    def test_softmax_output_positive(self):
        z = np.array([[1.0, -1.0, 0.5]])
        assert (softmax(z) > 0).all()

    def test_softmax_numerical_stability(self):
        z = np.array([[1000.0, 1001.0, 999.0]])
        result = softmax(z)
        assert not np.isnan(result).any()


# ─── MLP Architecture Tests ───────────────────────────────────────────────────

class TestMLPArchitecture:

    def test_weight_shapes_correct(self):
        mlp = MLP([3, 5, 2])
        assert mlp.weights[0].shape == (3, 5)
        assert mlp.weights[1].shape == (5, 2)

    def test_bias_shapes_correct(self):
        mlp = MLP([3, 5, 2])
        assert mlp.biases[0].shape == (1, 5)
        assert mlp.biases[1].shape == (1, 2)

    def test_number_of_layers(self):
        mlp = MLP([2, 4, 4, 1])
        assert len(mlp.weights) == 3
        assert len(mlp.biases) == 3

    def test_parameter_count(self):
        # [2, 4, 1] → W1: 2*4=8, b1: 4, W2: 4*1=4, b2: 1 → total=17
        mlp = MLP([2, 4, 1])
        total = sum(w.size + b.size for w, b in zip(mlp.weights, mlp.biases))
        assert total == 17


# ─── Forward Pass Tests ───────────────────────────────────────────────────────

class TestForwardPass:

    def test_output_shape_binary(self):
        mlp = MLP([3, 5, 1], output_activation='sigmoid')
        X = np.random.randn(10, 3)
        output, _ = mlp.forward(X)
        assert output.shape == (10, 1)

    def test_output_shape_multiclass(self):
        mlp = MLP([4, 8, 3], output_activation='softmax')
        X = np.random.randn(20, 4)
        output, _ = mlp.forward(X)
        assert output.shape == (20, 3)

    def test_sigmoid_output_range(self):
        mlp = MLP([2, 4, 1], output_activation='sigmoid')
        X = np.random.randn(50, 2)
        output, _ = mlp.forward(X)
        assert output.min() > 0.0 and output.max() < 1.0

    def test_softmax_output_sums_to_one(self):
        mlp = MLP([3, 6, 4], output_activation='softmax')
        X = np.random.randn(15, 3)
        output, _ = mlp.forward(X)
        assert np.allclose(output.sum(axis=1), np.ones(15))

    def test_cache_length_matches_layers(self):
        mlp = MLP([2, 4, 4, 1])
        X = np.random.randn(5, 2)
        _, cache = mlp.forward(X)
        assert len(cache) == 3  # 3 weight matrices = 3 layer outputs

    def test_single_sample_forward(self):
        mlp = MLP([2, 3, 1])
        X = np.array([[0.5, -0.3]])
        output, _ = mlp.forward(X)
        assert output.shape == (1, 1)

    def test_hidden_activations_nonnegative(self):
        """ReLU hidden layers must not produce negative activations."""
        mlp = MLP([2, 8, 1])
        X = np.random.randn(100, 2)
        _, cache = mlp.forward(X)
        # Check first hidden layer (index 0, post-activation A)
        A_hidden = cache[0][1]
        assert A_hidden.min() >= 0.0

    def test_deterministic_with_seed(self):
        mlp1 = MLP([3, 5, 1], seed=7)
        mlp2 = MLP([3, 5, 1], seed=7)
        X = np.random.randn(10, 3)
        out1, _ = mlp1.forward(X)
        out2, _ = mlp2.forward(X)
        assert np.allclose(out1, out2)


# ─── Integration Test ─────────────────────────────────────────────────────────

class TestIntegration:

    def test_xor_forward_runs(self):
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        mlp = MLP([2, 4, 1])
        output, cache = mlp.forward(X)
        assert output.shape == (4, 1)
        assert len(cache) == 2

    def test_predict_returns_binary(self):
        mlp = MLP([2, 4, 1])
        X = np.random.randn(20, 2)
        preds = mlp.predict(X)
        unique = set(preds.flatten().tolist())
        assert unique.issubset({0, 1})

    def test_deep_network_forward(self):
        """5-layer network should still produce valid output."""
        mlp = MLP([10, 64, 32, 16, 8, 1])
        X = np.random.randn(50, 10)
        output, _ = mlp.forward(X)
        assert output.shape == (50, 1)
        assert not np.isnan(output).any()
