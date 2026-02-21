"""
Day 128: Activation Functions — Test Suite
Run: python test_lesson.py   or   pytest test_lesson.py -v
"""

import numpy as np
import pytest
from lesson_code import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, get_activation, gradient_check


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def x_1d():
    np.random.seed(0)
    return np.random.randn(10)

@pytest.fixture
def x_2d():
    np.random.seed(1)
    return np.random.randn(5, 4)

@pytest.fixture
def x_mixed():
    return np.array([-3.0, -1.0, 0.0, 1.0, 3.0])


# ── ReLU Tests ────────────────────────────────────────────────────────────────

class TestReLU:
    def test_positive_passthrough(self):
        relu = ReLU()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu(x), x)

    def test_negative_zeroed(self):
        relu = ReLU()
        x = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(relu(x), np.zeros(3))

    def test_zero_boundary(self):
        relu = ReLU()
        assert relu(np.array([0.0]))[0] == 0.0

    def test_output_non_negative(self, x_2d):
        relu = ReLU()
        assert (relu(x_2d) >= 0).all()

    def test_backward_shape(self, x_2d):
        relu = ReLU()
        y = relu.forward(x_2d)
        grad = relu.backward(np.ones_like(y))
        assert grad.shape == x_2d.shape

    def test_gradient_check(self, x_1d):
        relu = ReLU()
        # Avoid exactly 0 (non-differentiable point)
        x = x_1d + 0.1
        passed, err = gradient_check(relu, x)
        assert passed, f"ReLU gradient check failed with error {err:.2e}"


# ── LeakyReLU Tests ───────────────────────────────────────────────────────────

class TestLeakyReLU:
    def test_negative_not_zero(self):
        lrelu = LeakyReLU(alpha=0.01)
        out = lrelu(np.array([-1.0]))
        assert out[0] == pytest.approx(-0.01)

    def test_positive_unchanged(self):
        lrelu = LeakyReLU(alpha=0.01)
        out = lrelu(np.array([5.0]))
        assert out[0] == pytest.approx(5.0)

    def test_gradient_check(self, x_1d):
        lrelu = LeakyReLU()
        x = x_1d + 0.1
        passed, err = gradient_check(lrelu, x)
        assert passed, f"LeakyReLU gradient check failed with error {err:.2e}"


# ── Sigmoid Tests ─────────────────────────────────────────────────────────────

class TestSigmoid:
    def test_output_range(self, x_2d):
        sig = Sigmoid()
        y = sig(x_2d)
        assert (y > 0).all() and (y < 1).all()

    def test_midpoint(self):
        sig = Sigmoid()
        assert sig(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_large_positive_approaches_one(self):
        sig = Sigmoid()
        assert sig(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-6)

    def test_large_negative_approaches_zero(self):
        sig = Sigmoid()
        assert sig(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-6)

    def test_gradient_check(self, x_1d):
        sig = Sigmoid()
        passed, err = gradient_check(sig, x_1d)
        assert passed, f"Sigmoid gradient check failed with error {err:.2e}"


# ── Tanh Tests ────────────────────────────────────────────────────────────────

class TestTanh:
    def test_output_range(self, x_2d):
        tanh = Tanh()
        y = tanh(x_2d)
        assert (y > -1).all() and (y < 1).all()

    def test_zero_center(self):
        tanh = Tanh()
        assert tanh(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_antisymmetry(self, x_1d):
        tanh = Tanh()
        np.testing.assert_allclose(tanh(x_1d), -tanh(-x_1d), atol=1e-10)

    def test_gradient_check(self, x_1d):
        tanh = Tanh()
        passed, err = gradient_check(tanh, x_1d)
        assert passed, f"Tanh gradient check failed with error {err:.2e}"


# ── Softmax Tests ─────────────────────────────────────────────────────────────

class TestSoftmax:
    def test_sums_to_one_1d(self):
        sm = Softmax()
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert sm(x).sum() == pytest.approx(1.0)

    def test_sums_to_one_batched(self, x_2d):
        sm = Softmax()
        y = sm(x_2d)
        np.testing.assert_allclose(y.sum(axis=-1), np.ones(x_2d.shape[0]), atol=1e-10)

    def test_all_positive(self, x_2d):
        sm = Softmax()
        assert (sm(x_2d) > 0).all()

    def test_max_index_preserved(self):
        sm = Softmax()
        x = np.array([0.1, 0.9, 0.3])
        assert np.argmax(sm(x)) == np.argmax(x)

    def test_numerical_stability(self):
        sm = Softmax()
        x = np.array([1000.0, 1001.0, 1002.0])
        y = sm(x)
        assert not np.any(np.isnan(y))
        assert y.sum() == pytest.approx(1.0)


# ── Factory Tests ─────────────────────────────────────────────────────────────

class TestFactory:
    def test_get_relu(self):
        act = get_activation('relu')
        assert isinstance(act, ReLU)

    def test_get_sigmoid(self):
        act = get_activation('sigmoid')
        assert isinstance(act, Sigmoid)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            get_activation('swiglu_xyz')

    def test_case_insensitive(self):
        act = get_activation('TANH')
        assert isinstance(act, Tanh)


# ── PyTorch Comparison ────────────────────────────────────────────────────────

class TestVsPyTorch:
    """Validate our NumPy implementations match PyTorch."""

    def setup_method(self):
        try:
            import torch
            self.torch = torch
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_relu_matches_torch(self, x_2d):
        our = ReLU()(x_2d)
        ref = self.torch.relu(self.torch.tensor(x_2d)).numpy()
        np.testing.assert_allclose(our, ref, atol=1e-6)

    def test_sigmoid_matches_torch(self, x_2d):
        our = Sigmoid()(x_2d)
        ref = self.torch.sigmoid(self.torch.tensor(x_2d)).numpy()
        np.testing.assert_allclose(our, ref, atol=1e-6)

    def test_tanh_matches_torch(self, x_2d):
        our = Tanh()(x_2d)
        ref = self.torch.tanh(self.torch.tensor(x_2d)).numpy()
        np.testing.assert_allclose(our, ref, atol=1e-6)

    def test_softmax_matches_torch(self, x_2d):
        our = Softmax()(x_2d)
        ref = self.torch.softmax(self.torch.tensor(x_2d), dim=-1).numpy()
        np.testing.assert_allclose(our, ref, atol=1e-6)


# ── Direct Run ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import subprocess, sys
    result = subprocess.run([sys.executable, '-m', 'pytest', __file__, '-v', '--tb=short'])
    sys.exit(result.returncode)
