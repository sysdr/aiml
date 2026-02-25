"""
Activation functions and their derivatives.
Used in forward pass (activations) and backward pass (derivatives).
"""
import numpy as np


def relu(z):
    """ReLU: max(0, z). Kills negative values, passes positives unchanged."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Gradient of ReLU: 1 where z > 0, else 0."""
    return (z > 0).astype(float)


def softmax(z):
    """
    Softmax converts raw scores into a probability distribution.
    Subtract max per row for numerical stability (prevents exp overflow).
    Output rows sum to 1.0.
    """
    shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)
