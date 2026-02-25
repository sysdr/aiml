"""
MNISTNetwork: 3-layer fully connected neural network.

Architecture:
  Input(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)

He initialization is used throughout for ReLU layers.
All intermediate values are cached for backpropagation (Days 132-133).
"""
import numpy as np
from activations import relu, relu_derivative, softmax


class MNISTNetwork:
    def __init__(self, seed=42):
        np.random.seed(seed)

        # He initialization: std = sqrt(2 / fan_in)
        # Keeps variance stable through deep ReLU layers
        self.W1 = np.random.randn(784, 128) * np.sqrt(2.0 / 784)
        self.b1 = np.zeros((1, 128))

        self.W2 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b2 = np.zeros((1, 64))

        self.W3 = np.random.randn(64, 10) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros((1, 10))

        # Cache: stores intermediate values needed by backprop
        self.cache = {}

    def forward(self, X):
        """
        Forward pass: propagate input X through all layers.
        Returns softmax probability distribution over 10 classes.
        Saves all pre-activation (z) and post-activation (a) values
        into self.cache for use during backpropagation.
        """
        # Layer 1: 784 → 128
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)

        # Layer 2: 128 → 64
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)

        # Output layer: 64 → 10 (probabilities)
        z3 = a2 @ self.W3 + self.b3
        a3 = softmax(z3)

        # Store everything for backprop
        self.cache = {
            'X': X,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'a3': a3,
        }

        return a3

    def predict(self, X):
        """Return integer class predictions (argmax of softmax output)."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def param_count(self):
        """Total number of trainable parameters."""
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size +
            self.W3.size + self.b3.size
        )
