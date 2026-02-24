"""
Day 130: Backpropagation Algorithm — From Scratch
Implements a fully-connected neural network trained via backprop.
No ML libraries — pure NumPy so every step is visible.
"""

import numpy as np
import matplotlib.pyplot as plt


# ── Activation functions ────────────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def mse_loss(y_pred, y_true):
    return 0.5 * np.mean((y_pred - y_true) ** 2)

def mse_loss_derivative(y_pred, y_true):
    return (y_pred - y_true) / y_true.shape[0]


# ── Neural Network ──────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    A 3-layer MLP trained with backpropagation.
    Architecture: input → hidden1 → hidden2 → output
    """

    def __init__(self, layer_sizes, activation='relu', seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.activation     = relu if activation == 'relu' else sigmoid
        self.activation_der = relu_derivative if activation == 'relu' else sigmoid_derivative

        # He initialization (good default for ReLU)
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

        self.cache = {}
        self.loss_history = []

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, X):
        """
        Propagate X through the network, caching values for backprop.
        Returns final output ŷ.
        """
        a = X
        self.cache['a0'] = a

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W.T + b                         # linear transform
            self.cache[f'z{i+1}'] = z               # cache pre-activation

            # Last layer: linear output (regression); others: activation
            if i < len(self.weights) - 1:
                a = self.activation(z)
            else:
                a = z                               # output layer: no activation

            self.cache[f'a{i+1}'] = a               # cache post-activation

        return a

    # ── Backward pass ───────────────────────────────────────────────────────

    def backward(self, y_true):
        """
        Compute gradients via chain rule, flowing right-to-left.
        Returns dict of gradients for all weights and biases.
        """
        grads = {}
        n_layers = len(self.weights)

        # Gradient of loss w.r.t. output layer (MSE derivative)
        y_pred = self.cache[f'a{n_layers}']
        delta  = mse_loss_derivative(y_pred, y_true)  # shape: (batch, output_size)

        for i in reversed(range(n_layers)):
            a_prev = self.cache[f'a{i}']             # input to this layer
            z_curr = self.cache[f'z{i+1}']           # pre-activation of this layer

            # For output layer: delta is already gradient w.r.t. z (no activation)
            # For hidden layers: delta is gradient w.r.t. a (post-activation)
            # Convert to gradient w.r.t. z by applying activation derivative
            if i < n_layers - 1:  # Not the output layer
                delta = delta * self.activation_der(z_curr)

            # Gradient for weights and biases
            grads[f'dW{i}'] = delta.T @ a_prev       # (out, in)
            grads[f'db{i}'] = delta.sum(axis=0, keepdims=True)

            # Propagate gradient backward (don't propagate past input)
            if i > 0:
                # Multiply by weight matrix to get gradient w.r.t. previous layer's output
                delta = delta @ self.weights[i]

        return grads

    # ── Weight update ────────────────────────────────────────────────────────

    def update_weights(self, grads, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads[f'dW{i}']
            self.biases[i]  -= lr * grads[f'db{i}']

    # ── Training loop ────────────────────────────────────────────────────────

    def train(self, X, y, epochs=1000, lr=0.01, verbose=True):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss   = mse_loss(y_pred, y)
            self.loss_history.append(loss)

            grads = self.backward(y)
            self.update_weights(grads, lr)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)


# ── Gradient Check ──────────────────────────────────────────────────────────

def gradient_check(net, X, y, layer_idx=0, eps=1e-5):
    """
    Verify backprop gradients against numerical finite differences.
    Should produce relative error < 1e-5 for a correct implementation.
    """
    # Analytic gradient from backprop
    net.forward(X)
    grads = net.backward(y)
    analytic = grads[f'dW{layer_idx}'].copy()

    # Numerical gradient (finite differences)
    W = net.weights[layer_idx]
    numerical = np.zeros_like(W)
    for r in range(W.shape[0]):
        for c in range(W.shape[1]):
            W[r, c] += eps
            loss_plus = mse_loss(net.forward(X), y)
            W[r, c] -= 2 * eps
            loss_minus = mse_loss(net.forward(X), y)
            W[r, c] += eps
            numerical[r, c] = (loss_plus - loss_minus) / (2 * eps)

    diff = np.abs(analytic - numerical)
    norm = np.abs(analytic) + np.abs(numerical) + 1e-15
    rel_error = np.max(diff / norm)
    return rel_error, analytic, numerical


# ── Demo ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Day 130: Backpropagation from Scratch")
    print("=" * 60)

    # --- Dataset: XOR problem (not linearly separable — needs hidden layers) ---
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)

    # Build network: 2 inputs → 4 hidden → 4 hidden → 1 output
    net = NeuralNetwork(layer_sizes=[2, 4, 4, 1], activation='relu', seed=0)

    # --- Gradient check before training ---
    print("\n[1] Running gradient check...")
    rel_err, analytic, numerical = gradient_check(net, X, y, layer_idx=0)
    print(f"    Relative error: {rel_err:.2e}  {'✓ PASS' if rel_err < 1e-4 else '✗ FAIL'}")

    # --- Train ---
    print("\n[2] Training on XOR dataset (1000 epochs, lr=0.05)...")
    net = NeuralNetwork(layer_sizes=[2, 4, 4, 1], activation='relu', seed=0)
    net.train(X, y, epochs=1000, lr=0.05, verbose=True)

    # --- Final predictions ---
    print("\n[3] Final predictions vs ground truth:")
    preds = net.predict(X)
    for xi, yi, pi in zip(X, y, preds):
        print(f"    Input: {xi} | Target: {yi[0]:.0f} | Predicted: {pi[0]:.4f}")

    # --- Vanishing gradient demo: sigmoid vs relu ---
    print("\n[4] Activation comparison — Sigmoid vs ReLU (5-layer network):")
    for act in ['sigmoid', 'relu']:
        net_deep = NeuralNetwork([2, 8, 8, 8, 8, 1], activation=act, seed=0)
        net_deep.train(X, y, epochs=500, lr=0.01, verbose=False)
        final_loss = net_deep.loss_history[-1]
        print(f"    {act:8s} final loss: {final_loss:.6f}")

    # --- Plot loss curve ---
    plt.figure(figsize=(8, 4))
    plt.plot(net.loss_history, color='steelblue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Day 130: Loss Convergence During Backpropagation')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    print("\n[5] Loss curve saved to loss_curve.png")
    print("\nDone.")


if __name__ == '__main__':
    main()
