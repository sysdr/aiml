"""
Day 129: Multi-Layer Perceptrons (MLPs)
=========================================
Building an MLP forward pass from scratch using NumPy only.
Mirrors the architecture used in production systems at Stripe, YouTube, Google.
"""

import numpy as np
import matplotlib.pyplot as plt


# ─── Activation Functions ─────────────────────────────────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit — the default hidden-layer activation."""
    return np.maximum(0, z)

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid — squashes output to (0,1), used for binary classification."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax — multi-class output probabilities that sum to 1."""
    shifted = z - z.max(axis=1, keepdims=True)  # numerical stability
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# ─── MLP Class ────────────────────────────────────────────────────────────────

class MLP:
    """
    Multi-Layer Perceptron with configurable hidden layers.

    Architecture:
        Input → [Hidden(ReLU) × n_layers] → Output(Sigmoid or Softmax)

    Parameters
    ----------
    layer_sizes : list[int]
        Full architecture including input and output dimensions.
        Example: [2, 4, 4, 1] → 2 inputs, two hidden layers of 4, 1 output.
    output_activation : str
        'sigmoid' for binary classification, 'softmax' for multi-class.
    seed : int
        Random seed for reproducible weight initialization.
    """

    def __init__(self, layer_sizes: list, output_activation: str = 'sigmoid', seed: int = 42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.output_activation = output_activation
        self.weights = []
        self.biases = []
        self._init_weights()

    def _init_weights(self):
        """
        He initialization for ReLU layers: scale by sqrt(2 / fan_in).
        Prevents vanishing/exploding gradients at network initialization.
        """
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            W = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * scale
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward propagation: pass data through every layer sequentially.

        Returns predictions and cache (pre/post activations per layer).
        Cache is required by backpropagation tomorrow.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        output : np.ndarray — final predictions
        cache  : list of (Z, A) tuples per layer
        """
        cache = []
        A = X  # current activation starts as raw input

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b  # linear transformation

            is_output = (i == len(self.weights) - 1)
            if is_output:
                A = sigmoid(Z) if self.output_activation == 'sigmoid' else softmax(Z)
            else:
                A = relu(Z)  # hidden layers use ReLU

            cache.append((Z, A))

        return A, cache

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary class predictions (0 or 1) from sigmoid output."""
        probs, _ = self.forward(X)
        return (probs >= threshold).astype(int)

    def summary(self):
        """Print architecture summary."""
        print("\n MLP Architecture Summary")
        print("─" * 40)
        total_params = 0
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            layer_type = "Output" if i == len(self.weights) - 1 else f"Hidden {i+1}"
            params = W.size + b.size
            total_params += params
            print(f"  Layer {i+1} ({layer_type}): {W.shape}  bias: {b.shape}  params: {params}")
        print(f"{'─'*40}")
        print(f"  Total trainable parameters: {total_params}\n")


# ─── XOR Demo ─────────────────────────────────────────────────────────────────

def xor_demo():
    """
    XOR is the canonical example of a problem a single perceptron cannot solve.
    An MLP with one hidden layer can — demonstrating non-linear representation.
    """
    print("=" * 50)
    print("Demo 1: XOR Problem — Why Layers Matter")
    print("=" * 50)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR truth table

    mlp = MLP(layer_sizes=[2, 4, 1], output_activation='sigmoid')
    mlp.summary()

    output, cache = mlp.forward(X)

    print("Forward Pass Results (untrained — random weights):")
    for i, (xi, yi, pred) in enumerate(zip(X, y, output)):
        print(f"  Input: {xi}  Expected: {yi[0]}  Predicted: {pred[0]:.4f}")

    print("\nNote: Predictions near 0.5 — network knows nothing yet.")
    print("Training via backpropagation (Day 130) will shape these values.\n")

    return cache


def fraud_detection_demo():
    """
    Simulates a production fraud detection MLP.
    Input features: [transaction_amount_norm, hour_of_day_norm,
                     geo_velocity, merchant_risk_score, device_match]
    """
    print("=" * 50)
    print("Demo 2: Fraud Detection MLP (Production Simulation)")
    print("=" * 50)

    np.random.seed(0)
    n_samples = 100
    X = np.random.randn(n_samples, 5)  # 5 transaction features

    # Stripe-style architecture: 5 → 16 → 8 → 1
    fraud_model = MLP(layer_sizes=[5, 16, 8, 1], output_activation='sigmoid')
    fraud_model.summary()

    fraud_probs, _ = fraud_model.forward(X)

    high_risk = (fraud_probs >= 0.7).sum()
    medium_risk = ((fraud_probs >= 0.3) & (fraud_probs < 0.7)).sum()
    low_risk = (fraud_probs < 0.3).sum()

    print(f"Processed {n_samples} transactions:")
    print(f"  High Risk   (>0.7): {high_risk}")
    print(f"  Medium Risk (0.3-0.7): {medium_risk}")
    print(f"  Low Risk    (<0.3): {low_risk}")
    print()


def visualize_activations(cache: list):
    """Plot activation distributions across hidden layers."""
    fig, axes = plt.subplots(1, len(cache), figsize=(12, 3))
    if len(cache) == 1:
        axes = [axes]

    for i, (Z, A) in enumerate(cache):
        ax = axes[i]
        label = "Output Layer" if i == len(cache) - 1 else f"Hidden Layer {i+1}"
        ax.hist(A.flatten(), bins=20, color='steelblue', edgecolor='white', alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Count" if i == 0 else "")
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle("Day 129 — MLP Activation Distributions per Layer", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("activation_distributions.png", dpi=150, bbox_inches='tight')
    print("Saved: activation_distributions.png")


if __name__ == "__main__":
    cache = xor_demo()
    fraud_detection_demo()

    print("=" * 50)
    print("Visualizing Layer Activations")
    print("=" * 50)
    visualize_activations(cache)
    print("\nDay 129 complete. Tomorrow: Backpropagation — teaching the network to learn.")
