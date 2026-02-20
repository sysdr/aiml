"""
Day 127: The Perceptron and Its Limitations
180-Day AI/ML Course — Module 4: Deep Learning

What this file covers:
  1. Perceptron class: forward pass, weight update (Rosenblatt rule)
  2. Training on AND gate (linearly separable — succeeds)
  3. Training on XOR gate (NOT linearly separable — fails by design)
  4. Decision boundary visualization for both cases
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# The Perceptron
# =============================================================================
class Perceptron:
    """
    Single-layer binary perceptron with Rosenblatt's update rule.
    This is the atomic building block of every dense layer in modern neural nets.
    """

    def __init__(self, n_inputs: int, learning_rate: float = 0.1, max_epochs: int = 100):
        # Small random init — avoids symmetry breaking issues in deeper nets too
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.training_errors = []  # track convergence

    def _step(self, z: float) -> int:
        """Heaviside step activation: fires (1) if z >= 0, silent (0) otherwise."""
        return 1 if z >= 0 else 0

    def predict(self, x: np.ndarray) -> int:
        """Forward pass: weighted sum → step activation → binary decision."""
        z = np.dot(self.weights, x) + self.bias
        return self._step(z)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict(x) for x in X])

    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """
        Rosenblatt's perceptron learning algorithm.
        Converges in finite steps IF data is linearly separable (guaranteed).
        If not separable, oscillates forever — you'll see this with XOR.
        """
        for epoch in range(self.max_epochs):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction  # 0, +1, or -1

                # Weight update: move in direction that reduces error
                self.weights += self.lr * error * xi
                self.bias   += self.lr * error
                errors += int(error != 0)

            self.training_errors.append(errors)

            if verbose:
                print(f"  Epoch {epoch+1:3d}: errors={errors}, "
                      f"weights={self.weights}, bias={self.bias:.2f}")

            # Early stop: convergence reached (Rosenblatt Convergence Theorem)
            if errors == 0:
                print(f"\n✓ Converged at epoch {epoch+1}!")
                return

        print(f"\n✗ Did not converge after {self.max_epochs} epochs "
              f"(final errors={errors}) — data is NOT linearly separable.")


# =============================================================================
# Datasets
# =============================================================================
def get_and_data():
    """AND gate — linearly separable."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    return X, y

def get_xor_data():
    """XOR gate — NOT linearly separable. The perceptron's known weakness."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y


# =============================================================================
# Visualization
# =============================================================================
def plot_decision_boundary(perceptron, X, y, title, ax):
    """
    Draw learned decision boundary: w0*x0 + w1*x1 + bias = 0
    Solve for x1: x1 = -(w0*x0 + bias) / w1
    """
    colors = ['#4A90D9' if label == 0 else '#E87722' for label in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=200, zorder=5, edgecolors='white', linewidth=1.5)

    # Annotate points
    labels_map = {(0,0):'(0,0)', (0,1):'(0,1)', (1,0):'(1,0)', (1,1):'(1,1)'}
    for xi, yi_val in zip(X, y):
        ax.annotate(f"  {labels_map[tuple(xi.astype(int))]}→{yi_val}",
                    xi, fontsize=9, color='#333333')

    # Decision boundary line (only if w1 != 0)
    w = perceptron.weights
    b = perceptron.bias
    if abs(w[1]) > 1e-6:
        x_range = np.linspace(-0.5, 1.5, 200)
        x1_boundary = -(w[0] * x_range + b) / w[1]
        ax.plot(x_range, x1_boundary, color='#2ECC71', linewidth=2.5,
                linestyle='--', label=f'Decision boundary\n{w[0]:.2f}x₀ + {w[1]:.2f}x₁ + {b:.2f} = 0')
        ax.legend(fontsize=8, loc='upper left')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Input x₀")
    ax.set_ylabel("Input x₁")
    ax.set_facecolor('#F9F9F9')
    ax.grid(True, alpha=0.3)

    patch0 = mpatches.Patch(color='#4A90D9', label='Output: 0')
    patch1 = mpatches.Patch(color='#E87722', label='Output: 1')
    ax.legend(handles=[patch0, patch1], loc='upper right', fontsize=8)


def plot_training_errors(p_and, p_xor, ax):
    ax.plot(p_and.training_errors, color='#2ECC71', linewidth=2.5, marker='o',
            markersize=4, label='AND (converges)')
    ax.plot(p_xor.training_errors, color='#E87722', linewidth=2.5, marker='s',
            markersize=4, linestyle='--', label='XOR (oscillates)')
    ax.set_title("Training Errors per Epoch", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassifications")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#F9F9F9')


# =============================================================================
# Main Demo
# =============================================================================
def main():
    print("=" * 60)
    print("  Day 127: The Perceptron and Its Limitations")
    print("=" * 60)

    # --- AND Gate (should converge) ---
    print("\n[1] Training on AND gate (linearly separable)")
    print("-" * 40)
    X_and, y_and = get_and_data()
    p_and = Perceptron(n_inputs=2, learning_rate=0.1, max_epochs=50)
    p_and.train(X_and, y_and)

    preds_and = p_and.predict_batch(X_and)
    accuracy_and = np.mean(preds_and == y_and)
    print(f"\nFinal predictions: {preds_and}")
    print(f"Expected:          {y_and}")
    print(f"Accuracy:          {accuracy_and*100:.1f}%")

    # --- XOR Gate (should NOT converge) ---
    print("\n[2] Training on XOR gate (NOT linearly separable)")
    print("-" * 40)
    X_xor, y_xor = get_xor_data()
    p_xor = Perceptron(n_inputs=2, learning_rate=0.1, max_epochs=50)
    p_xor.train(X_xor, y_xor, verbose=False)  # suppress epoch spam

    preds_xor = p_xor.predict_batch(X_xor)
    accuracy_xor = np.mean(preds_xor == y_xor)
    print(f"Final predictions: {preds_xor}")
    print(f"Expected:          {y_xor}")
    print(f"Best accuracy:     {accuracy_xor*100:.1f}% (max possible with 1 neuron: 75%)")

    # --- Visualize ---
    print("\n[3] Generating visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle("Day 127: The Perceptron — What It Can and Cannot Learn",
                 fontsize=14, fontweight='bold', y=1.02)

    plot_decision_boundary(p_and, X_and, y_and, "AND Gate\n(Perceptron Succeeds ✓)", axes[0])
    plot_decision_boundary(p_xor, X_xor, y_xor, "XOR Gate\n(Perceptron Fails ✗)", axes[1])
    plot_training_errors(p_and, p_xor, axes[2])

    plt.tight_layout()
    plt.savefig("perceptron_results.png", dpi=150, bbox_inches='tight')
    print("  Saved: perceptron_results.png")
    plt.show()

    print("\n[Key Insight]")
    print("The perceptron is guaranteed to find ANY linear boundary that exists.")
    print("But XOR has no linear boundary. That's not a bug — it's a fundamental limit.")
    print("The fix: stack multiple perceptrons (hidden layers). That's Day 129+.")


if __name__ == "__main__":
    main()
