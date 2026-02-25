"""
Day 131 Integration Demo
Demonstrates: data loading → preprocessing → forward pass → shape verification

Run: python lesson_code.py
"""
import numpy as np
from data_loader import load_mnist, one_hot
from network import MNISTNetwork


def main():
    print("\n=== Day 131: MNIST Neural Network — Forward Pass Demo ===\n")

    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test = load_mnist()

    # 2. One-hot encode labels
    Y_train = one_hot(y_train)
    Y_test  = one_hot(y_test)

    print(f"Label encoding check: y=7 → {one_hot(np.array([7]))[0]}")

    # 3. Build network and report size
    net = MNISTNetwork()
    print(f"\nNetwork parameters: {net.param_count():,}")

    # 4. Forward pass on a mini-batch
    batch_X = X_train[:32]
    batch_Y = Y_train[:32]

    probs = net.forward(batch_X)

    # 5. Shape verification
    print("\n--- Shape Verification ---")
    print(f"Input  : {batch_X.shape}   expected (32, 784)")
    print(f"Output : {probs.shape}    expected (32, 10)")
    print(f"Row sums (should be 1.0): {probs.sum(axis=1)[:4].round(6)}")

    # 6. Sample prediction (untrained — random weights)
    sample_idx = 0
    pred = np.argmax(probs[sample_idx])
    true = y_train[sample_idx]
    conf = probs[sample_idx][pred] * 100
    print(f"\nSample 0 — True: {true}  |  Pred: {pred}  |  Confidence: {conf:.1f}%")
    print("(Untrained network — predictions are random)")

    print("\n✓ Forward pass verified. Ready for Day 132: Loss + Backprop.\n")


if __name__ == '__main__':
    main()
