"""
MNIST data loading and preprocessing pipeline.
Mirrors production data ingestion patterns: load → normalize → encode.
"""
import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist(cache_dir=None):
    """
    Download and preprocess MNIST.
    Returns normalized float64 arrays split into train/test sets.
    """
    print("Loading MNIST (first run downloads ~12MB)...")
    mnist = fetch_openml(
        'mnist_784',
        version=1,
        as_frame=False,
        data_home=cache_dir,
        parser='liac-arff'
    )
    X, y = mnist.data, mnist.target.astype(int)

    # Normalize pixel values from [0, 255] → [0.0, 1.0]
    # Critical step: keeps gradient magnitudes stable during training
    X = X / 255.0

    # Standard MNIST split: 60k train, 10k test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def one_hot(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded matrix.
    Label 7  →  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    Required for cross-entropy loss computation.
    """
    n = len(labels)
    encoded = np.zeros((n, num_classes))
    encoded[np.arange(n), labels] = 1.0
    return encoded


def get_batch(X, Y, batch_size=32, shuffle=True):
    """
    Generator that yields (X_batch, Y_batch) pairs.
    Shuffling prevents the network from memorising sample order.
    """
    n = X.shape[0]
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield X[idx], Y[idx]
