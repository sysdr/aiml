"""
Day 64: Support Vector Machines Theory - From Scratch Implementation
Building intuition for how SVMs find the maximum margin decision boundary
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from scipy.optimize import minimize
import seaborn as sns

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class SimpleSVM:
    """
    Simplified SVM implementation for educational purposes.
    This demonstrates the core SVM concepts without production optimizations.
    
    Real production SVMs (like scikit-learn's) use:
    - Sequential Minimal Optimization (SMO) algorithm
    - Optimized C++ implementations
    - Advanced caching strategies
    
    This implementation focuses on understanding the math.
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma=0.1):
        """
        Initialize SVM classifier.
        
        Args:
            C: Soft margin parameter (higher = stricter classification)
            kernel: 'linear' or 'rbf' (radial basis function)
            gamma: RBF kernel parameter (higher = more curved boundaries)
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None
        
    def _linear_kernel(self, X1, X2):
        """Linear kernel: K(x1, x2) = x1 · x2"""
        return np.dot(X1, X2.T)
    
    def _rbf_kernel(self, X1, X2):
        """
        RBF (Gaussian) kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||²)
        This creates circular decision boundaries.
        
        Used in production for:
        - Image classification (pixel similarity)
        - Text categorization (document similarity)
        - Anomaly detection (distance-based patterns)
        """
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
            
        distances = np.sum(X1**2, axis=1, keepdims=True) + \
                   np.sum(X2**2, axis=1) - \
                   2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * distances)
    
    def _kernel(self, X1, X2):
        """Select kernel function"""
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """
        Train the SVM using simplified quadratic programming.
        
        In production (scikit-learn), this uses:
        - LIBSVM library (optimized C++ implementation)
        - SMO algorithm for efficiency
        - Advanced numerical optimizations
        
        Our version uses scipy.optimize for educational clarity.
        """
        n_samples, n_features = X.shape
        
        # Convert labels to {-1, 1}
        y = np.where(y <= 0, -1, 1)
        
        # Compute kernel matrix
        K = self._kernel(X, X)
        
        # Quadratic programming objective function
        def objective(alpha):
            """
            Minimize: (1/2) * sum(alpha_i * alpha_j * y_i * y_j * K(xi, xj)) - sum(alpha_i)
            This is the dual formulation of the SVM optimization problem.
            """
            return 0.5 * np.sum((alpha * y)[:, None] * (alpha * y) * K) - np.sum(alpha)
        
        # Constraints: 0 <= alpha_i <= C and sum(alpha_i * y_i) = 0
        constraints = [
            {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)},
        ]
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # Initial guess
        alpha0 = np.zeros(n_samples)
        
        # Solve optimization problem
        result = minimize(
            objective,
            alpha0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.alphas = result.x
        
        # Identify support vectors (alpha > threshold)
        sv_threshold = 1e-5
        sv_indices = self.alphas > sv_threshold
        
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_alphas = self.alphas[sv_indices]
        
        # Calculate weights for linear kernel
        if self.kernel == 'linear':
            self.w = np.sum((self.support_alphas * self.support_vector_labels)[:, None] * 
                          self.support_vectors, axis=0)
        
        # Calculate bias term
        # Use support vectors on the margin (0 < alpha < C)
        margin_sv = (self.support_alphas > sv_threshold) & (self.support_alphas < self.C)
        if np.any(margin_sv):
            K_sv = self._kernel(self.support_vectors[margin_sv], self.support_vectors)
            self.b = np.mean(
                self.support_vector_labels[margin_sv] - 
                np.sum((self.support_alphas * self.support_vector_labels) * K_sv, axis=1)
            )
        else:
            self.b = 0
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Decision function: sign(sum(alpha_i * y_i * K(xi, x)) + b)
        """
        K = self._kernel(X, self.support_vectors)
        decision = np.sum((self.support_alphas * self.support_vector_labels) * K, axis=1) + self.b
        return np.sign(decision)
    
    def decision_function(self, X):
        """
        Calculate decision function values (distance from decision boundary).
        
        In production systems:
        - Values > 0: Predict class +1
        - Values < 0: Predict class -1
        - Magnitude: Confidence level (larger = more confident)
        """
        K = self._kernel(X, self.support_vectors)
        return np.sum((self.support_alphas * self.support_vector_labels) * K, axis=1) + self.b


def visualize_decision_boundary(svm, X, y, title="SVM Decision Boundary"):
    """
    Visualize the SVM decision boundary and margin.
    
    This visualization shows:
    - Decision boundary (solid line)
    - Margin boundaries (dashed lines)
    - Support vectors (circled)
    - Training data points
    """
    plt.figure(figsize=(10, 8))
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contourf(xx, yy, Z, levels=[-np.inf, -1, 1, np.inf], 
                alpha=0.3, colors=['red', 'white', 'blue'])
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], 
               colors=['red', 'black', 'blue'],
               linestyles=['dashed', 'solid', 'dashed'],
               linewidths=[2, 3, 2])
    
    # Plot training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', 
               edgecolors='black', s=100, alpha=0.7)
    
    # Highlight support vectors
    if svm.support_vectors is not None:
        plt.scatter(svm.support_vectors[:, 0], 
                   svm.support_vectors[:, 1],
                   s=300, linewidth=3, facecolors='none', 
                   edgecolors='green', label='Support Vectors')
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    return plt


def demo_linear_svm():
    """
    Demonstrate linear SVM on linearly separable data.
    
    Use case: Gmail spam classification
    - Features: word frequencies, sender patterns
    - Linear boundary separates spam vs. legitimate email
    """
    print("=" * 60)
    print("Demo 1: Linear SVM - Spam Email Classification")
    print("=" * 60)
    
    # Generate linearly separable data
    np.random.seed(42)
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, 
                      cluster_std=1.5, random_state=42)
    y = np.where(y == 0, -1, 1)
    
    print(f"\nDataset: {len(X)} emails")
    print(f"Features: [word_frequency_'free', word_frequency_'click']")
    print(f"Classes: Spam ({np.sum(y == 1)}) vs Legitimate ({np.sum(y == -1)})")
    
    # Train linear SVM
    svm = SimpleSVM(C=1.0, kernel='linear')
    svm.fit(X, y)
    
    print(f"\nSupport Vectors: {len(svm.support_vectors)}/{len(X)} emails")
    print("These are the 'borderline' emails that define the classifier")
    
    # Calculate margin width
    if svm.w is not None:
        margin = 2 / np.linalg.norm(svm.w)
        print(f"Margin Width: {margin:.3f}")
        print("Wider margin = more robust to new email variations")
    
    # Test predictions
    predictions = svm.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nTraining Accuracy: {accuracy * 100:.1f}%")
    
    # Visualize
    vis = visualize_decision_boundary(svm, X, y, 
                                     "Linear SVM: Email Spam Classification")
    plt.savefig('linear_svm_demo.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved: linear_svm_demo.png")
    plt.close()


def demo_rbf_svm():
    """
    Demonstrate RBF kernel SVM on non-linear data.
    
    Use case: Tesla pedestrian detection
    - Features: HOG (Histogram of Oriented Gradients) features
    - Circular boundaries around pedestrian patterns
    """
    print("\n" + "=" * 60)
    print("Demo 2: RBF Kernel SVM - Pedestrian Detection")
    print("=" * 60)
    
    # Generate non-linear data (circles)
    np.random.seed(42)
    
    # Inner circle (pedestrians)
    r1 = np.random.randn(50) * 0.3 + 2
    theta1 = np.random.rand(50) * 2 * np.pi
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    y1 = np.ones(50)
    
    # Outer circle (background)
    r2 = np.random.randn(50) * 0.5 + 5
    theta2 = np.random.rand(50) * 2 * np.pi
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    y2 = -np.ones(50)
    
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    
    print(f"\nDataset: {len(X)} image regions")
    print(f"Features: [HOG_orientation, HOG_magnitude]")
    print(f"Classes: Pedestrian ({np.sum(y == 1)}) vs Background ({np.sum(y == -1)})")
    
    # Train RBF SVM
    svm = SimpleSVM(C=1.0, kernel='rbf', gamma=0.5)
    svm.fit(X, y)
    
    print(f"\nSupport Vectors: {len(svm.support_vectors)}/{len(X)} regions")
    print("RBF kernel creates circular decision boundary")
    print("Captures pedestrian shape patterns")
    
    # Test predictions
    predictions = svm.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nTraining Accuracy: {accuracy * 100:.1f}%")
    
    # Visualize
    vis = visualize_decision_boundary(svm, X, y,
                                     "RBF SVM: Pedestrian Detection (Tesla Autopilot)")
    plt.savefig('rbf_svm_demo.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved: rbf_svm_demo.png")
    plt.close()


def demo_soft_margin():
    """
    Demonstrate soft margin SVM with different C values.
    
    Use case: Airbnb fraud detection
    - Different C values for different tolerance levels
    - Balance between catching fraud and false alarms
    """
    print("\n" + "=" * 60)
    print("Demo 3: Soft Margin Tuning - Airbnb Fraud Detection")
    print("=" * 60)
    
    # Generate data with noise/outliers
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               flip_y=0.1, class_sep=1.5, random_state=42)
    y = np.where(y == 0, -1, 1)
    
    print(f"\nDataset: {len(X)} property listings")
    print(f"Features: [price_deviation, review_pattern_score]")
    print(f"Classes: Fraudulent ({np.sum(y == 1)}) vs Legitimate ({np.sum(y == -1)})")
    print("\nNote: Dataset includes outliers (noisy data)")
    
    # Compare different C values
    C_values = [0.1, 1.0, 10.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, C in enumerate(C_values):
        svm = SimpleSVM(C=C, kernel='linear')
        svm.fit(X, y)
        
        predictions = svm.predict(X)
        accuracy = np.mean(predictions == y)
        
        print(f"\nC = {C}:")
        print(f"  Support Vectors: {len(svm.support_vectors)}/{len(X)}")
        print(f"  Accuracy: {accuracy * 100:.1f}%")
        
        if C == 0.1:
            print("  → Wide margin, tolerates outliers")
            print("  → Best for high false positive tolerance")
        elif C == 1.0:
            print("  → Balanced approach")
            print("  → Production default for most cases")
        else:
            print("  → Strict classification, narrow margin")
            print("  → Risk of overfitting to outliers")
        
        # Visualize on subplot
        plt.sca(axes[idx])
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[idx].contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf],
                          alpha=0.3, colors=['red', 'blue'])
        axes[idx].contour(xx, yy, Z, levels=[0],
                         colors='black', linestyles='solid', linewidths=3)
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='bwr',
                         edgecolors='black', s=100, alpha=0.7)
        axes[idx].scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                         s=300, linewidth=3, facecolors='none',
                         edgecolors='green')
        
        axes[idx].set_title(f'C = {C}\n{len(svm.support_vectors)} Support Vectors',
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Price Deviation')
        axes[idx].set_ylabel('Review Pattern Score')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('soft_margin_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved: soft_margin_comparison.png")
    plt.close()


def compare_svm_vs_knn():
    """
    Compare SVM vs KNN to understand when to use each.
    
    Key insights:
    - SVMs: Better for high-dimensional data, smaller datasets
    - KNN: Better for local patterns, irregular boundaries
    """
    print("\n" + "=" * 60)
    print("Demo 4: SVM vs KNN Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               class_sep=2.0, random_state=42)
    y = np.where(y == 0, -1, 1)
    
    # Train SVM
    svm = SimpleSVM(C=1.0, kernel='linear')
    svm.fit(X, y)
    
    print("\nSVM Characteristics:")
    print(f"  Support Vectors: {len(svm.support_vectors)}/{len(X)} samples")
    print(f"  Memory: Stores only {len(svm.support_vectors)} points")
    print(f"  Decision: Global boundary optimization")
    print(f"  Best for: High dimensions, clear separation")
    
    print("\nKNN Characteristics (from Day 63):")
    print(f"  Stored Data: All {len(X)} samples")
    print(f"  Memory: Stores entire dataset")
    print(f"  Decision: Local neighborhood voting")
    print(f"  Best for: Irregular boundaries, local patterns")
    
    print("\nProduction Guidelines:")
    print("  Use SVM when:")
    print("    - High-dimensional data (text, genomics)")
    print("    - Clear class separation expected")
    print("    - Memory efficiency matters")
    print("    - Need interpretable decision boundary")
    print("\n  Use KNN when:")
    print("    - Irregular, complex decision boundaries")
    print("    - Local patterns more important than global")
    print("    - Continuously updating data")
    print("    - No training time constraint")


if __name__ == "__main__":
    print("\n🎯 Day 64: Support Vector Machines Theory")
    print("Building Maximum Margin Classifiers from Scratch\n")
    
    # Run all demonstrations
    demo_linear_svm()
    demo_rbf_svm()
    demo_soft_margin()
    compare_svm_vs_knn()
    
    print("\n" + "=" * 60)
    print("✅ All Demos Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. SVMs find the maximum margin decision boundary")
    print("2. Only support vectors matter - most data is redundant")
    print("3. Kernel trick handles non-linear boundaries efficiently")
    print("4. Soft margin (C parameter) balances accuracy vs generalization")
    print("5. Mathematical guarantees make SVMs production-reliable")
    print("\nNext: Day 65 - SVMs with Scikit-learn")
    print("We'll use optimized production implementations for real-world scale")
