"""
Day 62: K-Nearest Neighbors (KNN) Theory
Implementing KNN from scratch to understand the algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import seaborn as sns

class KNNClassifier:
    """
    K-Nearest Neighbors Classifier implemented from scratch
    
    This implementation helps understand:
    1. Distance calculation methods
    2. Finding nearest neighbors
    3. Majority voting
    4. The impact of K parameter
    """
    
    def __init__(self, k=5, distance_metric='euclidean'):
        """
        Initialize KNN classifier
        
        Args:
            k: Number of neighbors to consider
            distance_metric: 'euclidean' or 'manhattan'
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Store training data (KNN is a lazy learner)
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        print(f"Stored {len(X)} training examples")
    
    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two points
        
        Args:
            x1: First point
            x2: Second point
            
        Returns:
            Distance value
        """
        if self.distance_metric == 'euclidean':
            # Straight-line distance: sqrt(sum of squared differences)
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            # Grid distance: sum of absolute differences
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _get_neighbors(self, x):
        """
        Find K nearest neighbors for a single point
        
        Args:
            x: Query point
            
        Returns:
            Indices of K nearest neighbors
        """
        # Calculate distances to all training points
        distances = []
        for idx, train_point in enumerate(self.X_train):
            dist = self._calculate_distance(x, train_point)
            distances.append((dist, idx))
        
        # Sort by distance and get K nearest
        distances.sort(key=lambda x: x[0])
        k_nearest_indices = [idx for _, idx in distances[:self.k]]
        
        return k_nearest_indices
    
    def predict_single(self, x):
        """
        Predict class for a single point
        
        Args:
            x: Query point
            
        Returns:
            Predicted class label
        """
        # Find K nearest neighbors
        neighbor_indices = self._get_neighbors(x)
        
        # Get their labels
        neighbor_labels = [self.y_train[idx] for idx in neighbor_indices]
        
        # Majority vote
        most_common = Counter(neighbor_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        """
        Predict classes for multiple points
        
        Args:
            X: Query points (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)
    
    def score(self, X, y):
        """
        Calculate accuracy on test data
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy score (0-1)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


def demonstrate_distance_metrics():
    """Show how different distance metrics affect KNN"""
    print("\n" + "="*60)
    print("DEMONSTRATION 1: Distance Metrics Comparison")
    print("="*60)
    
    # Create two points
    point1 = np.array([0, 0])
    point2 = np.array([3, 4])
    
    # Euclidean distance (straight line)
    euclidean = np.sqrt(np.sum((point2 - point1) ** 2))
    print(f"\nPoint 1: {point1}")
    print(f"Point 2: {point2}")
    print(f"Euclidean Distance: {euclidean:.2f} (straight line)")
    
    # Manhattan distance (grid)
    manhattan = np.sum(np.abs(point2 - point1))
    print(f"Manhattan Distance: {manhattan:.2f} (grid/taxi distance)")
    
    print("\nReal-world analogy:")
    print("- Euclidean: 'As the crow flies' distance")
    print("- Manhattan: Walking distance in a city with grid streets")


def demonstrate_k_selection():
    """Show impact of different K values"""
    print("\n" + "="*60)
    print("DEMONSTRATION 2: Impact of K Parameter")
    print("="*60)
    
    # Generate simple 2D dataset
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Test different K values
    k_values = [1, 3, 5, 10, 20]
    results = []
    
    print("\nTesting different K values:")
    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        results.append((k, accuracy))
        print(f"K={k:2d}: Accuracy = {accuracy:.3f}")
    
    # Find best K
    best_k, best_acc = max(results, key=lambda x: x[1])
    print(f"\nBest K value: {best_k} (accuracy: {best_acc:.3f})")
    
    # Visualize decision boundaries for different K values
    visualize_k_comparison(X_train, y_train, X_test, y_test, [1, 5, 20])


def visualize_k_comparison(X_train, y_train, X_test, y_test, k_values):
    """Visualize decision boundaries for different K values"""
    fig, axes = plt.subplots(1, len(k_values), figsize=(15, 4))
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        # Train KNN
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        
        # Create mesh for decision boundary
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        
        # Predict on mesh
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        
        # Plot training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                  cmap='RdYlBu', edgecolor='black', s=50, label='Train')
        
        # Plot test points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                  cmap='RdYlBu', marker='s', edgecolor='black', 
                  s=100, alpha=0.7, label='Test')
        
        accuracy = knn.score(X_test, y_test)
        ax.set_title(f'K={k}, Accuracy={accuracy:.2f}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('knn_k_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Saved visualization: knn_k_comparison.png")


def demonstrate_iris_classification():
    """Real-world example: Iris flower classification"""
    print("\n" + "="*60)
    print("DEMONSTRATION 3: Iris Flower Classification")
    print("="*60)
    
    # Load famous iris dataset
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target  # Use only 2 features for visualization
    
    print(f"\nDataset: {len(X)} iris flowers")
    print(f"Features: {iris.feature_names[:2]}")
    print(f"Classes: {iris.target_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train KNN with K=5
    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
    
    # Evaluate
    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)
    
    print(f"\nResults with K=5:")
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    
    # Make prediction on a new flower
    new_flower = np.array([[5.0, 3.0]])  # New measurements
    prediction = knn.predict_single(new_flower)
    print(f"\nNew flower prediction:")
    print(f"Measurements: sepal_length={new_flower[0][0]}, sepal_width={new_flower[0][1]}")
    print(f"Predicted species: {iris.target_names[prediction]}")
    
    # Visualize
    visualize_iris_classification(X_train, y_train, X_test, y_test, new_flower, prediction)


def visualize_iris_classification(X_train, y_train, X_test, y_test, new_point, new_pred):
    """Visualize iris classification results"""
    plt.figure(figsize=(10, 6))
    
    # Train KNN
    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
    
    # Create decision boundary mesh
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot training data
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                         cmap='viridis', edgecolor='black', s=100,
                         label='Training Data')
    
    # Plot test data
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
               cmap='viridis', marker='s', edgecolor='red',
               s=150, linewidth=2, label='Test Data')
    
    # Plot new prediction
    plt.scatter(new_point[0][0], new_point[0][1], c=new_pred,
               cmap='viridis', marker='*', edgecolor='gold',
               s=500, linewidth=3, label='New Prediction')
    
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('KNN Iris Classification (K=5)')
    plt.legend()
    plt.colorbar(scatter, label='Species')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('knn_iris_classification.png', dpi=150, bbox_inches='tight')
    print("📊 Saved visualization: knn_iris_classification.png")


def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("DAY 62: K-NEAREST NEIGHBORS (KNN) THEORY")
    print("Understanding the Algorithm Behind Recommendations")
    print("="*60)
    
    # Demonstration 1: Distance metrics
    demonstrate_distance_metrics()
    
    # Demonstration 2: Impact of K
    demonstrate_k_selection()
    
    # Demonstration 3: Real-world classification
    demonstrate_iris_classification()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("✓ KNN stores all training data (lazy learning)")
    print("✓ Distance metrics matter: Euclidean vs Manhattan")
    print("✓ K parameter balances noise vs generalization")
    print("✓ Small K → sensitive to outliers")
    print("✓ Large K → overly generic predictions")
    print("✓ Used in Netflix, Spotify, Amazon recommendations")
    print("\n🎯 Tomorrow: Implementing KNN with scikit-learn!")


if __name__ == "__main__":
    main()
