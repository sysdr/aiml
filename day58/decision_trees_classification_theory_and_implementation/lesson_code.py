"""
Day 58: Decision Trees Theory - From Scratch Implementation
Build a decision tree classifier for customer churn prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Node:
    """
    Represents a node in the decision tree.
    
    Attributes:
        feature: Index of feature to split on (None for leaf nodes)
        threshold: Value to compare feature against (None for leaf nodes)
        left: Left child node (samples where feature <= threshold)
        right: Right child node (samples where feature > threshold)
        value: Class label for leaf nodes (None for internal nodes)
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes only
        
    def is_leaf(self):
        """Check if this is a leaf node"""
        return self.value is not None


class DecisionTree:
    """
    Decision Tree Classifier built from scratch.
    
    Uses information gain (entropy reduction) to select optimal splits.
    Supports binary classification problems.
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        """
        Initialize decision tree with hyperparameters.
        
        Args:
            max_depth: Maximum depth of tree (prevents overfitting)
            min_samples_split: Minimum samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """
        Build decision tree from training data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            feature_names: Optional names for features (for visualization)
        """
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(X.shape[1])]
        self.root = self._build_tree(X, y, depth=0)
        
    def _calculate_entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy of a label distribution.
        
        Entropy = -Σ(p_i * log2(p_i)) where p_i is proportion of class i
        
        Args:
            y: Array of class labels
            
        Returns:
            Entropy value (0 = pure, higher = more mixed)
        """
        if len(y) == 0:
            return 0
        
        # Calculate class proportions
        proportions = np.bincount(y) / len(y)
        
        # Remove zero proportions (log(0) is undefined)
        proportions = proportions[proportions > 0]
        
        # Calculate entropy
        entropy = -np.sum(proportions * np.log2(proportions))
        return entropy
    
    def _information_gain(self, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        """
        Calculate information gain from a split.
        
        Information Gain = Parent Entropy - Weighted Average of Children Entropy
        
        Args:
            parent: Parent node labels
            left: Left child labels
            right: Right child labels
            
        Returns:
            Information gain value (higher is better)
        """
        if len(left) == 0 or len(right) == 0:
            return 0
        
        # Calculate parent entropy
        parent_entropy = self._calculate_entropy(parent)
        
        # Calculate weighted average of children entropy
        n = len(parent)
        n_left, n_right = len(left), len(right)
        child_entropy = (n_left / n) * self._calculate_entropy(left) + \
                       (n_right / n) * self._calculate_entropy(right)
        
        # Information gain is reduction in entropy
        return parent_entropy - child_entropy
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """
        Find the best feature and threshold to split on.
        
        Evaluates all possible splits and selects the one with highest information gain.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            
        Returns:
            Tuple of (best_feature_index, best_threshold)
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Try every feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            # Try every unique value as a threshold
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                # Split data
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate information gain
                left_labels = y[left_mask]
                right_labels = y[right_mask]
                gain = self._information_gain(y, left_labels, right_labels)
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively build decision tree.
        
        Stopping conditions:
        1. Maximum depth reached
        2. Node is pure (all same class)
        3. Too few samples to split
        
        Args:
            X: Feature matrix for this node
            y: Labels for this node
            depth: Current depth in tree
            
        Returns:
            Node object (internal node or leaf)
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            # Create leaf node with majority class
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            # No valid split found, create leaf
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Create internal node
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_child, right=right_child)
    
    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """
        Traverse tree to make prediction for a single sample.
        
        Args:
            x: Single sample features
            node: Current node in traversal
            
        Returns:
            Predicted class label
        """
        if node.is_leaf():
            return node.value
        
        # Follow left or right branch based on feature value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for multiple samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def print_tree(self, node: Node = None, depth: int = 0):
        """
        Print tree structure in human-readable format.
        
        Args:
            node: Current node (starts at root)
            depth: Current depth for indentation
        """
        if node is None:
            node = self.root
        
        indent = "  " * depth
        
        if node.is_leaf():
            print(f"{indent}→ Predict: Class {node.value}")
        else:
            feature_name = self.feature_names[node.feature]
            print(f"{indent}└─ {feature_name} <= {node.threshold:.2f}?")
            print(f"{indent}   ├─ YES:")
            self.print_tree(node.left, depth + 2)
            print(f"{indent}   └─ NO:")
            self.print_tree(node.right, depth + 2)


def generate_customer_churn_data(n_samples: int = 1000, random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic customer churn dataset.
    
    Features:
    - login_days: Days since last login
    - features_used: Number of product features actively used
    - support_tickets: Number of support tickets opened
    - account_age_months: Age of account in months
    
    Label:
    - churned: 0 (retained) or 1 (churned)
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (features_df, labels_array)
    """
    np.random.seed(random_state)
    
    # Generate features with realistic patterns
    login_days = np.random.exponential(scale=10, size=n_samples)
    features_used = np.random.poisson(lam=3, size=n_samples)
    support_tickets = np.random.poisson(lam=2, size=n_samples)
    account_age_months = np.random.gamma(shape=2, scale=5, size=n_samples)
    
    # Clip values to realistic ranges
    login_days = np.clip(login_days, 0, 90)
    features_used = np.clip(features_used, 0, 10)
    support_tickets = np.clip(support_tickets, 0, 15)
    account_age_months = np.clip(account_age_months, 1, 60)
    
    # Generate churn labels based on feature patterns
    # High churn probability if: high login_days, low features_used, high support_tickets
    churn_probability = (
        0.7 * (login_days / 90) +  # More days since login → higher churn
        0.2 * (1 - features_used / 10) +  # Fewer features used → higher churn
        0.1 * (support_tickets / 15)  # More support tickets → higher churn
    )
    
    # Add some noise
    churn_probability = np.clip(churn_probability + np.random.normal(0, 0.1, n_samples), 0, 1)
    
    # Convert probabilities to binary labels
    churned = (np.random.random(n_samples) < churn_probability).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'login_days': login_days,
        'features_used': features_used,
        'support_tickets': support_tickets,
        'account_age_months': account_age_months
    })
    
    return df, churned


def visualize_decision_boundary(tree, X, y, feature_names):
    """
    Visualize decision boundaries for first two features.
    
    Args:
        tree: Trained decision tree
        X: Feature matrix
        y: Labels
        feature_names: Names of features
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use only first two features for 2D visualization
    X_plot = X[:, :2]
    
    # Create mesh
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Create feature array for mesh (pad with zeros for other features)
    mesh_features = np.c_[xx.ravel(), yy.ravel()]
    if X.shape[1] > 2:
        mesh_features = np.column_stack([
            mesh_features,
            np.zeros((mesh_features.shape[0], X.shape[1] - 2))
        ])
    
    # Predict on mesh
    Z = tree.predict(mesh_features)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plot training points
    scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y, 
                        cmap='RdYlBu', edgecolor='black', s=50, alpha=0.7)
    
    ax.set_xlabel(feature_names[0], fontsize=12)
    ax.set_ylabel(feature_names[1], fontsize=12)
    ax.set_title('Decision Tree Decision Boundaries', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Class')
    plt.tight_layout()
    plt.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
    print("✓ Decision boundary visualization saved to 'decision_boundary.png'")


def compare_with_sklearn(X_train, X_test, y_train, y_test, feature_names):
    """
    Compare our implementation with scikit-learn's DecisionTreeClassifier.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        feature_names: Feature names
    """
    print("\n" + "="*60)
    print("COMPARISON WITH SCIKIT-LEARN")
    print("="*60)
    
    # Train scikit-learn model
    sklearn_tree = DecisionTreeClassifier(max_depth=10, min_samples_split=2, random_state=42)
    sklearn_tree.fit(X_train, y_train)
    sklearn_pred = sklearn_tree.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    print(f"\nScikit-learn DecisionTreeClassifier Accuracy: {sklearn_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, sklearn_pred, target_names=['Retained', 'Churned']))
    
    # Feature importance
    print("\nFeature Importance (scikit-learn):")
    for name, importance in zip(feature_names, sklearn_tree.feature_importances_):
        print(f"  {name}: {importance:.4f}")


def main():
    """Main execution function"""
    print("="*60)
    print("DAY 58: DECISION TREES THEORY - FROM SCRATCH")
    print("="*60)
    
    # Generate data
    print("\n[1/5] Generating customer churn dataset...")
    X_df, y = generate_customer_churn_data(n_samples=1000, random_state=42)
    feature_names = list(X_df.columns)
    X = X_df.values
    
    print(f"✓ Generated {len(X)} samples with {X.shape[1]} features")
    print(f"✓ Churn rate: {y.mean():.1%}")
    
    # Split data
    print("\n[2/5] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Train our decision tree
    print("\n[3/5] Training decision tree from scratch...")
    tree = DecisionTree(max_depth=5, min_samples_split=10)
    tree.fit(X_train, y_train, feature_names=feature_names)
    print("✓ Tree trained successfully")
    
    # Make predictions
    print("\n[4/5] Evaluating model...")
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {train_accuracy:<15.4f} {test_accuracy:<15.4f}")
    
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, test_pred, target_names=['Retained', 'Churned']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    print(f"  [[True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}]")
    print(f"   [False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}]]")
    
    # Print tree structure
    print("\n" + "="*60)
    print("DECISION TREE STRUCTURE")
    print("="*60)
    tree.print_tree()
    
    # Visualize
    print("\n[5/5] Creating visualizations...")
    visualize_decision_boundary(tree, X_train, y_train, feature_names)
    
    # Compare with scikit-learn
    compare_with_sklearn(X_train, X_test, y_train, y_test, feature_names)
    
    print("\n" + "="*60)
    print("LESSON COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("• Decision trees use recursive splitting to partition data")
    print("• Information gain guides split selection (entropy reduction)")
    print("• Max depth and min samples prevent overfitting")
    print("• Trees are interpretable but can overfit without regularization")
    print("\nNext: Day 59 - Decision Trees with Scikit-learn")


if __name__ == "__main__":
    main()
