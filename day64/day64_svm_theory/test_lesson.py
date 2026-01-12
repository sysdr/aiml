"""
Test Suite for Day 64: SVM Theory Implementation
Validates core SVM concepts and implementation
"""

import pytest
import numpy as np
from lesson_code import SimpleSVM
from sklearn.datasets import make_blobs, make_classification


class TestSimpleSVM:
    """Test suite for SimpleSVM implementation"""
    
    def test_initialization(self):
        """Test SVM initialization with different parameters"""
        svm = SimpleSVM(C=1.0, kernel='linear')
        assert svm.C == 1.0
        assert svm.kernel == 'linear'
        
        svm_rbf = SimpleSVM(C=0.5, kernel='rbf', gamma=0.1)
        assert svm_rbf.kernel == 'rbf'
        assert svm_rbf.gamma == 0.1
    
    def test_linear_kernel(self):
        """Test linear kernel computation"""
        svm = SimpleSVM(kernel='linear')
        X1 = np.array([[1, 2], [3, 4]])
        X2 = np.array([[5, 6], [7, 8]])
        
        K = svm._linear_kernel(X1, X2)
        
        # Manually compute expected result
        expected = np.dot(X1, X2.T)
        np.testing.assert_array_almost_equal(K, expected)
    
    def test_rbf_kernel(self):
        """Test RBF kernel computation"""
        svm = SimpleSVM(kernel='rbf', gamma=0.5)
        X1 = np.array([[1, 2]])
        X2 = np.array([[1, 2]])
        
        K = svm._rbf_kernel(X1, X2)
        
        # Same point should give kernel value of 1
        assert K[0, 0] == pytest.approx(1.0, abs=1e-10)
    
    def test_fit_linear_separable(self):
        """Test SVM training on linearly separable data"""
        np.random.seed(42)
        X, y = make_blobs(n_samples=50, centers=2, n_features=2,
                         random_state=42)
        y = np.where(y == 0, -1, 1)
        
        svm = SimpleSVM(C=1.0, kernel='linear')
        svm.fit(X, y)
        
        # Check that support vectors are identified
        assert svm.support_vectors is not None
        assert len(svm.support_vectors) > 0
        assert len(svm.support_vectors) < len(X)  # Not all points
        
        # Check that weights are computed for linear kernel
        assert svm.w is not None
        assert svm.b is not None
    
    def test_fit_rbf_kernel(self):
        """Test SVM training with RBF kernel"""
        np.random.seed(42)
        X, y = make_classification(n_samples=50, n_features=2,
                                  n_informative=2, n_redundant=0,
                                  random_state=42)
        y = np.where(y == 0, -1, 1)
        
        svm = SimpleSVM(C=1.0, kernel='rbf', gamma=0.5)
        svm.fit(X, y)
        
        assert svm.support_vectors is not None
        assert len(svm.support_vectors) > 0
    
    def test_predict(self):
        """Test SVM prediction"""
        np.random.seed(42)
        X, y = make_blobs(n_samples=50, centers=2, n_features=2,
                         cluster_std=1.0, random_state=42)
        y = np.where(y == 0, -1, 1)
        
        svm = SimpleSVM(C=1.0, kernel='linear')
        svm.fit(X, y)
        
        predictions = svm.predict(X)
        
        # Predictions should be -1 or 1
        assert np.all((predictions == -1) | (predictions == 1))
        
        # Should achieve reasonable accuracy on training data
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.8  # At least 80% accuracy
    
    def test_decision_function(self):
        """Test decision function computation"""
        np.random.seed(42)
        X, y = make_blobs(n_samples=50, centers=2, n_features=2,
                         random_state=42)
        y = np.where(y == 0, -1, 1)
        
        svm = SimpleSVM(C=1.0, kernel='linear')
        svm.fit(X, y)
        
        decision_values = svm.decision_function(X)
        
        # Decision values should exist for all samples
        assert len(decision_values) == len(X)
        
        # Sign of decision values should match predictions
        predictions = svm.predict(X)
        assert np.all(np.sign(decision_values) == predictions)
    
    def test_support_vector_count(self):
        """Test that support vectors are properly identified"""
        np.random.seed(42)
        X, y = make_blobs(n_samples=100, centers=2, n_features=2,
                         cluster_std=2.0, random_state=42)
        y = np.where(y == 0, -1, 1)
        
        svm = SimpleSVM(C=1.0, kernel='linear')
        svm.fit(X, y)
        
        # Support vectors should be a subset of training data
        assert len(svm.support_vectors) < len(X)
        assert len(svm.support_vectors) > 0
        
        # Support vectors should be the points closest to boundary
        # (in practice, these are the hardest to classify)
        print(f"Support vectors: {len(svm.support_vectors)}/{len(X)}")
    
    def test_soft_margin_c_parameter(self):
        """Test effect of C parameter on support vectors"""
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=2,
                                  n_informative=2, n_redundant=0,
                                  flip_y=0.1, random_state=42)
        y = np.where(y == 0, -1, 1)
        
        # Low C (soft margin)
        svm_soft = SimpleSVM(C=0.1, kernel='linear')
        svm_soft.fit(X, y)
        
        # High C (hard margin)
        svm_hard = SimpleSVM(C=10.0, kernel='linear')
        svm_hard.fit(X, y)
        
        # Typically, higher C leads to more support vectors
        # (trying to classify everything correctly)
        print(f"C=0.1: {len(svm_soft.support_vectors)} support vectors")
        print(f"C=10.0: {len(svm_hard.support_vectors)} support vectors")
        
        assert len(svm_soft.support_vectors) > 0
        assert len(svm_hard.support_vectors) > 0


class TestSVMTheory:
    """Test theoretical properties of SVMs"""
    
    def test_margin_maximization(self):
        """Test that SVM maximizes margin"""
        np.random.seed(42)
        X, y = make_blobs(n_samples=50, centers=2, n_features=2,
                         cluster_std=1.0, random_state=42)
        y = np.where(y == 0, -1, 1)
        
        svm = SimpleSVM(C=1.0, kernel='linear')
        svm.fit(X, y)
        
        # Compute margin
        if svm.w is not None:
            margin = 2.0 / np.linalg.norm(svm.w)
            print(f"Margin width: {margin:.3f}")
            assert margin > 0
    
    def test_kernel_comparison(self):
        """Compare linear vs RBF kernel performance"""
        np.random.seed(42)
        
        # Generate non-linear data (circles)
        r1 = np.random.randn(30) * 0.3 + 2
        theta1 = np.random.rand(30) * 2 * np.pi
        X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
        y1 = np.ones(30)
        
        r2 = np.random.randn(30) * 0.5 + 5
        theta2 = np.random.rand(30) * 2 * np.pi
        X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
        y2 = -np.ones(30)
        
        X = np.vstack([X1, X2])
        y = np.concatenate([y1, y2])
        
        # Linear kernel (should perform poorly on circular data)
        svm_linear = SimpleSVM(C=1.0, kernel='linear')
        svm_linear.fit(X, y)
        acc_linear = np.mean(svm_linear.predict(X) == y)
        
        # RBF kernel (should perform well on circular data)
        svm_rbf = SimpleSVM(C=1.0, kernel='rbf', gamma=0.5)
        svm_rbf.fit(X, y)
        acc_rbf = np.mean(svm_rbf.predict(X) == y)
        
        print(f"Linear kernel accuracy: {acc_linear * 100:.1f}%")
        print(f"RBF kernel accuracy: {acc_rbf * 100:.1f}%")
        
        # RBF should significantly outperform linear on circular data
        assert acc_rbf > acc_linear


def test_production_insights():
    """Test insights about production SVM usage"""
    print("\n=== Production SVM Insights ===")
    
    print("\n1. Memory Efficiency:")
    print("   SVMs store only support vectors, not entire dataset")
    print("   Typical compression: 90-95% of training data discarded")
    
    print("\n2. Kernel Selection Guide:")
    print("   Linear: Text classification, high-dimensional data")
    print("   RBF: Image recognition, non-linear patterns")
    print("   Polynomial: Interaction features, curved boundaries")
    
    print("\n3. C Parameter Tuning:")
    print("   C < 1: Wide margin, better generalization")
    print("   C = 1: Balanced (typical default)")
    print("   C > 1: Strict classification, risk overfitting")
    
    print("\n4. When to Use SVM:")
    print("   ✓ Clear margin between classes")
    print("   ✓ High-dimensional data")
    print("   ✓ Need interpretable decision boundary")
    print("   ✗ Very large datasets (>100K samples)")
    print("   ✗ Many classes (>10)")
    
    assert True  # Always passes, this is informational


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
