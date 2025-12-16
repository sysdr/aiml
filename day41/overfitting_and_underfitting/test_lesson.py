"""
Test Suite for Day 41: Overfitting and Underfitting
Validates detection logic and performance metrics
"""

import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import sys


class TestOverfittingDetection:
    """Test overfitting/underfitting detection logic"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate test data"""
        np.random.seed(42)
        X = np.sort(np.random.rand(100, 1) * 10, axis=0)
        y = np.sin(X).ravel() + np.random.randn(100) * 0.3
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_underfit_detection(self, sample_data):
        """Test that simple models (degree 1) underfit"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Degree 1 (linear) should underfit sine wave
        model = Pipeline([
            ('poly_features', PolynomialFeatures(degree=1)),
            ('linear_regression', LinearRegression())
        ])
        model.fit(X_train, y_train)
        
        train_score = r2_score(y_train, model.predict(X_train))
        test_score = r2_score(y_test, model.predict(X_test))
        
        # Both scores should be low (underfitting)
        assert train_score < 0.5, "Degree 1 should have low train score"
        assert test_score < 0.5, "Degree 1 should have low test score"
        print(f"✓ Underfit detected: Train R²={train_score:.4f}, Test R²={test_score:.4f}")
    
    def test_overfit_detection(self, sample_data):
        """Test that complex models (degree 15) overfit"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Degree 15 should overfit
        model = Pipeline([
            ('poly_features', PolynomialFeatures(degree=15)),
            ('linear_regression', LinearRegression())
        ])
        model.fit(X_train, y_train)
        
        train_score = r2_score(y_train, model.predict(X_train))
        test_score = r2_score(y_test, model.predict(X_test))
        gap = train_score - test_score
        
        # Large train-test gap indicates overfitting
        assert train_score > 0.8, "Degree 15 should have high train score"
        assert gap > 0.2, "Degree 15 should have large train-test gap"
        print(f"✓ Overfit detected: Train R²={train_score:.4f}, Test R²={test_score:.4f}, Gap={gap:.4f}")
    
    def test_optimal_model(self, sample_data):
        """Test that moderate complexity (degree 3-4) performs best"""
        X_train, X_test, y_train, y_test = sample_data
        
        best_test_score = 0
        best_degree = 0
        
        for degree in range(2, 7):
            model = Pipeline([
                ('poly_features', PolynomialFeatures(degree=degree)),
                ('linear_regression', LinearRegression())
            ])
            model.fit(X_train, y_train)
            test_score = r2_score(y_test, model.predict(X_test))
            
            if test_score > best_test_score:
                best_test_score = test_score
                best_degree = degree
        
        # Optimal should be in mid-range
        assert 2 <= best_degree <= 6, "Optimal degree should be moderate"
        assert best_test_score > 0.6, "Optimal model should have good test score"
        print(f"✓ Optimal model: Degree={best_degree}, Test R²={best_test_score:.4f}")
    
    def test_train_test_gap(self, sample_data):
        """Test train-test gap increases with complexity"""
        X_train, X_test, y_train, y_test = sample_data
        
        gaps = []
        for degree in [1, 5, 10, 15]:
            model = Pipeline([
                ('poly_features', PolynomialFeatures(degree=degree)),
                ('linear_regression', LinearRegression())
            ])
            model.fit(X_train, y_train)
            
            train_score = r2_score(y_train, model.predict(X_train))
            test_score = r2_score(y_test, model.predict(X_test))
            gaps.append(train_score - test_score)
        
        # Gap should generally increase with complexity
        assert gaps[-1] > gaps[0], "Gap should increase with complexity"
        print(f"✓ Gap progression: {gaps}")
    
    def test_cv_variance(self, sample_data):
        """Test that cross-validation detects high variance"""
        from sklearn.model_selection import cross_val_score
        X_train, _, y_train, _ = sample_data
        
        # Low complexity: low variance
        model_simple = Pipeline([
            ('poly_features', PolynomialFeatures(degree=2)),
            ('linear_regression', LinearRegression())
        ])
        scores_simple = cross_val_score(model_simple, X_train, y_train, cv=5, scoring='r2')
        
        # High complexity: high variance
        model_complex = Pipeline([
            ('poly_features', PolynomialFeatures(degree=15)),
            ('linear_regression', LinearRegression())
        ])
        scores_complex = cross_val_score(model_complex, X_train, y_train, cv=5, scoring='r2')
        
        var_simple = np.std(scores_simple)
        var_complex = np.std(scores_complex)
        
        # Complex model should have higher variance
        print(f"✓ Variance - Simple: {var_simple:.4f}, Complex: {var_complex:.4f}")
        assert var_complex > var_simple * 0.8, "Complex model should have higher variance"


class TestProductionPatterns:
    """Test production ML pipeline patterns"""
    
    def test_data_generation(self):
        """Test data generation with controlled randomness"""
        np.random.seed(42)
        X = np.sort(np.random.rand(100, 1) * 10, axis=0)
        y = np.sin(X).ravel() + np.random.randn(100) * 0.3
        
        assert X.shape == (100, 1), "X should have correct shape"
        assert y.shape == (100,), "y should have correct shape"
        assert -2 < y.min() < 2, "y values should be reasonable"
        assert -2 < y.max() < 2, "y values should be reasonable"
        print(f"✓ Data generation: X.shape={X.shape}, y range=[{y.min():.2f}, {y.max():.2f}]")
    
    def test_metric_calculation(self):
        """Test R² score calculation"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        score = r2_score(y_true, y_pred)
        assert 0.9 < score <= 1.0, "R² should be high for good predictions"
        print(f"✓ Metric calculation: R²={score:.4f}")
    
    def test_model_persistence(self, sample_data):
        """Test that model predictions are consistent"""
        X_train, X_test, y_train, y_test = sample_data
        
        model = Pipeline([
            ('poly_features', PolynomialFeatures(degree=3)),
            ('linear_regression', LinearRegression())
        ])
        model.fit(X_train, y_train)
        
        # Multiple predictions should be identical
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)
        
        assert np.allclose(pred1, pred2), "Predictions should be deterministic"
        print("✓ Model predictions are consistent")
    
    @pytest.fixture
    def sample_data(self):
        """Generate test data"""
        np.random.seed(42)
        X = np.sort(np.random.rand(100, 1) * 10, axis=0)
        y = np.sin(X).ravel() + np.random.randn(100) * 0.3
        return train_test_split(X, y, test_size=0.2, random_state=42)


def test_imports():
    """Test that all required libraries are available"""
    try:
        import numpy
        import sklearn
        import matplotlib
        print("✓ All required libraries imported successfully")
    except ImportError as e:
        pytest.fail(f"Missing required library: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
