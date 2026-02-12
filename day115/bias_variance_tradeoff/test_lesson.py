"""
Comprehensive test suite for Day 115: Bias-Variance Tradeoff
Tests all diagnostic components and analysis functions
"""

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from lesson_code import BiasVarianceAnalyzer, BiasVarianceVisualizer


class TestBiasVarianceAnalyzer:
    """Test suite for BiasVarianceAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        return BiasVarianceAnalyzer(random_state=42)
    
    @pytest.fixture
    def sample_data(self, analyzer):
        X, y = analyzer.generate_synthetic_data(n_samples=100)
        return X, y
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.random_state == 42
        assert isinstance(analyzer, BiasVarianceAnalyzer)
    
    def test_generate_synthetic_data_shape(self, analyzer):
        """Test synthetic data generation returns correct shapes"""
        X, y = analyzer.generate_synthetic_data(n_samples=150)
        assert X.shape == (150, 1)
        assert y.shape == (150,)
    
    def test_generate_synthetic_data_complexity_low(self, analyzer):
        """Test low complexity data generation"""
        X, y = analyzer.generate_synthetic_data(
            n_samples=100,
            noise_level=0.0,
            complexity='low'
        )
        # Low complexity should be roughly linear
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.95, "Low complexity data should be nearly linear"
    
    def test_generate_synthetic_data_complexity_medium(self, analyzer):
        """Test medium complexity data generation"""
        X, y = analyzer.generate_synthetic_data(
            n_samples=100,
            noise_level=0.0,
            complexity='medium'
        )
        # Medium complexity should not be perfectly linear
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 < 0.95, "Medium complexity data should not be perfectly linear"
    
    def test_generate_synthetic_data_noise_effect(self, analyzer):
        """Test that noise level affects data variability"""
        X1, y1 = analyzer.generate_synthetic_data(
            n_samples=200,
            noise_level=0.01,
            complexity='low'
        )
        X2, y2 = analyzer.generate_synthetic_data(
            n_samples=200,
            noise_level=0.5,
            complexity='low'
        )
        
        # Higher noise should result in higher variance around trend
        assert np.std(y2) > np.std(y1)
    
    def test_compute_learning_curves_structure(self, analyzer, sample_data):
        """Test learning curves return correct structure"""
        X, y = sample_data
        model = LinearRegression()
        
        curves = analyzer.compute_learning_curves(model, X, y)
        
        required_keys = [
            'train_sizes', 'train_scores_mean', 'train_scores_std',
            'val_scores_mean', 'val_scores_std'
        ]
        for key in required_keys:
            assert key in curves, f"Missing key: {key}"
        
        # All arrays should have same length
        lengths = [len(curves[key]) for key in required_keys]
        assert len(set(lengths)) == 1, "All curve arrays should have same length"
    
    def test_compute_learning_curves_values(self, analyzer, sample_data):
        """Test learning curves produce reasonable values"""
        X, y = sample_data
        model = LinearRegression()
        
        curves = analyzer.compute_learning_curves(model, X, y)
        
        # Errors should be positive
        assert np.all(curves['train_scores_mean'] >= 0)
        assert np.all(curves['val_scores_mean'] >= 0)
        
        # Standard deviations should be non-negative
        assert np.all(curves['train_scores_std'] >= 0)
        assert np.all(curves['val_scores_std'] >= 0)
    
    def test_bootstrap_variance_analysis_structure(self, analyzer, sample_data):
        """Test bootstrap analysis returns correct structure"""
        X, y = sample_data
        X_train, X_test = X[:70], X[70:]
        y_train = y[:70]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        results = analyzer.bootstrap_variance_analysis(
            model, X_train, y_train, X_test, n_bootstraps=20
        )
        
        required_keys = ['predictions', 'mean_prediction', 'std_prediction', 'variance']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
        
        # Check shapes
        assert results['predictions'].shape == (20, len(X_test))
        assert results['mean_prediction'].shape == (len(X_test),)
        assert results['std_prediction'].shape == (len(X_test),)
        assert results['variance'].shape == (len(X_test),)
    
    def test_bootstrap_variance_analysis_values(self, analyzer, sample_data):
        """Test bootstrap analysis produces reasonable values"""
        X, y = sample_data
        X_train, X_test = X[:70], X[70:]
        y_train = y[:70]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        results = analyzer.bootstrap_variance_analysis(
            model, X_train, y_train, X_test, n_bootstraps=20
        )
        
        # Variance should be non-negative
        assert np.all(results['variance'] >= 0)
        
        # Mean prediction should be close to direct prediction
        direct_pred = model.predict(X_test)
        np.testing.assert_allclose(
            results['mean_prediction'],
            direct_pred,
            rtol=0.2  # Allow 20% difference due to bootstrap sampling
        )
    
    def test_model_complexity_analysis_structure(self, analyzer, sample_data):
        """Test complexity analysis returns correct structure"""
        X, y = sample_data
        
        results = analyzer.model_complexity_analysis(X, y, max_degree=5)
        
        assert 'degrees' in results
        assert 'train_errors' in results
        assert 'val_errors' in results
        
        assert len(results['degrees']) == 5
        assert len(results['train_errors']) == 5
        assert len(results['val_errors']) == 5
    
    def test_model_complexity_analysis_trend(self, analyzer):
        """Test that complexity analysis shows expected trends"""
        # Generate quadratic data
        X, y = analyzer.generate_synthetic_data(
            n_samples=150,
            noise_level=0.1,
            complexity='medium'
        )
        
        results = analyzer.model_complexity_analysis(X, y, max_degree=8)
        
        # Training error should generally decrease with complexity
        train_errors = results['train_errors']
        assert train_errors[0] > train_errors[-1], \
            "Training error should decrease with complexity"
        
        # Validation error should have a minimum (not monotonically decreasing)
        val_errors = results['val_errors']
        min_idx = np.argmin(val_errors)
        assert 0 < min_idx < len(val_errors) - 1, \
            "Validation error should have minimum in middle range"
    
    def test_cross_validation_stability_structure(self, analyzer, sample_data):
        """Test CV stability returns correct structure"""
        X, y = sample_data
        model = LinearRegression()
        
        results = analyzer.cross_validation_stability(model, X, y, n_folds=5)
        
        required_keys = [
            'cv_scores', 'mean_score', 'std_score',
            'min_score', 'max_score', 'coefficient_of_variation'
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
        
        assert len(results['cv_scores']) == 5
    
    def test_cross_validation_stability_values(self, analyzer, sample_data):
        """Test CV stability produces reasonable values"""
        X, y = sample_data
        model = LinearRegression()
        
        results = analyzer.cross_validation_stability(model, X, y, n_folds=5)
        
        # All scores should be positive
        assert np.all(results['cv_scores'] > 0)
        
        # Mean should be between min and max
        assert results['min_score'] <= results['mean_score'] <= results['max_score']
        
        # Coefficient of variation should be non-negative
        assert results['coefficient_of_variation'] >= 0
    
    def test_diagnose_model_high_bias(self, analyzer):
        """Test diagnosis of high bias (underfitting) model"""
        # Generate complex data
        X, y = analyzer.generate_synthetic_data(
            n_samples=150,
            noise_level=0.1,
            complexity='high'
        )
        
        # Use overly simple model
        simple_model = LinearRegression()
        diagnosis = analyzer.diagnose_model(simple_model, X, y)
        
        assert 'issue' in diagnosis
        assert 'High Bias' in diagnosis['issue'] or 'Underfitting' in diagnosis['issue']
        assert diagnosis['train_error'] > 0
        assert diagnosis['val_error'] > 0
    
    def test_diagnose_model_high_variance(self, analyzer):
        """Test diagnosis of high variance (overfitting) model"""
        # Generate simple data
        X, y = analyzer.generate_synthetic_data(
            n_samples=100,
            noise_level=0.1,
            complexity='low'
        )
        
        # Use overly complex model
        complex_model = DecisionTreeRegressor(max_depth=20, random_state=42)
        diagnosis = analyzer.diagnose_model(complex_model, X, y)
        
        # Should show high variance characteristics
        assert diagnosis['error_ratio'] > 1.5  # Val error >> train error
    
    def test_diagnose_model_recommendations(self, analyzer, sample_data):
        """Test that diagnosis provides recommendations"""
        X, y = sample_data
        model = LinearRegression()
        
        diagnosis = analyzer.diagnose_model(model, X, y)
        
        assert 'recommendations' in diagnosis
        assert len(diagnosis['recommendations']) > 0
        assert isinstance(diagnosis['recommendations'], list)


class TestBiasVarianceVisualizer:
    """Test suite for BiasVarianceVisualizer"""
    
    @pytest.fixture
    def visualizer(self):
        return BiasVarianceVisualizer()
    
    @pytest.fixture
    def analyzer(self):
        return BiasVarianceAnalyzer(random_state=42)
    
    @pytest.fixture
    def sample_curves(self, analyzer):
        X, y = analyzer.generate_synthetic_data(n_samples=100)
        model = LinearRegression()
        return analyzer.compute_learning_curves(model, X, y)
    
    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initialization"""
        assert isinstance(visualizer, BiasVarianceVisualizer)
        assert 'train' in visualizer.colors
        assert 'val' in visualizer.colors
        assert 'optimal' in visualizer.colors
    
    def test_plot_learning_curves_no_crash(self, visualizer, sample_curves):
        """Test that plotting learning curves doesn't crash"""
        try:
            visualizer.plot_learning_curves(
                sample_curves,
                title="Test Learning Curves"
            )
            assert True
        except Exception as e:
            pytest.fail(f"Plotting learning curves failed: {e}")
    
    def test_plot_complexity_analysis_no_crash(self, visualizer, analyzer):
        """Test that plotting complexity analysis doesn't crash"""
        X, y = analyzer.generate_synthetic_data(n_samples=100)
        complexity_results = analyzer.model_complexity_analysis(X, y, max_degree=5)
        
        try:
            visualizer.plot_complexity_analysis(
                complexity_results,
                title="Test Complexity Analysis"
            )
            assert True
        except Exception as e:
            pytest.fail(f"Plotting complexity analysis failed: {e}")
    
    def test_plot_bootstrap_predictions_no_crash(self, visualizer, analyzer):
        """Test that plotting bootstrap predictions doesn't crash"""
        X, y = analyzer.generate_synthetic_data(n_samples=100)
        X_train, X_test = X[:70], X[70:]
        y_train, y_test = y[:70], y[70:]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        bootstrap_results = analyzer.bootstrap_variance_analysis(
            model, X_train, y_train, X_test, n_bootstraps=20
        )
        
        try:
            visualizer.plot_bootstrap_predictions(
                X_test,
                bootstrap_results,
                y_test,
                title="Test Bootstrap Predictions"
            )
            assert True
        except Exception as e:
            pytest.fail(f"Plotting bootstrap predictions failed: {e}")


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_complete_diagnostic_workflow(self):
        """Test complete diagnostic workflow end-to-end"""
        # Initialize
        analyzer = BiasVarianceAnalyzer(random_state=42)
        
        # Generate data
        X, y = analyzer.generate_synthetic_data(n_samples=150)
        
        # Test multiple models
        models = [
            ('Simple', LinearRegression()),
            ('Complex', DecisionTreeRegressor(max_depth=15, random_state=42))
        ]
        
        for name, model in models:
            # Diagnose
            diagnosis = analyzer.diagnose_model(model, X, y)
            
            # Verify diagnosis structure
            assert 'issue' in diagnosis
            assert 'recommendations' in diagnosis
            assert 'train_error' in diagnosis
            assert 'val_error' in diagnosis
            
            # Learning curves
            curves = analyzer.compute_learning_curves(model, X, y)
            assert len(curves['train_sizes']) > 0
            
            # CV stability
            cv_results = analyzer.cross_validation_stability(model, X, y)
            assert 'cv_scores' in cv_results
    
    def test_numerical_stability(self):
        """Test numerical stability with edge cases"""
        analyzer = BiasVarianceAnalyzer(random_state=42)
        
        # Test with very small dataset
        X_small, y_small = analyzer.generate_synthetic_data(n_samples=20)
        model = LinearRegression()
        
        try:
            diagnosis = analyzer.diagnose_model(model, X_small, y_small)
            assert diagnosis is not None
        except Exception as e:
            pytest.fail(f"Failed on small dataset: {e}")
        
        # Test with zero noise
        X_clean, y_clean = analyzer.generate_synthetic_data(
            n_samples=100,
            noise_level=0.0
        )
        
        try:
            diagnosis = analyzer.diagnose_model(model, X_clean, y_clean)
            assert diagnosis is not None
        except Exception as e:
            pytest.fail(f"Failed on clean dataset: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
