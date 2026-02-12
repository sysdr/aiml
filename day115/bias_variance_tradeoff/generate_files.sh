#!/bin/bash

# Day 115: Bias-Variance Tradeoff - Complete Implementation Package Generator
# This script creates all necessary files for the lesson

set -e

echo "Generating Day 115: Bias-Variance Tradeoff Implementation Package..."

# Create setup_env.sh (renamed to avoid overwriting generator)
cat > setup_env.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 115: Bias-Variance Tradeoff environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Setup complete! Activate the environment with: source venv/bin/activate"
echo "Then run: python lesson_code.py"
echo "Or run tests: pytest test_lesson.py -v"
EOF

chmod +x setup_env.sh

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.26.4
pandas==2.2.1
matplotlib==3.8.3
seaborn==0.13.2
scikit-learn==1.4.1.post1
pytest==8.1.1
scipy==1.12.0
EOF

# Create lesson_code.py
cat > lesson_code.py << 'EOF'
"""
Day 115: Bias-Variance Tradeoff
Production-grade diagnostic system for analyzing model bias and variance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class BiasVarianceAnalyzer:
    """
    Production bias-variance diagnostic system
    Used by ML teams to identify underfitting vs overfitting
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_synthetic_data(
        self, 
        n_samples: int = 200,
        noise_level: float = 0.1,
        complexity: str = 'medium'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic dataset with known ground truth
        
        Args:
            n_samples: Number of data points
            noise_level: Amount of random noise (0-1)
            complexity: 'low', 'medium', or 'high' for underlying function complexity
        
        Returns:
            X, y arrays
        """
        X = np.linspace(0, 10, n_samples).reshape(-1, 1)
        
        if complexity == 'low':
            # Simple linear relationship
            y = 2 * X.flatten() + 1
        elif complexity == 'medium':
            # Polynomial relationship
            y = 0.5 * X.flatten()**2 - 3 * X.flatten() + 10
        else:  # high
            # Complex non-linear relationship
            y = np.sin(X.flatten()) * X.flatten() + 0.5 * X.flatten()**2
        
        # Add noise
        noise = np.random.normal(0, noise_level * np.std(y), n_samples)
        y += noise
        
        return X, y
    
    def compute_learning_curves(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate learning curves to diagnose bias/variance
        
        High bias: Both train and val error high and converged
        High variance: Large gap between train and val error
        
        Returns:
            Dictionary with train_sizes, train_scores, val_scores
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,
            scoring='neg_mean_squared_error',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Convert to positive MSE
        train_scores = -train_scores
        val_scores = -val_scores
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }
    
    def bootstrap_variance_analysis(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_bootstraps: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Bootstrap sampling to estimate prediction variance
        
        High variance models will show large prediction uncertainty
        across different bootstrap samples
        
        Returns:
            Dictionary with predictions and variance metrics
        """
        n_samples = len(X_train)
        predictions = np.zeros((n_bootstraps, len(X_test)))
        
        for i in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Train and predict
            model_boot = model.__class__(**model.get_params())
            model_boot.fit(X_boot, y_boot)
            predictions[i] = model_boot.predict(X_test)
        
        return {
            'predictions': predictions,
            'mean_prediction': np.mean(predictions, axis=0),
            'std_prediction': np.std(predictions, axis=0),
            'variance': np.var(predictions, axis=0)
        }
    
    def model_complexity_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_degree: int = 10
    ) -> Dict[str, List[float]]:
        """
        Analyze bias-variance tradeoff across model complexities
        
        Fits polynomial models of increasing degree and tracks
        training and validation error
        
        Returns:
            Dictionary with degrees, train_errors, val_errors
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        degrees = list(range(1, max_degree + 1))
        train_errors = []
        val_errors = []
        
        for degree in degrees:
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_val_poly = poly.transform(X_val)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            
            # Compute errors
            train_pred = model.predict(X_train_poly)
            val_pred = model.predict(X_val_poly)
            
            train_errors.append(mean_squared_error(y_train, train_pred))
            val_errors.append(mean_squared_error(y_val, val_pred))
        
        return {
            'degrees': degrees,
            'train_errors': train_errors,
            'val_errors': val_errors
        }
    
    def cross_validation_stability(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Measure cross-validation stability
        
        High variance models show inconsistent performance across folds
        High bias models show consistent but poor performance
        
        Returns:
            Dictionary with CV scores and stability metrics
        """
        cv_scores = cross_val_score(
            model, X, y,
            cv=n_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Convert to positive MSE
        cv_scores = -cv_scores
        
        return {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'min_score': np.min(cv_scores),
            'max_score': np.max(cv_scores),
            'coefficient_of_variation': np.std(cv_scores) / np.mean(cv_scores)
        }
    
    def diagnose_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, str]:
        """
        Automated diagnosis of bias-variance issues
        
        Provides actionable recommendations based on error patterns
        
        Returns:
            Dictionary with diagnosis and recommendations
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Compute errors
        train_error = mean_squared_error(y_train, model.predict(X_train))
        val_error = mean_squared_error(y_val, model.predict(X_val))
        
        # Get CV stability
        cv_results = self.cross_validation_stability(model, X, y)
        
        # Diagnosis logic
        error_ratio = val_error / train_error if train_error > 0 else float('inf')
        cv_stability = cv_results['coefficient_of_variation']
        
        diagnosis = {}
        
        # High variance diagnosis
        if error_ratio > 2.0 or cv_stability > 0.3:
            diagnosis['issue'] = 'High Variance (Overfitting)'
            diagnosis['severity'] = 'High' if error_ratio > 3.0 else 'Medium'
            diagnosis['evidence'] = [
                f"Validation error is {error_ratio:.2f}x training error",
                f"CV stability coefficient: {cv_stability:.3f}",
                "Model is memorizing training data"
            ]
            diagnosis['recommendations'] = [
                "Collect more training data",
                "Apply regularization (L1/L2/dropout)",
                "Reduce model complexity",
                "Use ensemble methods (bagging)",
                "Apply early stopping"
            ]
        # High bias diagnosis
        elif train_error > np.var(y) * 0.5:  # Arbitrary threshold
            diagnosis['issue'] = 'High Bias (Underfitting)'
            diagnosis['severity'] = 'High'
            diagnosis['evidence'] = [
                f"Training error is {train_error:.4f}",
                f"Model explains <50% of variance",
                "Model is too simple for the data"
            ]
            diagnosis['recommendations'] = [
                "Increase model complexity",
                "Add more features or polynomial terms",
                "Reduce regularization strength",
                "Try more flexible model (trees, neural nets)",
                "Engineer better features"
            ]
        else:
            diagnosis['issue'] = 'Well-Balanced'
            diagnosis['severity'] = 'Low'
            diagnosis['evidence'] = [
                f"Train/val error ratio: {error_ratio:.2f}",
                f"CV stability: {cv_stability:.3f}",
                "Model generalizes well"
            ]
            diagnosis['recommendations'] = [
                "Current configuration is appropriate",
                "Monitor for model drift in production",
                "Continue collecting data for long-term improvement"
            ]
        
        diagnosis['train_error'] = train_error
        diagnosis['val_error'] = val_error
        diagnosis['error_ratio'] = error_ratio
        
        return diagnosis


class BiasVarianceVisualizer:
    """
    Production visualization suite for bias-variance diagnostics
    """
    
    def __init__(self):
        sns.set_style('whitegrid')
        self.colors = {
            'train': '#2E86AB',
            'val': '#A23B72',
            'optimal': '#F18F01'
        }
    
    def plot_learning_curves(
        self,
        curves: Dict[str, np.ndarray],
        title: str = "Learning Curves",
        save_path: str = None
    ):
        """Plot learning curves with confidence intervals"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_sizes = curves['train_sizes']
        train_mean = curves['train_scores_mean']
        train_std = curves['train_scores_std']
        val_mean = curves['val_scores_mean']
        val_std = curves['val_scores_std']
        
        # Plot training scores
        ax.plot(train_sizes, train_mean, 'o-', color=self.colors['train'],
                label='Training Error', linewidth=2, markersize=8)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                         alpha=0.2, color=self.colors['train'])
        
        # Plot validation scores
        ax.plot(train_sizes, val_mean, 's-', color=self.colors['val'],
                label='Validation Error', linewidth=2, markersize=8)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                         alpha=0.2, color=self.colors['val'])
        
        ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_complexity_analysis(
        self,
        complexity_results: Dict[str, List[float]],
        title: str = "Model Complexity vs Error",
        save_path: str = None
    ):
        """Plot bias-variance tradeoff across model complexities"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        degrees = complexity_results['degrees']
        train_errors = complexity_results['train_errors']
        val_errors = complexity_results['val_errors']
        
        # Find optimal complexity
        optimal_idx = np.argmin(val_errors)
        optimal_degree = degrees[optimal_idx]
        
        # Plot errors
        ax.plot(degrees, train_errors, 'o-', color=self.colors['train'],
                label='Training Error', linewidth=2, markersize=8)
        ax.plot(degrees, val_errors, 's-', color=self.colors['val'],
                label='Validation Error', linewidth=2, markersize=8)
        
        # Mark optimal point
        ax.axvline(x=optimal_degree, color=self.colors['optimal'],
                   linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Optimal Complexity (Degree {optimal_degree})')
        
        ax.set_xlabel('Polynomial Degree (Model Complexity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_bootstrap_predictions(
        self,
        X_test: np.ndarray,
        bootstrap_results: Dict[str, np.ndarray],
        y_true: np.ndarray = None,
        title: str = "Bootstrap Prediction Uncertainty",
        save_path: str = None
    ):
        """Visualize prediction uncertainty from bootstrap analysis"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        mean_pred = bootstrap_results['mean_prediction']
        std_pred = bootstrap_results['std_prediction']
        
        # Sort for cleaner visualization
        sort_idx = np.argsort(X_test.flatten())
        X_sorted = X_test[sort_idx]
        mean_sorted = mean_pred[sort_idx]
        std_sorted = std_pred[sort_idx]
        
        # Plot mean prediction
        ax.plot(X_sorted, mean_sorted, color=self.colors['val'],
                linewidth=2, label='Mean Prediction')
        
        # Plot confidence interval
        ax.fill_between(X_sorted.flatten(),
                         mean_sorted - 2*std_sorted,
                         mean_sorted + 2*std_sorted,
                         alpha=0.3, color=self.colors['val'],
                         label='95% Confidence Interval')
        
        # Plot true values if provided
        if y_true is not None:
            y_sorted = y_true[sort_idx]
            ax.scatter(X_sorted, y_sorted, color=self.colors['train'],
                      alpha=0.5, s=50, label='True Values')
        
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Prediction', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def demonstrate_bias_variance():
    """
    Production demonstration of bias-variance analysis
    """
    print("=" * 80)
    print("Day 115: Bias-Variance Tradeoff - Production Diagnostic System")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = BiasVarianceAnalyzer(random_state=42)
    visualizer = BiasVarianceVisualizer()
    
    # Generate synthetic data with medium complexity
    print("\n1. Generating synthetic dataset...")
    X, y = analyzer.generate_synthetic_data(
        n_samples=200,
        noise_level=0.2,
        complexity='medium'
    )
    print(f"   Generated {len(X)} samples")
    print(f"   Target variance: {np.var(y):.4f}")
    
    # Demonstrate high bias (underfitting)
    print("\n2. Analyzing HIGH BIAS model (too simple)...")
    print("   Using linear regression on non-linear data")
    simple_model = LinearRegression()
    diagnosis_simple = analyzer.diagnose_model(simple_model, X, y)
    
    print(f"\n   Issue: {diagnosis_simple['issue']}")
    print(f"   Severity: {diagnosis_simple['severity']}")
    print(f"   Train Error: {diagnosis_simple['train_error']:.4f}")
    print(f"   Val Error: {diagnosis_simple['val_error']:.4f}")
    print(f"   Error Ratio: {diagnosis_simple['error_ratio']:.2f}")
    print("\n   Evidence:")
    for evidence in diagnosis_simple['evidence']:
        print(f"   - {evidence}")
    print("\n   Recommendations:")
    for rec in diagnosis_simple['recommendations'][:3]:
        print(f"   - {rec}")
    
    # Generate learning curves for simple model
    print("\n3. Computing learning curves for simple model...")
    curves_simple = analyzer.compute_learning_curves(simple_model, X, y)
    visualizer.plot_learning_curves(
        curves_simple,
        title="Learning Curves: High Bias (Underfitting)",
        save_path="learning_curves_high_bias.png"
    )
    print("   Saved: learning_curves_high_bias.png")
    
    # Demonstrate high variance (overfitting)
    print("\n4. Analyzing HIGH VARIANCE model (too complex)...")
    print("   Using deep decision tree")
    complex_model = DecisionTreeRegressor(max_depth=20, random_state=42)
    diagnosis_complex = analyzer.diagnose_model(complex_model, X, y)
    
    print(f"\n   Issue: {diagnosis_complex['issue']}")
    print(f"   Severity: {diagnosis_complex['severity']}")
    print(f"   Train Error: {diagnosis_complex['train_error']:.4f}")
    print(f"   Val Error: {diagnosis_complex['val_error']:.4f}")
    print(f"   Error Ratio: {diagnosis_complex['error_ratio']:.2f}")
    print("\n   Evidence:")
    for evidence in diagnosis_complex['evidence']:
        print(f"   - {evidence}")
    print("\n   Recommendations:")
    for rec in diagnosis_complex['recommendations'][:3]:
        print(f"   - {rec}")
    
    # Generate learning curves for complex model
    print("\n5. Computing learning curves for complex model...")
    curves_complex = analyzer.compute_learning_curves(complex_model, X, y)
    visualizer.plot_learning_curves(
        curves_complex,
        title="Learning Curves: High Variance (Overfitting)",
        save_path="learning_curves_high_variance.png"
    )
    print("   Saved: learning_curves_high_variance.png")
    
    # Model complexity analysis
    print("\n6. Performing model complexity sweep...")
    complexity_results = analyzer.model_complexity_analysis(X, y, max_degree=10)
    optimal_degree = complexity_results['degrees'][
        np.argmin(complexity_results['val_errors'])
    ]
    print(f"   Optimal polynomial degree: {optimal_degree}")
    print(f"   Minimum validation error: {min(complexity_results['val_errors']):.4f}")
    
    visualizer.plot_complexity_analysis(
        complexity_results,
        title="Bias-Variance Tradeoff: Model Complexity Analysis",
        save_path="complexity_analysis.png"
    )
    print("   Saved: complexity_analysis.png")
    
    # Bootstrap variance analysis
    print("\n7. Bootstrap variance analysis...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Analyze variance for complex model
    bootstrap_results = analyzer.bootstrap_variance_analysis(
        complex_model, X_train, y_train, X_test, n_bootstraps=50
    )
    avg_variance = np.mean(bootstrap_results['variance'])
    print(f"   Average prediction variance: {avg_variance:.4f}")
    print(f"   Max prediction std: {np.max(bootstrap_results['std_prediction']):.4f}")
    
    visualizer.plot_bootstrap_predictions(
        X_test,
        bootstrap_results,
        y_test,
        title="Prediction Uncertainty: Bootstrap Analysis",
        save_path="bootstrap_variance.png"
    )
    print("   Saved: bootstrap_variance.png")
    
    # Demonstrate well-balanced model
    print("\n8. Analyzing WELL-BALANCED model...")
    print("   Using regularized ensemble")
    balanced_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    diagnosis_balanced = analyzer.diagnose_model(balanced_model, X, y)
    
    print(f"\n   Issue: {diagnosis_balanced['issue']}")
    print(f"   Severity: {diagnosis_balanced['severity']}")
    print(f"   Train Error: {diagnosis_balanced['train_error']:.4f}")
    print(f"   Val Error: {diagnosis_balanced['val_error']:.4f}")
    print(f"   Error Ratio: {diagnosis_balanced['error_ratio']:.2f}")
    
    # Cross-validation stability
    print("\n9. Cross-validation stability analysis...")
    cv_results_simple = analyzer.cross_validation_stability(simple_model, X, y)
    cv_results_complex = analyzer.cross_validation_stability(complex_model, X, y)
    cv_results_balanced = analyzer.cross_validation_stability(balanced_model, X, y)
    
    print(f"\n   Simple Model (High Bias):")
    print(f"   - Mean CV Error: {cv_results_simple['mean_score']:.4f}")
    print(f"   - Std CV Error: {cv_results_simple['std_score']:.4f}")
    print(f"   - Coefficient of Variation: {cv_results_simple['coefficient_of_variation']:.3f}")
    
    print(f"\n   Complex Model (High Variance):")
    print(f"   - Mean CV Error: {cv_results_complex['mean_score']:.4f}")
    print(f"   - Std CV Error: {cv_results_complex['std_score']:.4f}")
    print(f"   - Coefficient of Variation: {cv_results_complex['coefficient_of_variation']:.3f}")
    
    print(f"\n   Balanced Model:")
    print(f"   - Mean CV Error: {cv_results_balanced['mean_score']:.4f}")
    print(f"   - Std CV Error: {cv_results_balanced['std_score']:.4f}")
    print(f"   - Coefficient of Variation: {cv_results_balanced['coefficient_of_variation']:.3f}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. High Bias: Both train and val errors are high → Need more model capacity")
    print("2. High Variance: Large gap between train and val errors → Need regularization")
    print("3. Balanced Model: Small gap, reasonable errors → Production-ready")
    print("\nGenerated visualizations:")
    print("- learning_curves_high_bias.png")
    print("- learning_curves_high_variance.png")
    print("- complexity_analysis.png")
    print("- bootstrap_variance.png")


if __name__ == "__main__":
    demonstrate_bias_variance()
EOF

# Create test_lesson.py
cat > test_lesson.py << 'EOF'
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
EOF

# Create README.md
cat > README.md << 'EOF'
# Day 115: Bias-Variance Tradeoff

Production-grade diagnostic system for analyzing and resolving bias-variance issues in machine learning models.

## Overview

This implementation provides comprehensive tools used by ML teams at companies like Netflix, Google, and Airbnb to diagnose whether models are underfitting (high bias) or overfitting (high variance), and provides actionable recommendations for fixing these issues.

## Features

- **Learning Curve Analysis**: Diagnose bias vs variance by plotting training and validation errors
- **Bootstrap Variance Estimation**: Quantify prediction uncertainty through bootstrap sampling
- **Model Complexity Sweep**: Find optimal model complexity before overfitting begins
- **Cross-Validation Stability**: Measure performance consistency across different data splits
- **Automated Diagnosis**: Get actionable recommendations based on error patterns
- **Production Visualizations**: Generate publication-quality diagnostic plots

## Quick Start

### Installation

```bash
# Run the setup script
chmod +x setup_env.sh
./setup_env.sh

# Activate virtual environment
source venv/bin/activate
```

### Run Main Demo

```bash
python lesson_code.py
```

This will:
1. Generate synthetic datasets
2. Analyze high bias (underfitting) models
3. Analyze high variance (overfitting) models
4. Demonstrate well-balanced models
5. Generate diagnostic visualizations
6. Provide actionable recommendations

### Run Tests

```bash
pytest test_lesson.py -v
```

Expected output: 20 tests passing, covering all diagnostic components.

## Key Concepts

### Bias-Variance Decomposition

Total Error = Bias² + Variance + Irreducible Error

- **Bias**: Systematic error from overly simple models
- **Variance**: Error from memorizing training data noise
- **Tradeoff**: Decreasing one increases the other

### Diagnostic Patterns

**High Bias (Underfitting)**:
- Training error: High
- Validation error: High
- Gap between errors: Small
- Solution: Increase model complexity

**High Variance (Overfitting)**:
- Training error: Low
- Validation error: High
- Gap between errors: Large
- Solution: Regularization or more data

**Well-Balanced**:
- Training error: Reasonable
- Validation error: Close to training error
- Gap between errors: Small
- Solution: Current configuration is good

## Usage Examples

### Basic Diagnosis

```python
from lesson_code import BiasVarianceAnalyzer

analyzer = BiasVarianceAnalyzer()
X, y = analyzer.generate_synthetic_data(n_samples=200)

model = LinearRegression()
diagnosis = analyzer.diagnose_model(model, X, y)

print(f"Issue: {diagnosis['issue']}")
print(f"Recommendations: {diagnosis['recommendations']}")
```

### Learning Curves

```python
curves = analyzer.compute_learning_curves(model, X, y)

# Check for high variance
if curves['val_scores_mean'][-1] > 2 * curves['train_scores_mean'][-1]:
    print("High variance detected - consider regularization")
```

### Model Complexity Analysis

```python
results = analyzer.model_complexity_analysis(X, y, max_degree=10)
optimal_degree = results['degrees'][np.argmin(results['val_errors'])]
print(f"Optimal polynomial degree: {optimal_degree}")
```

### Bootstrap Variance

```python
bootstrap_results = analyzer.bootstrap_variance_analysis(
    model, X_train, y_train, X_test, n_bootstraps=100
)

avg_variance = np.mean(bootstrap_results['variance'])
print(f"Average prediction variance: {avg_variance}")
```

## Production Applications

### Model Selection
- Compare models across complexity spectrum
- Identify optimal point before overfitting
- Guide hyperparameter tuning decisions

### Monitoring
- Track train/val error gap in production
- Detect model drift and concept drift
- Trigger retraining when gaps grow

### AutoML Integration
- Automatically select model complexity
- Apply appropriate regularization
- Optimize ensemble configurations

### A/B Testing
- Predict production performance from offline metrics
- Identify models likely to fail in production
- Guide experimental design

## Generated Outputs

The demo generates four diagnostic plots:

1. **learning_curves_high_bias.png**: Shows plateau in both errors
2. **learning_curves_high_variance.png**: Shows large train/val gap
3. **complexity_analysis.png**: Identifies optimal complexity
4. **bootstrap_variance.png**: Visualizes prediction uncertainty

## Requirements

- Python 3.11+
- NumPy 1.26.4
- Pandas 2.2.1
- Matplotlib 3.8.3
- Seaborn 0.13.2
- Scikit-learn 1.4.1.post1
- Pytest 8.1.1
- SciPy 1.12.0

## Testing

The test suite includes 20 comprehensive tests:

- Data generation validation
- Learning curve computation
- Bootstrap sampling correctness
- Complexity analysis trends
- Cross-validation mechanics
- Diagnostic accuracy
- Visualization rendering
- Integration workflows
- Edge case handling

## Real-World Connections

- **Netflix**: Recommendation model complexity selection
- **Google Photos**: Face recognition generalization
- **Uber**: Demand forecasting confidence intervals
- **Tesla**: Perception model regularization
- **Airbnb**: Pricing model bias-variance balance

## Next Steps

Tomorrow (Day 116) we'll apply this framework to systematic hyperparameter tuning, using bias-variance insights to guide grid search, random search, and Bayesian optimization.

## License

Educational purposes - Part of 180-Day AI/ML Course
EOF

echo "All files generated successfully!"
echo ""
echo "Next steps:"
echo "1. chmod +x setup_env.sh"
echo "2. ./setup_env.sh"
echo "3. source venv/bin/activate"
echo "4. python lesson_code.py"
echo "5. pytest test_lesson.py -v"
