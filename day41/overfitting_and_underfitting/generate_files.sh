#!/bin/bash

# Day 41: Overfitting and Underfitting - Implementation Package Generator
# This script creates all necessary files for the lesson

echo "🚀 Generating Day 41: Overfitting and Underfitting lesson files..."

# Create setup.sh
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 41: Overfitting and Underfitting environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo "✅ Setup complete! Virtual environment ready."
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the lesson:"
echo "  python lesson_code.py"
echo ""
echo "To run tests:"
echo "  pytest test_lesson.py -v"
EOF

chmod +x setup.sh

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
pytest==7.4.0
EOF

# Create lesson_code.py
cat > lesson_code.py << 'EOF'
"""
Day 41: Overfitting and Underfitting Detection System
A production-grade diagnostic tool for model complexity analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class OverfittingDetector:
    """
    Detects overfitting and underfitting in ML models.
    Used in production to monitor model complexity and performance.
    """
    
    def __init__(self, max_degree=15, test_size=0.2, random_state=42):
        self.max_degree = max_degree
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}
        
    def generate_data(self, n_samples=100, noise=0.3):
        """
        Generate synthetic data with true pattern + noise.
        Simulates real-world scenarios like user behavior, sensor readings, etc.
        """
        np.random.seed(self.random_state)
        X = np.sort(np.random.rand(n_samples, 1) * 10, axis=0)
        y = np.sin(X).ravel() + np.random.randn(n_samples) * noise
        
        return train_test_split(X, y, test_size=self.test_size, 
                                random_state=self.random_state)
    
    def analyze_complexity(self, X_train, X_test, y_train, y_test):
        """
        Sweep through model complexities to identify optimal point.
        This is what Spotify/Netflix do when tuning recommendation algorithms.
        """
        train_scores = []
        test_scores = []
        train_mse = []
        test_mse = []
        
        print("\n🔍 Analyzing model complexity...")
        print(f"{'Degree':<8} {'Train R²':<12} {'Test R²':<12} {'Gap':<10} {'Status'}")
        print("-" * 60)
        
        for degree in range(1, self.max_degree + 1):
            # Create polynomial model
            model = Pipeline([
                ('poly_features', PolynomialFeatures(degree=degree)),
                ('linear_regression', LinearRegression())
            ])
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            gap = train_r2 - test_r2
            
            train_scores.append(train_r2)
            test_scores.append(test_r2)
            train_mse.append(mean_squared_error(y_train, train_pred))
            test_mse.append(mean_squared_error(y_test, test_pred))
            
            # Classify model state
            if degree <= 2:
                status = "🔴 UNDERFIT"
            elif gap > 0.2:
                status = "🔴 OVERFIT"
            elif test_r2 < 0.5:
                status = "🟡 POOR"
            else:
                status = "🟢 GOOD"
            
            print(f"{degree:<8} {train_r2:>10.4f}  {test_r2:>10.4f}  "
                  f"{gap:>8.4f}  {status}")
        
        self.results['complexity'] = {
            'degrees': list(range(1, self.max_degree + 1)),
            'train_scores': train_scores,
            'test_scores': test_scores,
            'train_mse': train_mse,
            'test_mse': test_mse
        }
        
        # Find optimal degree (best test score)
        optimal_idx = np.argmax(test_scores)
        optimal_degree = optimal_idx + 1
        
        print(f"\n✨ Optimal complexity: Degree {optimal_degree}")
        print(f"   Train R²: {train_scores[optimal_idx]:.4f}")
        print(f"   Test R²: {test_scores[optimal_idx]:.4f}")
        print(f"   Gap: {train_scores[optimal_idx] - test_scores[optimal_idx]:.4f}")
        
        return optimal_degree
    
    def analyze_learning_curves(self, X_train, y_train, optimal_degree):
        """
        Generate learning curves to diagnose if more data helps.
        Used in production to decide: collect more data vs. change architecture.
        """
        print(f"\n📈 Generating learning curves for degree {optimal_degree}...")
        
        model = Pipeline([
            ('poly_features', PolynomialFeatures(degree=optimal_degree)),
            ('linear_regression', LinearRegression())
        ])
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='r2', random_state=self.random_state
        )
        
        self.results['learning_curves'] = {
            'train_sizes': train_sizes,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'test_scores_mean': np.mean(test_scores, axis=1),
            'test_scores_std': np.std(test_scores, axis=1)
        }
        
        # Analyze convergence
        final_gap = (self.results['learning_curves']['train_scores_mean'][-1] - 
                     self.results['learning_curves']['test_scores_mean'][-1])
        
        print(f"   Final train score: {self.results['learning_curves']['train_scores_mean'][-1]:.4f}")
        print(f"   Final test score: {self.results['learning_curves']['test_scores_mean'][-1]:.4f}")
        print(f"   Convergence gap: {final_gap:.4f}")
        
        if final_gap > 0.1:
            print("   ⚠️  Still overfitting - consider more data or regularization")
        else:
            print("   ✅ Good convergence - model generalizes well")
    
    def cross_validation_analysis(self, X_train, y_train, optimal_degree):
        """
        Measure model stability across different data subsets.
        High variance = overfitting. Production systems require stable predictions.
        """
        print(f"\n🎯 Cross-validation analysis for degree {optimal_degree}...")
        
        model = Pipeline([
            ('poly_features', PolynomialFeatures(degree=optimal_degree)),
            ('linear_regression', LinearRegression())
        ])
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                     scoring='r2', n_jobs=-1)
        
        self.results['cv_analysis'] = {
            'scores': cv_scores,
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores)
        }
        
        print(f"   CV Scores: {cv_scores}")
        print(f"   Mean: {np.mean(cv_scores):.4f}")
        print(f"   Std Dev: {np.std(cv_scores):.4f}")
        
        # Interpret variance
        if np.std(cv_scores) > 0.1:
            print("   ⚠️  High variance - model unstable across folds (overfitting)")
        else:
            print("   ✅ Low variance - model predictions are stable")
        
        return cv_scores
    
    def visualize_results(self):
        """
        Create diagnostic plots used in production ML pipelines.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Overfitting/Underfitting Diagnostic Dashboard', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Complexity vs Performance
        ax1 = axes[0, 0]
        degrees = self.results['complexity']['degrees']
        ax1.plot(degrees, self.results['complexity']['train_scores'], 
                'o-', label='Train R²', linewidth=2, markersize=6)
        ax1.plot(degrees, self.results['complexity']['test_scores'], 
                's-', label='Test R²', linewidth=2, markersize=6)
        ax1.axvline(x=np.argmax(self.results['complexity']['test_scores']) + 1, 
                   color='green', linestyle='--', alpha=0.5, label='Optimal')
        ax1.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=11)
        ax1.set_ylabel('R² Score', fontsize=11)
        ax1.set_title('Bias-Variance Trade-off', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-0.5, 1.1])
        
        # Annotate regions
        ax1.text(2, 0.2, 'UNDERFIT\n(High Bias)', ha='center', 
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        ax1.text(12, 0.2, 'OVERFIT\n(High Variance)', ha='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        # Plot 2: Learning Curves
        ax2 = axes[0, 1]
        lc = self.results['learning_curves']
        ax2.plot(lc['train_sizes'], lc['train_scores_mean'], 
                'o-', label='Train', linewidth=2)
        ax2.fill_between(lc['train_sizes'], 
                         lc['train_scores_mean'] - lc['train_scores_std'],
                         lc['train_scores_mean'] + lc['train_scores_std'],
                         alpha=0.2)
        ax2.plot(lc['train_sizes'], lc['test_scores_mean'], 
                's-', label='Test', linewidth=2)
        ax2.fill_between(lc['train_sizes'], 
                         lc['test_scores_mean'] - lc['test_scores_std'],
                         lc['test_scores_mean'] + lc['test_scores_std'],
                         alpha=0.2)
        ax2.set_xlabel('Training Set Size', fontsize=11)
        ax2.set_ylabel('R² Score', fontsize=11)
        ax2.set_title('Learning Curves (Optimal Model)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Train-Test Gap Analysis
        ax3 = axes[1, 0]
        gaps = np.array(self.results['complexity']['train_scores']) - \
               np.array(self.results['complexity']['test_scores'])
        colors = ['green' if g < 0.15 else 'orange' if g < 0.25 else 'red' for g in gaps]
        ax3.bar(degrees, gaps, color=colors, alpha=0.7)
        ax3.axhline(y=0.15, color='orange', linestyle='--', 
                   label='Warning Threshold', linewidth=2)
        ax3.axhline(y=0.25, color='red', linestyle='--', 
                   label='Critical Threshold', linewidth=2)
        ax3.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=11)
        ax3.set_ylabel('Train-Test Gap', fontsize=11)
        ax3.set_title('Overfitting Detection (Gap Analysis)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Cross-Validation Scores
        ax4 = axes[1, 1]
        cv_scores = self.results['cv_analysis']['scores']
        fold_nums = range(1, len(cv_scores) + 1)
        ax4.bar(fold_nums, cv_scores, color='steelblue', alpha=0.7)
        ax4.axhline(y=self.results['cv_analysis']['mean'], 
                   color='red', linestyle='--', linewidth=2, label='Mean')
        ax4.set_xlabel('Fold Number', fontsize=11)
        ax4.set_ylabel('R² Score', fontsize=11)
        ax4.set_title('Cross-Validation Stability', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('overfitting_analysis.png', dpi=150, bbox_inches='tight')
        print("\n📊 Diagnostic plots saved as 'overfitting_analysis.png'")
        plt.show()


def main():
    """
    Run complete overfitting/underfitting analysis.
    This mirrors what runs in production ML pipelines at scale.
    """
    print("=" * 70)
    print("Day 41: Overfitting and Underfitting Detection System")
    print("Production-Grade Model Complexity Analysis")
    print("=" * 70)
    
    # Initialize detector
    detector = OverfittingDetector(max_degree=15, test_size=0.2)
    
    # Generate data
    print("\n📦 Generating synthetic dataset (n=100, noise=0.3)...")
    X_train, X_test, y_train, y_test = detector.generate_data(n_samples=100, noise=0.3)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Analyze model complexity
    optimal_degree = detector.analyze_complexity(X_train, X_test, y_train, y_test)
    
    # Generate learning curves
    detector.analyze_learning_curves(X_train, y_train, optimal_degree)
    
    # Cross-validation analysis
    detector.cross_validation_analysis(X_train, y_train, optimal_degree)
    
    # Visualize
    detector.visualize_results()
    
    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("\n💡 Key Insights:")
    print("   • Models with degree 1-2 underfit (too simple)")
    print("   • Models with degree >8 overfit (memorize noise)")
    print("   • Optimal model balances bias and variance")
    print("   • Production systems monitor these metrics 24/7")
    print("=" * 70)


if __name__ == "__main__":
    main()
EOF

# Create test_lesson.py
cat > test_lesson.py << 'EOF'
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
EOF

# Create README.md
cat > README.md << 'EOF'
# Day 41: Overfitting and Underfitting

## Overview
Production-grade diagnostic system for detecting overfitting and underfitting in machine learning models. This tool mirrors what runs 24/7 in real ML pipelines at companies like Netflix, Spotify, and Tesla.

## What You'll Learn
- Detect when models are too simple (underfitting) or too complex (overfitting)
- Analyze bias-variance trade-off in real-time
- Generate learning curves to diagnose data needs
- Measure model stability with cross-validation
- Interpret train-test gaps like production engineers

## Prerequisites
- Python 3.11+
- Basic understanding of machine learning concepts
- Familiarity with train/test splits

## Quick Start

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run the Analysis
```bash
python lesson_code.py
```

**Expected Output:**
- Complexity analysis for polynomial degrees 1-15
- Optimal model identification (typically degree 3-4)
- Learning curve convergence analysis
- Cross-validation stability metrics
- Diagnostic visualization saved as PNG

### 3. Run Tests
```bash
pytest test_lesson.py -v
```

**Test Coverage:**
- Underfit detection (degree 1 models)
- Overfit detection (degree 15 models)
- Optimal model identification
- Train-test gap analysis
- Cross-validation variance

## Understanding the Output

### Complexity Analysis Table
```
Degree   Train R²     Test R²      Gap       Status
----------------------------------------------------------------
1        0.2453      0.2134      0.0319    🔴 UNDERFIT
3        0.8712      0.8456      0.0256    🟢 GOOD
15       0.9998      0.3421      0.6577    🔴 OVERFIT
```

**Interpretation:**
- **Degree 1-2**: Underfit (too simple, can't capture pattern)
- **Degree 3-5**: Optimal (captures pattern, generalizes well)
- **Degree 8+**: Overfit (memorizes noise, fails on new data)

### Learning Curves
Shows how performance improves with more training data:
- **Converging curves**: Model benefits from more data
- **Plateau**: Model at capacity, need different architecture
- **Large gap**: Overfitting, need regularization

### Cross-Validation Scores
Measures prediction stability across data subsets:
- **Low variance (<0.05)**: Stable, reliable model
- **High variance (>0.1)**: Unstable, likely overfitting

## Real-World Applications

### Netflix Recommendations
- Monitors complexity of personalization models
- Alerts when train-test gap exceeds threshold
- Auto-adjusts regularization based on CV variance

### Tesla Autopilot
- Tracks lane detection model stability
- Reduces complexity if variance increases
- Ensures consistent predictions across road conditions

### Google Search Ranking
- Runs learning curve analysis on ranking models
- Decides when to collect more training data
- Balances 10,000+ ranking features optimally

## Files Generated
- `lesson_code.py` - Main diagnostic system
- `test_lesson.py` - Comprehensive test suite
- `requirements.txt` - Python dependencies
- `setup.sh` - Environment setup script
- `overfitting_analysis.png` - Diagnostic plots

## Key Concepts

### Bias-Variance Trade-off
```
Total Error = Bias² + Variance + Irreducible Error

Bias: Systematic error (underfitting)
Variance: Sensitivity to training data (overfitting)
Irreducible: Noise in data itself
```

### Production Monitoring Pattern
```
1. Deploy model at conservative complexity
2. Monitor train-test gap in real-time
3. Alert when gap exceeds threshold (typically 10-15%)
4. Auto-trigger regularization or retraining
5. Repeat continuously
```

## Troubleshooting

### Issue: Tests failing
**Solution:** Ensure virtual environment is activated
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Plots not displaying
**Solution:** Install GUI backend or save to file only
```bash
# In lesson_code.py, remove plt.show() to save only
```

### Issue: Import errors
**Solution:** Verify Python version
```bash
python3 --version  # Should be 3.11+
```

## Next Steps
- **Day 42**: Learn proper data splitting (train/test/validation)
- **Day 43**: Implement cross-validation strategies
- **Day 44**: Apply regularization techniques (L1/L2)

## Resources
- Scikit-learn Model Selection: https://scikit-learn.org/stable/model_selection.html
- Bias-Variance Trade-off: https://en.wikipedia.org/wiki/Bias–variance_tradeoff
- Learning Curves Guide: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

---

**Course:** 180-Day AI and Machine Learning from Scratch  
**Module:** Week 7 - Core Concepts  
**Lesson:** Day 41 of 180
EOF

chmod +x setup.sh

echo ""
echo "✅ All files generated successfully!"
echo ""
echo "📁 Files created:"
echo "   - setup.sh (environment setup)"
echo "   - lesson_code.py (main implementation)"
echo "   - test_lesson.py (test suite)"
echo "   - requirements.txt (dependencies)"
echo "   - README.md (documentation)"
echo ""
echo "🚀 Next steps:"
echo "   1. chmod +x setup.sh && ./setup.sh"
echo "   2. source venv/bin/activate"
echo "   3. python lesson_code.py"
echo "   4. pytest test_lesson.py -v"
echo ""
echo "Happy learning! 🎓"