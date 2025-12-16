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
