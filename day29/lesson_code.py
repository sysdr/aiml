"""
Day 29: Central Limit Theorem - Production ML Toolkit
Demonstrates CLT through sampling distributions, confidence intervals, and A/B testing
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class CentralLimitTheoremSimulator:
    """
    Demonstrates the Central Limit Theorem with various population distributions.
    Shows how sample means converge to normal distribution regardless of source.
    """
    
    def __init__(self, population_size: int = 10000):
        """
        Initialize simulator with population size.
        
        Args:
            population_size: Size of the population to sample from
        """
        self.population_size = population_size
        self.distributions = {
            'uniform': self._generate_uniform,
            'exponential': self._generate_exponential,
            'bimodal': self._generate_bimodal,
            'skewed': self._generate_skewed
        }
    
    def _generate_uniform(self) -> np.ndarray:
        """Generate uniform distribution (flat, equal probability)"""
        return np.random.uniform(0, 10, self.population_size)
    
    def _generate_exponential(self) -> np.ndarray:
        """Generate exponential distribution (server response times, wait times)"""
        return np.random.exponential(scale=2.0, size=self.population_size)
    
    def _generate_bimodal(self) -> np.ndarray:
        """Generate bimodal distribution (two peaks, like user engagement patterns)"""
        mode1 = np.random.normal(3, 0.5, self.population_size // 2)
        mode2 = np.random.normal(7, 0.5, self.population_size // 2)
        return np.concatenate([mode1, mode2])
    
    def _generate_skewed(self) -> np.ndarray:
        """Generate right-skewed distribution (income, website visit duration)"""
        return np.random.gamma(shape=2, scale=2, size=self.population_size)
    
    def demonstrate_clt(self, distribution_type: str = 'exponential', 
                       sample_size: int = 30, num_samples: int = 1000) -> Dict:
        """
        Demonstrate CLT by repeatedly sampling and calculating means.
        
        Args:
            distribution_type: Type of population distribution
            sample_size: Size of each sample (n)
            num_samples: Number of samples to draw
            
        Returns:
            Dictionary with population, sample means, and statistics
        """
        # Generate population
        if distribution_type not in self.distributions:
            raise ValueError(f"Unknown distribution: {distribution_type}")
        
        population = self.distributions[distribution_type]()
        
        # Draw multiple samples and calculate means
        sample_means = []
        for _ in range(num_samples):
            sample = np.random.choice(population, size=sample_size, replace=False)
            sample_means.append(np.mean(sample))
        
        sample_means = np.array(sample_means)
        
        # Calculate statistics
        pop_mean = np.mean(population)
        pop_std = np.std(population, ddof=0)
        
        sample_means_mean = np.mean(sample_means)
        sample_means_std = np.std(sample_means, ddof=1)
        
        # Theoretical standard error
        theoretical_se = pop_std / np.sqrt(sample_size)
        
        return {
            'population': population,
            'sample_means': sample_means,
            'pop_mean': pop_mean,
            'pop_std': pop_std,
            'sample_means_mean': sample_means_mean,
            'sample_means_std': sample_means_std,
            'theoretical_se': theoretical_se,
            'sample_size': sample_size,
            'num_samples': num_samples
        }
    
    def visualize_clt(self, results: Dict, title: str = "Central Limit Theorem Demo"):
        """
        Create visualization comparing population and sampling distribution.
        
        Args:
            results: Dictionary from demonstrate_clt()
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Original Population Distribution
        axes[0].hist(results['population'], bins=50, density=True, 
                    alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(results['pop_mean'], color='red', linestyle='--', 
                       linewidth=2, label=f'Population Mean: {results["pop_mean"]:.2f}')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Original Population Distribution\n(May be non-normal)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Sampling Distribution (Sample Means)
        sample_means = results['sample_means']
        axes[1].hist(sample_means, bins=30, density=True, 
                    alpha=0.7, color='lightcoral', edgecolor='black',
                    label='Sample Means Distribution')
        
        # Overlay theoretical normal distribution
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        theoretical_normal = stats.norm.pdf(x, results['pop_mean'], results['theoretical_se'])
        axes[1].plot(x, theoretical_normal, 'g-', linewidth=2, 
                    label='Theoretical Normal Distribution')
        
        axes[1].axvline(results['sample_means_mean'], color='red', linestyle='--', 
                       linewidth=2, label=f'Mean of Sample Means: {results["sample_means_mean"]:.2f}')
        
        axes[1].set_xlabel('Sample Mean')
        axes[1].set_ylabel('Density')
        axes[1].set_title(f'Sampling Distribution of Mean\n(n={results["sample_size"]}, Always Normal!)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('clt_demonstration.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to 'clt_demonstration.png'")
        plt.show()
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"Population Statistics:")
        print(f"  Mean: {results['pop_mean']:.4f}")
        print(f"  Std Dev: {results['pop_std']:.4f}")
        print(f"\nSampling Distribution Statistics:")
        print(f"  Mean: {results['sample_means_mean']:.4f}")
        print(f"  Std Dev (Observed): {results['sample_means_std']:.4f}")
        print(f"  Std Error (Theoretical): {results['theoretical_se']:.4f}")
        print(f"\nCLT Verification:")
        print(f"  Sample means mean ≈ Population mean: {abs(results['sample_means_mean'] - results['pop_mean']) < 0.1}")
        print(f"  Observed SE ≈ Theoretical SE: {abs(results['sample_means_std'] - results['theoretical_se']) < 0.05}")
        print(f"{'='*60}\n")


class MLConfidenceCalculator:
    """
    Calculate confidence intervals for ML model performance metrics.
    Uses CLT to provide statistical rigor to model evaluation.
    """
    
    @staticmethod
    def calculate_accuracy_ci(predictions: np.ndarray, labels: np.ndarray, 
                             confidence: float = 0.95) -> Dict:
        """
        Calculate confidence interval for model accuracy.
        
        Args:
            predictions: Model predictions (0 or 1)
            labels: True labels (0 or 1)
            confidence: Confidence level (0.95 for 95%)
            
        Returns:
            Dictionary with accuracy, standard error, and confidence interval
        """
        n = len(predictions)
        correct = np.sum(predictions == labels)
        accuracy = correct / n
        
        # Standard error for proportion
        se = np.sqrt(accuracy * (1 - accuracy) / n)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Confidence interval
        margin_of_error = z_score * se
        ci_lower = max(0, accuracy - margin_of_error)
        ci_upper = min(1, accuracy + margin_of_error)
        
        return {
            'accuracy': accuracy,
            'standard_error': se,
            'confidence_level': confidence,
            'margin_of_error': margin_of_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'sample_size': n
        }
    
    @staticmethod
    def compare_models(model_a_results: Dict, model_b_results: Dict) -> Dict:
        """
        Compare two models using confidence intervals.
        
        Args:
            model_a_results: Results from calculate_accuracy_ci for model A
            model_b_results: Results from calculate_accuracy_ci for model B
            
        Returns:
            Dictionary with comparison results
        """
        # Check for overlap in confidence intervals
        a_lower, a_upper = model_a_results['ci_lower'], model_a_results['ci_upper']
        b_lower, b_upper = model_b_results['ci_lower'], model_b_results['ci_upper']
        
        overlap = not (a_upper < b_lower or b_upper < a_lower)
        
        # Calculate difference in accuracy
        diff = model_a_results['accuracy'] - model_b_results['accuracy']
        
        # Combined standard error for difference
        se_diff = np.sqrt(model_a_results['standard_error']**2 + 
                         model_b_results['standard_error']**2)
        
        # Is difference statistically significant?
        z_statistic = diff / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        significant = p_value < 0.05
        
        return {
            'model_a_accuracy': model_a_results['accuracy'],
            'model_b_accuracy': model_b_results['accuracy'],
            'difference': diff,
            'intervals_overlap': overlap,
            'statistically_significant': significant,
            'p_value': p_value,
            'z_statistic': z_statistic
        }
    
    @staticmethod
    def visualize_comparison(model_a_results: Dict, model_b_results: Dict,
                            model_a_name: str = "Model A", 
                            model_b_name: str = "Model B"):
        """
        Visualize confidence intervals for two models.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = [model_a_name, model_b_name]
        accuracies = [model_a_results['accuracy'], model_b_results['accuracy']]
        errors = [model_a_results['margin_of_error'], model_b_results['margin_of_error']]
        colors = ['steelblue', 'lightcoral']
        
        y_pos = np.arange(len(models))
        
        # Plot bars with error bars
        bars = ax.barh(y_pos, accuracies, xerr=errors, color=colors, 
                      alpha=0.7, capsize=10, edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_title('Model Accuracy Comparison with 95% Confidence Intervals', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for i, (acc, err) in enumerate(zip(accuracies, errors)):
            ax.text(acc, i, f' {acc:.3f} ± {err:.3f}', 
                   va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to 'model_comparison.png'")
        plt.show()


class ABTestCalculator:
    """
    Calculate required sample sizes for A/B tests comparing ML models.
    Uses power analysis based on CLT.
    """
    
    @staticmethod
    def calculate_sample_size(baseline_rate: float, minimum_detectable_effect: float,
                             alpha: float = 0.05, power: float = 0.80) -> Dict:
        """
        Calculate required sample size for A/B test.
        
        Args:
            baseline_rate: Current model's success rate (e.g., 0.75 for 75% accuracy)
            minimum_detectable_effect: Smallest improvement to detect (e.g., 0.02 for 2%)
            alpha: Significance level (Type I error rate)
            power: Statistical power (1 - Type II error rate)
            
        Returns:
            Dictionary with sample size requirements
        """
        # New rate after improvement
        new_rate = baseline_rate + minimum_detectable_effect
        
        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed test
        z_beta = stats.norm.ppf(power)
        
        # Pooled proportion
        pooled_p = (baseline_rate + new_rate) / 2
        
        # Sample size formula for proportions
        numerator = (z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) + 
                    z_beta * np.sqrt(baseline_rate * (1 - baseline_rate) + 
                                    new_rate * (1 - new_rate)))**2
        denominator = (minimum_detectable_effect)**2
        
        n_per_group = int(np.ceil(numerator / denominator))
        total_n = 2 * n_per_group
        
        return {
            'sample_size_per_group': n_per_group,
            'total_sample_size': total_n,
            'baseline_rate': baseline_rate,
            'new_rate': new_rate,
            'minimum_detectable_effect': minimum_detectable_effect,
            'alpha': alpha,
            'power': power
        }
    
    @staticmethod
    def visualize_power_analysis(baseline_rate: float, 
                                effect_sizes: List[float] = [0.01, 0.02, 0.03, 0.05, 0.10]):
        """
        Visualize how sample size changes with effect size.
        """
        sample_sizes = []
        for effect in effect_sizes:
            result = ABTestCalculator.calculate_sample_size(baseline_rate, effect)
            sample_sizes.append(result['total_sample_size'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot([e*100 for e in effect_sizes], sample_sizes, 
               marker='o', linewidth=2, markersize=10, color='steelblue')
        ax.fill_between([e*100 for e in effect_sizes], sample_sizes, 
                       alpha=0.3, color='steelblue')
        
        ax.set_xlabel('Minimum Detectable Effect (%)', fontsize=12)
        ax.set_ylabel('Total Sample Size Required', fontsize=12)
        ax.set_title(f'A/B Test Sample Size Requirements\n(Baseline: {baseline_rate*100:.1f}%, Power: 80%, α: 0.05)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for effect, n in zip(effect_sizes, sample_sizes):
            ax.annotate(f'{n:,}', xy=(effect*100, n), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('power_analysis.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved power analysis to 'power_analysis.png'")
        plt.show()


def main():
    """
    Run all CLT demonstrations and calculations.
    """
    print("="*70)
    print("DAY 29: CENTRAL LIMIT THEOREM - PRODUCTION ML TOOLKIT")
    print("="*70)
    
    # Part 1: CLT Demonstration
    print("\n1. DEMONSTRATING CENTRAL LIMIT THEOREM")
    print("-" * 70)
    
    simulator = CentralLimitTheoremSimulator()
    
    # Show CLT with exponential distribution (common in ML: response times, wait times)
    print("\nSimulating with exponential distribution (like server response times)...")
    results = simulator.demonstrate_clt(
        distribution_type='exponential',
        sample_size=30,
        num_samples=1000
    )
    simulator.visualize_clt(results, "CLT: Exponential → Normal")
    
    # Part 2: ML Model Confidence Intervals
    print("\n2. CALCULATING ML MODEL CONFIDENCE INTERVALS")
    print("-" * 70)
    
    # Simulate model predictions
    np.random.seed(42)
    test_size = 1000
    
    # Model A: 85% accurate
    model_a_preds = np.random.binomial(1, 0.85, test_size)
    true_labels = np.ones(test_size)
    
    model_a_results = MLConfidenceCalculator.calculate_accuracy_ci(
        model_a_preds, true_labels
    )
    
    print(f"\nModel A Performance:")
    print(f"  Accuracy: {model_a_results['accuracy']*100:.2f}%")
    print(f"  Standard Error: {model_a_results['standard_error']:.4f}")
    print(f"  95% CI: [{model_a_results['ci_lower']*100:.2f}%, {model_a_results['ci_upper']*100:.2f}%]")
    print(f"  Margin of Error: ±{model_a_results['margin_of_error']*100:.2f}%")
    
    # Model B: 83% accurate
    model_b_preds = np.random.binomial(1, 0.83, test_size)
    model_b_results = MLConfidenceCalculator.calculate_accuracy_ci(
        model_b_preds, true_labels
    )
    
    print(f"\nModel B Performance:")
    print(f"  Accuracy: {model_b_results['accuracy']*100:.2f}%")
    print(f"  Standard Error: {model_b_results['standard_error']:.4f}")
    print(f"  95% CI: [{model_b_results['ci_lower']*100:.2f}%, {model_b_results['ci_upper']*100:.2f}%]")
    print(f"  Margin of Error: ±{model_b_results['margin_of_error']*100:.2f}%")
    
    # Compare models
    comparison = MLConfidenceCalculator.compare_models(model_a_results, model_b_results)
    
    print(f"\nModel Comparison:")
    print(f"  Difference: {comparison['difference']*100:.2f}%")
    print(f"  CIs Overlap: {comparison['intervals_overlap']}")
    print(f"  Statistically Significant: {comparison['statistically_significant']}")
    print(f"  P-value: {comparison['p_value']:.4f}")
    
    MLConfidenceCalculator.visualize_comparison(
        model_a_results, model_b_results,
        "Model A (New)", "Model B (Baseline)"
    )
    
    # Part 3: A/B Test Sample Size
    print("\n3. A/B TEST SAMPLE SIZE CALCULATION")
    print("-" * 70)
    
    baseline = 0.75  # Current model: 75% accuracy
    mde = 0.02  # Want to detect 2% improvement
    
    sample_calc = ABTestCalculator.calculate_sample_size(baseline, mde)
    
    print(f"\nA/B Test Requirements:")
    print(f"  Baseline Rate: {baseline*100:.1f}%")
    print(f"  Target Rate: {sample_calc['new_rate']*100:.1f}%")
    print(f"  Minimum Detectable Effect: {mde*100:.1f}%")
    print(f"  Sample Size per Group: {sample_calc['sample_size_per_group']:,}")
    print(f"  Total Sample Size: {sample_calc['total_sample_size']:,}")
    print(f"  Power: {sample_calc['power']*100:.0f}%")
    print(f"  Significance Level (α): {sample_calc['alpha']:.2f}")
    
    print("\nGenerating power analysis curve...")
    ABTestCalculator.visualize_power_analysis(baseline)
    
    print("\n" + "="*70)
    print("LESSON COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • CLT makes sample means normally distributed regardless of source")
    print("  • Confidence intervals quantify uncertainty in ML metrics")
    print("  • Proper sample sizing prevents underpowered experiments")
    print("  • Production AI systems use these tools for every model evaluation")
    print("\nNext: Day 30 - Apply all statistics concepts to real dataset analysis!")


if __name__ == "__main__":
    main()
