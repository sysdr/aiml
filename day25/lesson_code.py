"""
Day 25: Random Variables and Probability Distributions
A comprehensive implementation of probability distributions for AI/ML
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class DistributionSimulator:
    """
    Simulates and visualizes probability distributions commonly used in AI/ML
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize with reproducible random seed for consistency"""
        np.random.seed(random_seed)
        self.distributions = {}
    
    def bernoulli_trial(self, p: float = 0.5, n_trials: int = 1000) -> np.ndarray:
        """
        Simulate Bernoulli trials - fundamental for binary classification
        
        Args:
            p: Probability of success (like probability of positive class)
            n_trials: Number of trials (like number of predictions)
            
        Returns:
            Array of 0s and 1s representing outcomes
            
        Real AI use case: Each prediction in binary classification is a Bernoulli trial
        """
        outcomes = np.random.binomial(1, p, n_trials)
        self.distributions['bernoulli'] = outcomes
        
        print(f"\n🎲 Bernoulli Distribution (p={p})")
        print(f"   Total trials: {n_trials}")
        print(f"   Successes (1s): {outcomes.sum()} ({outcomes.sum()/n_trials*100:.1f}%)")
        print(f"   Failures (0s): {n_trials - outcomes.sum()} ({(1-outcomes.sum()/n_trials)*100:.1f}%)")
        print(f"   Theoretical expectation: {p*100:.1f}%")
        
        return outcomes
    
    def binomial_experiment(self, n: int = 10, p: float = 0.5, 
                           experiments: int = 10000) -> np.ndarray:
        """
        Simulate binomial distribution - counting successes in multiple trials
        
        Args:
            n: Number of trials per experiment (e.g., users shown an ad)
            p: Success probability (e.g., click-through rate)
            experiments: Number of experiments to run
            
        Returns:
            Array of success counts
            
        Real AI use case: A/B testing click rates, user engagement metrics
        """
        results = np.random.binomial(n, p, experiments)
        self.distributions['binomial'] = results
        
        print(f"\n🎯 Binomial Distribution (n={n}, p={p})")
        print(f"   Number of experiments: {experiments}")
        print(f"   Mean successes: {results.mean():.2f}")
        print(f"   Theoretical mean: {n*p:.2f}")
        print(f"   Standard deviation: {results.std():.2f}")
        print(f"   Theoretical std: {np.sqrt(n*p*(1-p)):.2f}")
        
        return results
    
    def normal_samples(self, mu: float = 0, sigma: float = 1, 
                       n_samples: int = 10000) -> np.ndarray:
        """
        Generate samples from normal distribution - most important in ML
        
        Args:
            mu: Mean (center of distribution)
            sigma: Standard deviation (spread)
            n_samples: Number of samples
            
        Returns:
            Array of samples
            
        Real AI use case: Neural network weight initialization, noise modeling
        """
        samples = np.random.normal(mu, sigma, n_samples)
        self.distributions['normal'] = samples
        
        print(f"\n📊 Normal Distribution (μ={mu}, σ={sigma})")
        print(f"   Samples: {n_samples}")
        print(f"   Sample mean: {samples.mean():.3f}")
        print(f"   Sample std: {samples.std():.3f}")
        print(f"   Min: {samples.min():.2f}, Max: {samples.max():.2f}")
        print(f"   ~68% within 1σ: {np.sum(np.abs(samples-mu) <= sigma)/n_samples*100:.1f}%")
        
        return samples
    
    def demonstrate_central_limit_theorem(self, n_samples: int = 30, 
                                         iterations: int = 1000):
        """
        Show how averages of random samples approach normal distribution
        
        This is WHY normal distribution appears everywhere in ML!
        """
        print(f"\n🌟 Central Limit Theorem Demonstration")
        print(f"   Taking means of {n_samples} uniform samples, {iterations} times")
        
        # Sample from uniform distribution (not normal!)
        sample_means = []
        for _ in range(iterations):
            samples = np.random.uniform(0, 1, n_samples)
            sample_means.append(samples.mean())
        
        sample_means = np.array(sample_means)
        
        print(f"   Mean of sample means: {sample_means.mean():.3f}")
        print(f"   Theoretical mean: 0.500")
        print(f"   Std of sample means: {sample_means.std():.3f}")
        print(f"   Theoretical std: {1/np.sqrt(12*n_samples):.3f}")
        print(f"   ✅ Even though we sampled from UNIFORM, means are NORMAL!")
        
        return sample_means
    
    def model_confidence_analysis(self, n_predictions: int = 1000):
        """
        Simulate and analyze ML model confidence scores
        
        Real scenario: Checking if your classifier is well-calibrated
        """
        print(f"\n🤖 AI Model Confidence Distribution Analysis")
        
        # Simulate a well-calibrated model (beta distribution)
        well_calibrated = np.random.beta(2, 2, n_predictions)
        
        # Simulate an overconfident model (pushes toward extremes)
        overconfident = np.random.beta(0.5, 0.5, n_predictions)
        
        print(f"\n   Well-calibrated model:")
        print(f"   Mean confidence: {well_calibrated.mean():.3f}")
        print(f"   Confidences > 0.9: {np.sum(well_calibrated > 0.9)/n_predictions*100:.1f}%")
        
        print(f"\n   Overconfident model:")
        print(f"   Mean confidence: {overconfident.mean():.3f}")
        print(f"   Confidences > 0.9: {np.sum(overconfident > 0.9)/n_predictions*100:.1f}%")
        print(f"   ⚠️  High extremes suggest overconfidence - dangerous in production!")
        
        return well_calibrated, overconfident
    
    def visualize_all(self):
        """Create comprehensive visualization of all distributions"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Day 25: Probability Distributions in AI/ML', 
                     fontsize=16, fontweight='bold')
        
        # 1. Bernoulli outcomes
        if 'bernoulli' in self.distributions:
            ax = axes[0, 0]
            outcomes = self.distributions['bernoulli']
            values, counts = np.unique(outcomes, return_counts=True)
            ax.bar(values, counts, color=['#3498db', '#e74c3c'], alpha=0.7)
            ax.set_title('Bernoulli Distribution\n(Binary Classification)', fontweight='bold')
            ax.set_xlabel('Outcome (0=Negative, 1=Positive)')
            ax.set_ylabel('Frequency')
            ax.set_xticks([0, 1])
            
        # 2. Binomial distribution
        if 'binomial' in self.distributions:
            ax = axes[0, 1]
            results = self.distributions['binomial']
            ax.hist(results, bins=30, density=True, alpha=0.7, 
                   color='#2ecc71', edgecolor='black')
            ax.set_title('Binomial Distribution\n(A/B Test Results)', fontweight='bold')
            ax.set_xlabel('Number of Successes')
            ax.set_ylabel('Probability Density')
            
        # 3. Normal distribution
        if 'normal' in self.distributions:
            ax = axes[0, 2]
            samples = self.distributions['normal']
            ax.hist(samples, bins=50, density=True, alpha=0.7, 
                   color='#9b59b6', edgecolor='black')
            
            # Overlay theoretical normal curve
            x = np.linspace(samples.min(), samples.max(), 100)
            ax.plot(x, stats.norm.pdf(x, samples.mean(), samples.std()), 
                   'r-', linewidth=2, label='Theoretical')
            ax.set_title('Normal Distribution\n(Weight Initialization)', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Probability Density')
            ax.legend()
        
        # 4. Central Limit Theorem
        ax = axes[1, 0]
        sample_means = self.demonstrate_central_limit_theorem()
        ax.hist(sample_means, bins=40, density=True, alpha=0.7, 
               color='#f39c12', edgecolor='black')
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        ax.plot(x, stats.norm.pdf(x, sample_means.mean(), sample_means.std()), 
               'r-', linewidth=2)
        ax.set_title('Central Limit Theorem\n(Why Normal Appears Everywhere)', 
                    fontweight='bold')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Density')
        
        # 5. Model confidence comparison
        ax = axes[1, 1]
        well_cal, over_conf = self.model_confidence_analysis()
        ax.hist(well_cal, bins=30, alpha=0.5, label='Well-calibrated', 
               color='#27ae60')
        ax.hist(over_conf, bins=30, alpha=0.5, label='Overconfident', 
               color='#e74c3c')
        ax.set_title('Model Confidence Distributions\n(Calibration Check)', 
                    fontweight='bold')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # 6. Comparing PMF vs PDF concept
        ax = axes[1, 2]
        # Discrete PMF (Poisson)
        x_discrete = np.arange(0, 20)
        pmf = stats.poisson.pmf(x_discrete, mu=5)
        ax.bar(x_discrete, pmf, alpha=0.6, label='Discrete (PMF)', 
              color='#3498db')
        
        # Continuous PDF (Normal)
        x_continuous = np.linspace(0, 20, 100)
        pdf = stats.norm.pdf(x_continuous, 10, 2)
        ax.plot(x_continuous, pdf, 'r-', linewidth=2, 
               label='Continuous (PDF)')
        ax.set_title('PMF vs PDF\n(Discrete vs Continuous)', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability (Mass/Density)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('distributions_visualization.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as 'distributions_visualization.png'")
        plt.show()


class AIApplicationExamples:
    """
    Real-world AI scenarios using probability distributions
    """
    
    @staticmethod
    def neural_network_initialization(layer_size: Tuple[int, int]):
        """
        Demonstrate proper weight initialization using distributions
        
        Args:
            layer_size: (input_dim, output_dim)
        """
        input_dim, output_dim = layer_size
        
        print(f"\n🧠 Neural Network Weight Initialization")
        print(f"   Layer: {input_dim} → {output_dim}")
        
        # Xavier/Glorot initialization
        xavier_std = np.sqrt(2.0 / (input_dim + output_dim))
        xavier_weights = np.random.normal(0, xavier_std, (input_dim, output_dim))
        
        # He initialization (for ReLU)
        he_std = np.sqrt(2.0 / input_dim)
        he_weights = np.random.normal(0, he_std, (input_dim, output_dim))
        
        print(f"\n   Xavier initialization: N(0, {xavier_std:.4f})")
        print(f"   Weight range: [{xavier_weights.min():.3f}, {xavier_weights.max():.3f}]")
        print(f"   Weight std: {xavier_weights.std():.4f}")
        
        print(f"\n   He initialization: N(0, {he_std:.4f})")
        print(f"   Weight range: [{he_weights.min():.3f}, {he_weights.max():.3f}]")
        print(f"   Weight std: {he_weights.std():.4f}")
        
        return xavier_weights, he_weights
    
    @staticmethod
    def dropout_simulation(n_neurons: int = 100, dropout_rate: float = 0.5, 
                          trials: int = 10):
        """
        Simulate dropout regularization as Bernoulli random variables
        
        Args:
            n_neurons: Number of neurons in layer
            dropout_rate: Probability of dropping a neuron
            trials: Number of forward passes to simulate
        """
        print(f"\n🎯 Dropout Regularization Simulation")
        print(f"   Neurons: {n_neurons}")
        print(f"   Dropout rate: {dropout_rate}")
        print(f"   Keep probability: {1-dropout_rate}")
        
        active_counts = []
        for trial in range(trials):
            # Each neuron: Bernoulli(p = 1-dropout_rate)
            active_mask = np.random.binomial(1, 1-dropout_rate, n_neurons)
            active_counts.append(active_mask.sum())
        
        print(f"\n   Active neurons per forward pass:")
        print(f"   Mean: {np.mean(active_counts):.1f}")
        print(f"   Expected: {n_neurons * (1-dropout_rate):.1f}")
        print(f"   Range: [{min(active_counts)}, {max(active_counts)}]")
        print(f"   ✅ Random dropout creates ensemble effect!")
        
        return active_counts
    
    @staticmethod
    def uncertainty_quantification(n_samples: int = 1000):
        """
        Demonstrate uncertainty in predictions using distributions
        
        Models should output distributions, not just point estimates!
        """
        print(f"\n📊 Uncertainty Quantification in Predictions")
        
        # Confident prediction (narrow distribution)
        confident = np.random.normal(0.85, 0.05, n_samples)
        
        # Uncertain prediction (wide distribution)
        uncertain = np.random.normal(0.60, 0.20, n_samples)
        
        print(f"\n   Confident prediction:")
        print(f"   Mean: {confident.mean():.3f} ± {confident.std():.3f}")
        print(f"   95% interval: [{np.percentile(confident, 2.5):.3f}, "
              f"{np.percentile(confident, 97.5):.3f}]")
        
        print(f"\n   Uncertain prediction:")
        print(f"   Mean: {uncertain.mean():.3f} ± {uncertain.std():.3f}")
        print(f"   95% interval: [{np.percentile(uncertain, 2.5):.3f}, "
              f"{np.percentile(uncertain, 97.5):.3f}]")
        print(f"   ⚠️  Wide uncertainty → Model isn't sure → Need more data!")
        
        return confident, uncertain


def main():
    """
    Main execution: Run all demonstrations
    """
    print("=" * 70)
    print("Day 25: Random Variables and Probability Distributions")
    print("Understanding Uncertainty in AI/ML Systems")
    print("=" * 70)
    
    # Initialize simulator
    sim = DistributionSimulator(random_seed=42)
    
    # 1. Basic distributions
    print("\n" + "=" * 70)
    print("SECTION 1: Fundamental Distributions")
    print("=" * 70)
    
    sim.bernoulli_trial(p=0.7, n_trials=1000)
    sim.binomial_experiment(n=100, p=0.3, experiments=10000)
    sim.normal_samples(mu=0, sigma=1, n_samples=10000)
    
    # 2. AI Applications
    print("\n" + "=" * 70)
    print("SECTION 2: AI/ML Applications")
    print("=" * 70)
    
    ai_examples = AIApplicationExamples()
    ai_examples.neural_network_initialization((784, 128))
    ai_examples.dropout_simulation(n_neurons=256, dropout_rate=0.5)
    ai_examples.uncertainty_quantification()
    
    # 3. Visualize everything
    print("\n" + "=" * 70)
    print("SECTION 3: Comprehensive Visualization")
    print("=" * 70)
    
    sim.visualize_all()
    
    print("\n" + "=" * 70)
    print("✅ Day 25 Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Random variables map outcomes to numbers")
    print("2. Different distributions model different types of uncertainty")
    print("3. Normal distribution appears everywhere due to CLT")
    print("4. AI systems use distributions for initialization, regularization, and uncertainty")
    print("\nNext: Day 26 - Descriptive Statistics (Mean, Median, Mode)")


if __name__ == "__main__":
    main()
