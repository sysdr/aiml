#!/bin/bash

# Day 25: Random Variables and Probability Distributions - Lesson Files Generator
# This script creates all necessary files for the lesson

echo "Creating Day 25 lesson files..."

# Create env_setup.sh
cat > env_setup.sh << 'EOF'
#!/bin/bash

echo "Setting up environment for Day 25: Random Variables and Probability Distributions"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

echo "Detected Python version: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"; then
    echo "Error: Python 3.11 or higher required"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "Then run the lesson code:"
echo "python lesson_code.py"
EOF

chmod +x env_setup.sh

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy>=1.26.0
scipy>=1.11.0
matplotlib>=3.8.0
jupyter>=1.0.0
pytest>=7.4.0
EOF

# Create lesson_code.py
cat > lesson_code.py << 'EOF'
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
EOF

# Create test_lesson.py
cat > test_lesson.py << 'EOF'
"""
Tests for Day 25: Random Variables and Probability Distributions
Run with: pytest test_lesson.py -v
"""

import pytest
import numpy as np
from lesson_code import DistributionSimulator, AIApplicationExamples


class TestDistributionSimulator:
    """Test the DistributionSimulator class"""
    
    @pytest.fixture
    def simulator(self):
        """Create a simulator instance for testing"""
        return DistributionSimulator(random_seed=42)
    
    def test_bernoulli_trial(self, simulator):
        """Test Bernoulli trial generation"""
        outcomes = simulator.bernoulli_trial(p=0.5, n_trials=10000)
        
        assert len(outcomes) == 10000
        assert set(outcomes).issubset({0, 1})
        # Check if proportion is close to 0.5 (within 5%)
        proportion = outcomes.sum() / len(outcomes)
        assert 0.45 <= proportion <= 0.55
    
    def test_binomial_experiment(self, simulator):
        """Test binomial distribution generation"""
        results = simulator.binomial_experiment(n=10, p=0.5, experiments=10000)
        
        assert len(results) == 10000
        # Mean should be close to n*p = 10*0.5 = 5
        assert 4.5 <= results.mean() <= 5.5
        # All results should be between 0 and n
        assert results.min() >= 0
        assert results.max() <= 10
    
    def test_normal_samples(self, simulator):
        """Test normal distribution sampling"""
        samples = simulator.normal_samples(mu=0, sigma=1, n_samples=10000)
        
        assert len(samples) == 10000
        # Mean should be close to 0
        assert -0.1 <= samples.mean() <= 0.1
        # Std should be close to 1
        assert 0.9 <= samples.std() <= 1.1
    
    def test_central_limit_theorem(self, simulator):
        """Test CLT demonstration"""
        sample_means = simulator.demonstrate_central_limit_theorem(
            n_samples=30, iterations=1000
        )
        
        assert len(sample_means) == 1000
        # Means should be approximately 0.5 (uniform 0-1)
        assert 0.45 <= sample_means.mean() <= 0.55
    
    def test_model_confidence_analysis(self, simulator):
        """Test model confidence simulation"""
        well_cal, over_conf = simulator.model_confidence_analysis(n_predictions=1000)
        
        assert len(well_cal) == 1000
        assert len(over_conf) == 1000
        # All confidences should be between 0 and 1
        assert 0 <= well_cal.min() and well_cal.max() <= 1
        assert 0 <= over_conf.min() and over_conf.max() <= 1


class TestAIApplicationExamples:
    """Test AI application examples"""
    
    def test_neural_network_initialization(self):
        """Test weight initialization"""
        xavier_weights, he_weights = AIApplicationExamples.neural_network_initialization(
            (784, 128)
        )
        
        # Check dimensions
        assert xavier_weights.shape == (784, 128)
        assert he_weights.shape == (784, 128)
        
        # Xavier std should be sqrt(2/(in+out))
        expected_xavier_std = np.sqrt(2.0 / (784 + 128))
        assert 0.8 * expected_xavier_std <= xavier_weights.std() <= 1.2 * expected_xavier_std
        
        # He std should be sqrt(2/in)
        expected_he_std = np.sqrt(2.0 / 784)
        assert 0.8 * expected_he_std <= he_weights.std() <= 1.2 * expected_he_std
    
    def test_dropout_simulation(self):
        """Test dropout simulation"""
        active_counts = AIApplicationExamples.dropout_simulation(
            n_neurons=100, dropout_rate=0.5, trials=100
        )
        
        assert len(active_counts) == 100
        # Mean active neurons should be around 50 (100 * 0.5)
        mean_active = np.mean(active_counts)
        assert 40 <= mean_active <= 60
    
    def test_uncertainty_quantification(self):
        """Test uncertainty quantification"""
        confident, uncertain = AIApplicationExamples.uncertainty_quantification(
            n_samples=1000
        )
        
        assert len(confident) == 1000
        assert len(uncertain) == 1000
        
        # Confident should have smaller std than uncertain
        assert confident.std() < uncertain.std()


class TestStatisticalProperties:
    """Test statistical properties of distributions"""
    
    def test_bernoulli_expectation(self):
        """Test that Bernoulli expectation equals p"""
        np.random.seed(42)
        p = 0.7
        samples = np.random.binomial(1, p, 100000)
        assert abs(samples.mean() - p) < 0.01
    
    def test_binomial_mean_variance(self):
        """Test binomial mean and variance formulas"""
        np.random.seed(42)
        n, p = 100, 0.3
        samples = np.random.binomial(n, p, 100000)
        
        # E[X] = np
        assert abs(samples.mean() - n*p) < 0.5
        
        # Var[X] = np(1-p)
        assert abs(samples.var() - n*p*(1-p)) < 1.0
    
    def test_normal_68_95_99_rule(self):
        """Test empirical rule for normal distribution"""
        np.random.seed(42)
        samples = np.random.normal(0, 1, 100000)
        
        # ~68% within 1 std
        within_1std = np.sum(np.abs(samples) <= 1) / len(samples)
        assert 0.66 <= within_1std <= 0.70
        
        # ~95% within 2 stds
        within_2std = np.sum(np.abs(samples) <= 2) / len(samples)
        assert 0.94 <= within_2std <= 0.96


def test_reproducibility():
    """Test that random seed ensures reproducibility"""
    sim1 = DistributionSimulator(random_seed=42)
    outcomes1 = sim1.bernoulli_trial(p=0.5, n_trials=100)
    
    sim2 = DistributionSimulator(random_seed=42)
    outcomes2 = sim2.bernoulli_trial(p=0.5, n_trials=100)
    
    assert np.array_equal(outcomes1, outcomes2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

# Create README.md
cat > README.md << 'EOF'
# Day 25: Random Variables and Probability Distributions

## 🎯 Learning Objectives

By the end of this lesson, you will:
- Understand discrete vs continuous random variables
- Work with Bernoulli, Binomial, and Normal distributions in Python
- Connect probability distributions to real AI/ML applications
- Implement distribution simulations from scratch
- Visualize and analyze distribution properties

## 🚀 Quick Start

### 1. Setup Environment

```bash
chmod +x env_setup.sh
./env_setup.sh
source venv/bin/activate
```

### 2. Run the Lesson Code

```bash
python lesson_code.py
```

This will:
- Generate samples from various distributions
- Demonstrate the Central Limit Theorem
- Show real AI applications (weight init, dropout, uncertainty)
- Create comprehensive visualizations

### 3. Run Tests

```bash
pytest test_lesson.py -v
```

## 📚 What You'll Learn

### Core Concepts

1. **Random Variables**
   - Discrete vs Continuous
   - PMF vs PDF
   - Expected value and variance

2. **Key Distributions**
   - Bernoulli: Binary outcomes (classification)
   - Binomial: Counting successes (A/B testing)
   - Normal: The bell curve (everywhere in ML!)

3. **AI/ML Applications**
   - Neural network weight initialization
   - Dropout regularization
   - Uncertainty quantification
   - Model calibration

### Real-World Connections

- **Weight Initialization**: Xavier and He initialization use carefully chosen normal distributions
- **Dropout**: Each neuron follows a Bernoulli distribution during training
- **Confidence Scores**: Well-calibrated models have specific confidence distributions
- **Central Limit Theorem**: Why normal distributions appear everywhere

## 📊 Output Files

After running the code, you'll have:
- `distributions_visualization.png` - Comprehensive visualization of all concepts
- Test results showing statistical properties

## 🔍 Key Insights

1. **Distributions are specifications for uncertainty** - They tell you not just what's likely, but HOW likely
2. **Normal distribution is special** - CLT explains why it appears everywhere in nature and ML
3. **Proper initialization matters** - Wrong distribution = exploding/vanishing gradients
4. **Uncertainty quantification is critical** - Production AI needs to know when it's unsure

## 🎓 Prerequisites

- Day 24: Conditional Probability and Bayes' Theorem
- Basic Python and NumPy
- Understanding of probability basics

## ➡️ Next Steps

Tomorrow: Day 26 - Descriptive Statistics (Mean, Median, Mode)

We'll learn how to summarize distributions with single numbers and when each measure is appropriate.

## 📖 Additional Resources

- NumPy random module: https://numpy.org/doc/stable/reference/random/
- SciPy stats: https://docs.scipy.org/doc/scipy/reference/stats.html
- Xavier initialization paper: "Understanding the difficulty of training deep feedforward neural networks"

## 💡 Pro Tips

1. Always set random seeds for reproducibility
2. Visualize distributions before using them
3. Check statistical properties match theory
4. In production, monitor your model's output distributions
5. Use appropriate distributions for initialization based on activation functions

## 🐛 Troubleshooting

**Issue**: Plots not showing
**Solution**: Run in Jupyter notebook or ensure matplotlib backend is configured

**Issue**: Import errors
**Solution**: Make sure virtual environment is activated: `source venv/bin/activate`

**Issue**: Tests failing
**Solution**: Statistical tests can occasionally fail due to randomness. Run again or increase sample sizes.

---

Built with ❤️ for the 180-Day AI/ML Course
EOF

echo ""
echo "✅ All files created successfully!"
echo ""
echo "📁 Files created:"
echo "   - env_setup.sh (environment setup script)"
echo "   - requirements.txt (Python dependencies)"
echo "   - lesson_code.py (main implementation)"
echo "   - test_lesson.py (test suite)"
echo "   - README.md (quick start guide)"
echo ""
echo "🚀 To get started:"
echo "   1. chmod +x env_setup.sh && ./env_setup.sh"
echo "   2. source venv/bin/activate"
echo "   3. python lesson_code.py"
echo ""