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
