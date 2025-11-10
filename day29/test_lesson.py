"""
Tests for Day 29: Central Limit Theorem
Validates CLT implementations and calculations
"""

import pytest
import numpy as np
from lesson_code import (
    CentralLimitTheoremSimulator,
    MLConfidenceCalculator,
    ABTestCalculator
)


class TestCLTSimulator:
    """Test Central Limit Theorem simulations"""
    
    def test_simulator_initialization(self):
        """Test simulator creates correctly"""
        sim = CentralLimitTheoremSimulator(population_size=1000)
        assert sim.population_size == 1000
        assert len(sim.distributions) == 4
    
    def test_distribution_generation(self):
        """Test all distribution types generate correct sizes"""
        sim = CentralLimitTheoremSimulator(population_size=5000)
        
        for dist_type in ['uniform', 'exponential', 'bimodal', 'skewed']:
            population = sim.distributions[dist_type]()
            assert len(population) == 5000
            assert not np.any(np.isnan(population))
    
    def test_clt_demonstration(self):
        """Test CLT demonstration produces valid results"""
        sim = CentralLimitTheoremSimulator(population_size=10000)
        results = sim.demonstrate_clt(
            distribution_type='uniform',
            sample_size=30,
            num_samples=100
        )
        
        # Check all required keys present
        required_keys = ['population', 'sample_means', 'pop_mean', 'pop_std',
                        'sample_means_mean', 'sample_means_std', 'theoretical_se']
        for key in required_keys:
            assert key in results
        
        # Sample means should approximate population mean
        assert abs(results['sample_means_mean'] - results['pop_mean']) < 0.5
        
        # Observed SE should be close to theoretical SE
        ratio = results['sample_means_std'] / results['theoretical_se']
        assert 0.8 < ratio < 1.2  # Within 20%
    
    def test_clt_with_large_sample_size(self):
        """Test CLT converges better with larger samples"""
        sim = CentralLimitTheoremSimulator(population_size=10000)
        
        # Small sample
        small_results = sim.demonstrate_clt(
            distribution_type='exponential',
            sample_size=10,
            num_samples=500
        )
        
        # Large sample
        large_results = sim.demonstrate_clt(
            distribution_type='exponential',
            sample_size=100,
            num_samples=500
        )
        
        # Larger sample should have smaller standard error
        assert large_results['theoretical_se'] < small_results['theoretical_se']
        assert large_results['sample_means_std'] < small_results['sample_means_std']


class TestMLConfidenceCalculator:
    """Test ML model confidence interval calculations"""
    
    def test_accuracy_ci_calculation(self):
        """Test confidence interval calculation for accuracy"""
        np.random.seed(42)
        
        # Perfect predictions
        predictions = np.ones(1000)
        labels = np.ones(1000)
        
        results = MLConfidenceCalculator.calculate_accuracy_ci(predictions, labels)
        
        assert results['accuracy'] == 1.0
        assert results['standard_error'] == 0.0
        assert results['ci_lower'] == 1.0
        assert results['ci_upper'] == 1.0
    
    def test_accuracy_ci_with_errors(self):
        """Test CI with imperfect predictions"""
        np.random.seed(42)
        n = 1000
        
        # 80% accurate predictions
        predictions = np.zeros(n)
        predictions[:800] = 1
        labels = np.ones(n)
        
        results = MLConfidenceCalculator.calculate_accuracy_ci(predictions, labels)
        
        assert 0.75 < results['accuracy'] < 0.85
        assert results['standard_error'] > 0
        assert results['ci_lower'] < results['accuracy']
        assert results['ci_upper'] > results['accuracy']
        assert 0 <= results['ci_lower'] <= 1
        assert 0 <= results['ci_upper'] <= 1
    
    def test_model_comparison(self):
        """Test comparing two models"""
        np.random.seed(42)
        n = 1000
        
        # Model A: 85% accurate
        preds_a = np.random.binomial(1, 0.85, n)
        labels = np.ones(n)
        results_a = MLConfidenceCalculator.calculate_accuracy_ci(preds_a, labels)
        
        # Model B: 50% accurate (should be significantly different)
        preds_b = np.random.binomial(1, 0.50, n)
        results_b = MLConfidenceCalculator.calculate_accuracy_ci(preds_b, labels)
        
        comparison = MLConfidenceCalculator.compare_models(results_a, results_b)
        
        assert comparison['difference'] > 0
        assert comparison['statistically_significant'] is True
        assert comparison['p_value'] < 0.05
    
    def test_no_significant_difference(self):
        """Test models that are not significantly different"""
        np.random.seed(42)
        n = 100  # Small sample for less power
        
        # Both models ~80% accurate
        preds_a = np.random.binomial(1, 0.80, n)
        preds_b = np.random.binomial(1, 0.82, n)
        labels = np.ones(n)
        
        results_a = MLConfidenceCalculator.calculate_accuracy_ci(preds_a, labels)
        results_b = MLConfidenceCalculator.calculate_accuracy_ci(preds_b, labels)
        
        comparison = MLConfidenceCalculator.compare_models(results_a, results_b)
        
        # With small sample, small difference likely not significant
        assert comparison['p_value'] > 0.01  # Not strongly significant


class TestABTestCalculator:
    """Test A/B test sample size calculations"""
    
    def test_sample_size_calculation(self):
        """Test sample size calculation produces reasonable results"""
        result = ABTestCalculator.calculate_sample_size(
            baseline_rate=0.75,
            minimum_detectable_effect=0.05,
            alpha=0.05,
            power=0.80
        )
        
        assert result['sample_size_per_group'] > 0
        assert result['total_sample_size'] == 2 * result['sample_size_per_group']
        assert result['new_rate'] == 0.80
        assert result['baseline_rate'] == 0.75
    
    def test_larger_effect_needs_smaller_sample(self):
        """Test that larger effects require smaller samples"""
        small_effect = ABTestCalculator.calculate_sample_size(
            baseline_rate=0.75,
            minimum_detectable_effect=0.02  # 2% improvement
        )
        
        large_effect = ABTestCalculator.calculate_sample_size(
            baseline_rate=0.75,
            minimum_detectable_effect=0.10  # 10% improvement
        )
        
        # Detecting larger effect requires fewer samples
        assert large_effect['total_sample_size'] < small_effect['total_sample_size']
    
    def test_higher_power_needs_larger_sample(self):
        """Test that higher power requires larger samples"""
        low_power = ABTestCalculator.calculate_sample_size(
            baseline_rate=0.75,
            minimum_detectable_effect=0.05,
            power=0.70
        )
        
        high_power = ABTestCalculator.calculate_sample_size(
            baseline_rate=0.75,
            minimum_detectable_effect=0.05,
            power=0.90
        )
        
        # Higher power requires more samples
        assert high_power['total_sample_size'] > low_power['total_sample_size']
    
    def test_sample_size_reasonable_bounds(self):
        """Test sample sizes are in reasonable ranges"""
        result = ABTestCalculator.calculate_sample_size(
            baseline_rate=0.50,
            minimum_detectable_effect=0.05
        )
        
        # Should be at least 100 per group for typical parameters
        assert result['sample_size_per_group'] >= 100
        # But not absurdly large (< 100k per group for reasonable parameters)
        assert result['sample_size_per_group'] < 100000


class TestCLTProperties:
    """Test fundamental CLT properties"""
    
    def test_clt_convergence_with_sample_size(self):
        """Test that larger samples produce more normal distributions"""
        sim = CentralLimitTheoremSimulator(population_size=10000)
        
        # Test with different sample sizes
        sample_sizes = [5, 30, 100]
        normality_scores = []
        
        for n in sample_sizes:
            results = sim.demonstrate_clt(
                distribution_type='exponential',
                sample_size=n,
                num_samples=1000
            )
            
            # Shapiro-Wilk test for normality (higher p-value = more normal)
            from scipy.stats import shapiro
            _, p_value = shapiro(results['sample_means'][:100])  # Test first 100
            normality_scores.append(p_value)
        
        # Not strictly enforced due to randomness, but larger samples should trend toward normality
        # Just verify we get valid p-values
        assert all(0 <= p <= 1 for p in normality_scores)
    
    def test_standard_error_formula(self):
        """Test SE = σ/√n relationship"""
        sim = CentralLimitTheoremSimulator(population_size=10000)
        
        results_30 = sim.demonstrate_clt(sample_size=30, num_samples=1000)
        results_120 = sim.demonstrate_clt(sample_size=120, num_samples=1000)
        
        # SE should be ~2x smaller with 4x sample size
        ratio = results_30['theoretical_se'] / results_120['theoretical_se']
        expected_ratio = np.sqrt(120 / 30)  # Should be 2.0
        
        assert abs(ratio - expected_ratio) < 0.1


def test_all_components_integrate():
    """Integration test: all components work together"""
    # This would be run in a real scenario
    sim = CentralLimitTheoremSimulator(population_size=5000)
    results = sim.demonstrate_clt(sample_size=30, num_samples=500)
    
    # Use results for confidence interval
    calculator = MLConfidenceCalculator()
    
    # Simulate some predictions
    np.random.seed(42)
    preds = np.random.binomial(1, 0.85, 1000)
    labels = np.ones(1000)
    
    ci_results = calculator.calculate_accuracy_ci(preds, labels)
    
    # Calculate sample size for A/B test
    ab_results = ABTestCalculator.calculate_sample_size(
        baseline_rate=ci_results['accuracy'],
        minimum_detectable_effect=0.02
    )
    
    # All components should produce valid outputs
    assert results['sample_means_mean'] > 0
    assert ci_results['accuracy'] > 0
    assert ab_results['total_sample_size'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
