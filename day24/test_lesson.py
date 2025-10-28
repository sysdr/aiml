"""
Tests for Day 24: Conditional Probability and Bayes' Theorem
Verify your understanding with these automated tests
"""

import unittest
import numpy as np
from lesson_code import (
    ConditionalProbability,
    BayesianSpamFilter,
    BayesianMedicalDiagnosis
)


class TestConditionalProbability(unittest.TestCase):
    """Test basic conditional probability calculations"""
    
    def setUp(self):
        self.cp = ConditionalProbability()
    
    def test_simple_conditional(self):
        """Test P(A|B) = P(A and B) / P(B)"""
        # P(Rain and Cold) = 0.1, P(Cold) = 0.3
        result = self.cp.calculate_simple(p_a_and_b=0.1, p_b=0.3)
        self.assertAlmostEqual(result, 0.333, places=2)
    
    def test_zero_denominator(self):
        """Should raise error when P(B) = 0"""
        with self.assertRaises(ValueError):
            self.cp.calculate_simple(p_a_and_b=0.1, p_b=0.0)
    
    def test_perfect_correlation(self):
        """When A and B always happen together"""
        result = self.cp.calculate_simple(p_a_and_b=0.5, p_b=0.5)
        self.assertAlmostEqual(result, 1.0)


class TestBayesianSpamFilter(unittest.TestCase):
    """Test spam filter implementation"""
    
    def setUp(self):
        self.filter = BayesianSpamFilter()
    
    def test_obvious_spam(self):
        """Strongly spammy words should classify as spam"""
        words = ['free', 'win', 'prize', 'urgent']
        result = self.filter.calculate_posterior(words)
        
        self.assertTrue(result['is_spam'])
        self.assertGreater(result['spam_probability'], 0.9)
    
    def test_obvious_ham(self):
        """Strongly legitimate words should classify as ham"""
        words = ['meeting', 'report', 'project', 'deadline']
        result = self.filter.calculate_posterior(words)
        
        self.assertFalse(result['is_spam'])
        self.assertLess(result['spam_probability'], 0.1)
    
    def test_mixed_signals(self):
        """Mixed words should have moderate probability"""
        words = ['meeting', 'free']
        result = self.filter.calculate_posterior(words)
        
        # Should be somewhere in the middle
        self.assertGreater(result['spam_probability'], 0.2)
        self.assertLess(result['spam_probability'], 0.8)
    
    def test_empty_words(self):
        """Empty word list should return prior probability"""
        words = []
        result = self.filter.calculate_posterior(words)
        
        # Should be close to prior (0.3)
        self.assertAlmostEqual(result['spam_probability'], 0.3, places=1)
    
    def test_probability_sum(self):
        """Spam and ham probabilities should sum to 1"""
        words = ['click', 'offer']
        result = self.filter.calculate_posterior(words)
        
        prob_sum = result['spam_probability'] + result['ham_probability']
        self.assertAlmostEqual(prob_sum, 1.0, places=6)


class TestMedicalDiagnosis(unittest.TestCase):
    """Test medical diagnosis with Bayes' Theorem"""
    
    def test_rare_disease_positive_test(self):
        """Even with accurate test, rare disease = low probability"""
        diagnosis = BayesianMedicalDiagnosis(
            disease_name="Rare Disease",
            prevalence=0.001  # 0.1%
        )
        
        result = diagnosis.diagnose(
            test_positive=True,
            sensitivity=0.99,  # 99% accurate
            specificity=0.95   # 95% accurate
        )
        
        # Despite 99% accurate test, probability should be low due to rarity
        self.assertLess(result['disease_probability'], 0.2)
        self.assertGreater(result['disease_probability'], 0.01)
    
    def test_common_disease_positive_test(self):
        """Common disease with positive test = high probability"""
        diagnosis = BayesianMedicalDiagnosis(
            disease_name="Common Disease",
            prevalence=0.1  # 10%
        )
        
        result = diagnosis.diagnose(
            test_positive=True,
            sensitivity=0.95,
            specificity=0.95
        )
        
        # Should have high probability
        self.assertGreater(result['disease_probability'], 0.6)
    
    def test_negative_test_low_probability(self):
        """Negative test should give very low probability"""
        diagnosis = BayesianMedicalDiagnosis(
            disease_name="Any Disease",
            prevalence=0.05
        )
        
        result = diagnosis.diagnose(
            test_positive=False,
            sensitivity=0.95,
            specificity=0.98
        )
        
        # Negative test should mean very low probability
        self.assertLess(result['disease_probability'], 0.05)
    
    def test_perfect_test_rare_disease(self):
        """Perfect test on rare disease"""
        diagnosis = BayesianMedicalDiagnosis(
            disease_name="Rare Disease",
            prevalence=0.001
        )
        
        result = diagnosis.diagnose(
            test_positive=True,
            sensitivity=1.0,  # Perfect
            specificity=1.0   # Perfect
        )
        
        # Even perfect test, but low base rate
        # P(Disease|Positive) should be higher but still limited by prevalence
        self.assertGreater(result['disease_probability'], 0.1)


class TestBayesianLogic(unittest.TestCase):
    """Test core Bayesian reasoning"""
    
    def test_bayes_theorem_basic(self):
        """Verify Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)"""
        # Example: Disease diagnosis
        p_disease = 0.01  # 1% prevalence
        p_positive_given_disease = 0.99  # 99% sensitivity
        p_positive_given_healthy = 0.05  # 5% false positive
        
        # Calculate P(Positive)
        p_positive = (p_positive_given_disease * p_disease + 
                     p_positive_given_healthy * (1 - p_disease))
        
        # Apply Bayes' Theorem
        p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
        
        # Verify it's a valid probability
        self.assertGreaterEqual(p_disease_given_positive, 0)
        self.assertLessEqual(p_disease_given_positive, 1)
        
        # Should be around 16% (counterintuitive but correct!)
        self.assertAlmostEqual(p_disease_given_positive, 0.166, places=2)
    
    def test_prior_impact(self):
        """Show that prior probability significantly impacts posterior"""
        # Same test, different priors
        sensitivity = 0.9
        specificity = 0.9
        
        def calculate_posterior(prior):
            p_positive = sensitivity * prior + (1 - specificity) * (1 - prior)
            return (sensitivity * prior) / p_positive
        
        low_prior = calculate_posterior(0.01)
        high_prior = calculate_posterior(0.5)
        
        # Higher prior should give higher posterior
        self.assertLess(low_prior, high_prior)
        
        # The difference should be substantial
        self.assertGreater(high_prior - low_prior, 0.3)


def run_tests_with_feedback():
    """Run tests with detailed feedback"""
    print("\n" + "="*70)
    print("🧪 RUNNING TESTS FOR DAY 24")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestConditionalProbability))
    suite.addTests(loader.loadTestsFromTestCase(TestBayesianSpamFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestMedicalDiagnosis))
    suite.addTests(loader.loadTestsFromTestCase(TestBayesianLogic))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! You understand Bayes' Theorem!")
        print("\n💡 Key Concepts Verified:")
        print("   ✓ Conditional probability calculations")
        print("   ✓ Bayesian spam filtering")
        print("   ✓ Medical diagnosis with base rates")
        print("   ✓ Core Bayesian reasoning")
    else:
        print("\n📚 Some tests failed. Review the concepts and try again!")
    
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests_with_feedback()
    exit(0 if success else 1)
