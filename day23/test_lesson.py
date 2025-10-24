"""
Tests for Day 23: Introduction to Probability
Run with: python test_lesson.py
"""

import unittest
from lesson_code import ProbabilityBasics, SpamClassifier, ProbabilityDistribution


class TestProbabilityBasics(unittest.TestCase):
    """Test basic probability calculations"""
    
    def setUp(self):
        self.prob = ProbabilityBasics()
    
    def test_simple_probability(self):
        """Test basic probability calculation"""
        # Rolling a 4 on a die
        result = self.prob.calculate_simple_probability(1, 6)
        self.assertAlmostEqual(result, 1/6, places=4)
        
        # Coin flip
        result = self.prob.calculate_simple_probability(1, 2)
        self.assertEqual(result, 0.5)
    
    def test_zero_outcomes(self):
        """Test edge case with zero total outcomes"""
        result = self.prob.calculate_simple_probability(0, 0)
        self.assertEqual(result, 0.0)
    
    def test_joint_probability(self):
        """Test joint probability for independent events"""
        # Two coin flips
        result = self.prob.calculate_joint_probability(0.5, 0.5)
        self.assertEqual(result, 0.25)
        
        # Two die rolls
        result = self.prob.calculate_joint_probability(1/6, 1/6)
        self.assertAlmostEqual(result, 1/36, places=4)


class TestSpamClassifier(unittest.TestCase):
    """Test spam classifier functionality"""
    
    def setUp(self):
        self.classifier = SpamClassifier()
        
        # Simple training data
        self.spam = [
            "free money win",
            "win prize free"
        ]
        self.ham = [
            "meeting tomorrow project",
            "project deadline meeting"
        ]
        
        self.classifier.train(self.spam, self.ham)
    
    def test_training(self):
        """Test that training sets correct probabilities"""
        # Prior probabilities should be 0.5 each (2 spam, 2 ham)
        self.assertEqual(self.classifier.p_spam, 0.5)
        self.assertEqual(self.classifier.p_ham, 0.5)
        
        # Should have learned some words
        self.assertGreater(len(self.classifier.spam_word_probs), 0)
        self.assertGreater(len(self.classifier.ham_word_probs), 0)
    
    def test_classification_structure(self):
        """Test that classification returns correct structure"""
        result = self.classifier.classify("free win prize")
        
        # Check all required fields exist
        self.assertIn('text', result)
        self.assertIn('spam_probability', result)
        self.assertIn('ham_probability', result)
        self.assertIn('classification', result)
        self.assertIn('confidence', result)
    
    def test_probability_sum(self):
        """Test that probabilities sum to 1"""
        result = self.classifier.classify("meeting project")
        
        prob_sum = result['spam_probability'] + result['ham_probability']
        self.assertAlmostEqual(prob_sum, 1.0, places=5)
    
    def test_spam_detection(self):
        """Test that obvious spam is classified correctly"""
        result = self.classifier.classify("free money win prize")
        self.assertEqual(result['classification'], 'SPAM')
    
    def test_ham_detection(self):
        """Test that obvious ham is classified correctly"""
        result = self.classifier.classify("meeting project deadline")
        self.assertEqual(result['classification'], 'HAM')


class TestProbabilityDistribution(unittest.TestCase):
    """Test probability distribution functions"""
    
    def setUp(self):
        self.dist = ProbabilityDistribution()
    
    def test_dice_distribution(self):
        """Test uniform distribution for fair die"""
        dist = self.dist.create_dice_distribution()
        
        # Should have 6 outcomes
        self.assertEqual(len(dist), 6)
        
        # Each should have probability 1/6
        for outcome in range(1, 7):
            self.assertAlmostEqual(dist[outcome], 1/6, places=4)
        
        # Total probability should sum to 1
        total_prob = sum(dist.values())
        self.assertAlmostEqual(total_prob, 1.0, places=4)
    
    def test_simulated_distribution(self):
        """Test that simulation produces valid distribution"""
        dist = self.dist.simulate_distribution(1000)
        
        # Should have outcomes 1-6
        self.assertEqual(len(dist), 6)
        
        # All probabilities should be positive
        for prob in dist.values():
            self.assertGreater(prob, 0)
        
        # Total should sum to approximately 1
        total_prob = sum(dist.values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)


def run_tests():
    """Run all tests and display results"""
    print("=" * 60)
    print("Running Day 23 Tests")
    print("=" * 60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestProbabilityBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestSpamClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestProbabilityDistribution))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("❌ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(run_tests())


