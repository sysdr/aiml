#!/bin/bash

# Day 24: Conditional Probability and Bayes' Theorem - Implementation Generator
# This script creates all necessary files for the lesson

set -e

echo "🚀 Generating Day 24 lesson files..."

# Create setup.sh
cat > setup.sh << 'EOF'
#!/bin/bash

echo "🔧 Setting up Day 24: Conditional Probability and Bayes' Theorem"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete! Activate your environment with: source venv/bin/activate"
echo "📚 Run the lesson with: python lesson_code.py"
echo "🧪 Run tests with: python test_lesson.py"
EOF

chmod +x setup.sh

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy>=1.24.0
matplotlib>=3.7.0
jupyter>=1.0.0
pandas>=2.0.0
scipy>=1.11.0
EOF

# Create lesson_code.py
cat > lesson_code.py << 'EOF'
"""
Day 24: Conditional Probability and Bayes' Theorem
A practical implementation for AI/ML applications
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json


class ConditionalProbability:
    """Understanding conditional probability through practical examples"""
    
    def __init__(self):
        self.examples = []
    
    def calculate_simple(self, p_a_and_b: float, p_b: float) -> float:
        """
        Calculate P(A|B) = P(A and B) / P(B)
        
        Args:
            p_a_and_b: Probability of both A and B occurring
            p_b: Probability of B occurring
            
        Returns:
            Conditional probability P(A|B)
        """
        if p_b == 0:
            raise ValueError("P(B) cannot be zero")
        return p_a_and_b / p_b
    
    def user_behavior_example(self):
        """
        Real-world example: E-commerce user behavior
        Calculate probability of purchase given different user actions
        """
        print("\n=== E-commerce User Behavior Analysis ===\n")
        
        # Sample data (in production, this comes from databases)
        total_users = 10000
        added_to_cart = 2000
        added_and_purchased = 800
        
        # P(Purchase | Added to Cart)
        p_purchase_given_cart = added_and_purchased / added_to_cart
        
        print(f"Total Users: {total_users}")
        print(f"Users who added to cart: {added_to_cart}")
        print(f"Users who added to cart AND purchased: {added_and_purchased}")
        print(f"\nP(Purchase | Added to Cart) = {p_purchase_given_cart:.1%}")
        print(f"\n💡 Insight: {p_purchase_given_cart:.1%} of users who add items to cart complete the purchase")
        print("This metric helps AI systems predict conversion and optimize recommendations")
        
        return p_purchase_given_cart


class BayesianSpamFilter:
    """
    Production-grade spam filter using Naive Bayes
    This is the foundation of email filtering at scale
    """
    
    def __init__(self):
        # Prior probabilities (learned from training data)
        self.p_spam = 0.3  # 30% of emails are spam
        self.p_ham = 0.7   # 70% are legitimate (ham)
        
        # Word likelihoods: P(word|class)
        # In production, these are learned from millions of emails
        self.word_prob_spam = {
            'free': 0.8, 'win': 0.75, 'click': 0.7, 'urgent': 0.85,
            'prize': 0.9, 'congratulations': 0.8, 'offer': 0.65,
            'meeting': 0.05, 'report': 0.03, 'project': 0.04,
            'deadline': 0.1, 'attached': 0.08, 'review': 0.06
        }
        
        self.word_prob_ham = {
            'free': 0.05, 'win': 0.02, 'click': 0.08, 'urgent': 0.15,
            'prize': 0.01, 'congratulations': 0.1, 'offer': 0.1,
            'meeting': 0.7, 'report': 0.75, 'project': 0.8,
            'deadline': 0.6, 'attached': 0.7, 'review': 0.65
        }
        
        self.classification_history = []
    
    def calculate_posterior(self, words: List[str]) -> Dict:
        """
        Apply Bayes' Theorem to classify email
        
        P(Spam|Words) = P(Words|Spam) × P(Spam) / P(Words)
        
        We use the naive assumption: words are independent
        So P(Words|Spam) = P(W1|Spam) × P(W2|Spam) × ...
        """
        # Start with prior probabilities (log space to prevent underflow)
        log_spam_score = np.log(self.p_spam)
        log_ham_score = np.log(self.p_ham)
        
        word_contributions = []
        
        # Update beliefs based on each word (evidence)
        for word in words:
            word_lower = word.lower()
            
            if word_lower in self.word_prob_spam:
                # Get likelihoods
                p_word_given_spam = self.word_prob_spam[word_lower]
                p_word_given_ham = self.word_prob_ham[word_lower]
                
                # Update in log space
                log_spam_score += np.log(p_word_given_spam)
                log_ham_score += np.log(p_word_given_ham)
                
                # Track contribution for explainability
                word_contributions.append({
                    'word': word_lower,
                    'spam_likelihood': p_word_given_spam,
                    'ham_likelihood': p_word_given_ham,
                    'spam_boost': p_word_given_spam / p_word_given_ham
                })
        
        # Convert back from log space and normalize
        spam_score = np.exp(log_spam_score)
        ham_score = np.exp(log_ham_score)
        total = spam_score + ham_score
        
        p_spam_given_words = spam_score / total
        
        result = {
            'is_spam': p_spam_given_words > 0.5,
            'spam_probability': p_spam_given_words,
            'ham_probability': 1 - p_spam_given_words,
            'confidence': max(p_spam_given_words, 1 - p_spam_given_words),
            'prior_spam': self.p_spam,
            'posterior_spam': p_spam_given_words,
            'words_analyzed': len(word_contributions),
            'top_indicators': sorted(word_contributions, 
                                   key=lambda x: abs(np.log(x['spam_boost'])), 
                                   reverse=True)[:3]
        }
        
        self.classification_history.append(result)
        return result
    
    def explain_classification(self, result: Dict):
        """Explain the classification decision (AI explainability)"""
        print(f"\n{'='*60}")
        print(f"📧 EMAIL CLASSIFICATION REPORT")
        print(f"{'='*60}")
        print(f"\n🎯 Decision: {'🚫 SPAM' if result['is_spam'] else '✅ LEGITIMATE'}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"\n📈 Probability Evolution:")
        print(f"   Prior (before analyzing):  P(Spam) = {result['prior_spam']:.1%}")
        print(f"   Posterior (after analyzing): P(Spam) = {result['posterior_spam']:.1%}")
        print(f"   Change: {abs(result['posterior_spam'] - result['prior_spam']):.1%} {'increase' if result['posterior_spam'] > result['prior_spam'] else 'decrease'}")
        
        if result['top_indicators']:
            print(f"\n🔍 Top Spam Indicators:")
            for i, indicator in enumerate(result['top_indicators'], 1):
                direction = "📈 SPAM" if indicator['spam_boost'] > 1 else "📉 HAM"
                print(f"   {i}. '{indicator['word']}' - {direction}")
                print(f"      Spam likelihood: {indicator['spam_likelihood']:.1%}")
                print(f"      Ham likelihood:  {indicator['ham_likelihood']:.1%}")


class BayesianMedicalDiagnosis:
    """
    Medical diagnosis AI using Bayes' Theorem
    Demonstrates the importance of base rates in AI decision-making
    """
    
    def __init__(self, disease_name: str, prevalence: float):
        self.disease_name = disease_name
        self.prevalence = prevalence  # P(Disease) - base rate
        
    def diagnose(self, 
                test_positive: bool,
                sensitivity: float,  # P(Positive|Disease) - true positive rate
                specificity: float   # P(Negative|Healthy) - true negative rate
                ) -> Dict:
        """
        Calculate probability of disease given test result
        
        Bayes' Theorem:
        P(Disease|Positive) = P(Positive|Disease) × P(Disease) / P(Positive)
        """
        
        if test_positive:
            # Calculate P(Positive) using law of total probability
            # P(Positive) = P(Positive|Disease)×P(Disease) + P(Positive|Healthy)×P(Healthy)
            p_positive = (sensitivity * self.prevalence + 
                         (1 - specificity) * (1 - self.prevalence))
            
            # Apply Bayes' Theorem
            p_disease_given_positive = (sensitivity * self.prevalence) / p_positive
            
            return {
                'test_result': 'Positive',
                'disease_probability': p_disease_given_positive,
                'explanation': self._explain_positive_result(
                    p_disease_given_positive, sensitivity, self.prevalence
                )
            }
        else:
            # Calculate P(Negative)
            p_negative = (specificity * (1 - self.prevalence) + 
                         (1 - sensitivity) * self.prevalence)
            
            # P(Disease|Negative)
            p_disease_given_negative = ((1 - sensitivity) * self.prevalence) / p_negative
            
            return {
                'test_result': 'Negative',
                'disease_probability': p_disease_given_negative,
                'explanation': self._explain_negative_result(
                    p_disease_given_negative, specificity
                )
            }
    
    def _explain_positive_result(self, prob: float, sensitivity: float, prevalence: float):
        """Generate human-readable explanation"""
        return f"""
🏥 Medical Diagnosis Analysis:
   Disease: {self.disease_name}
   Test Result: POSITIVE
   
   Even with {sensitivity:.1%} test accuracy (sensitivity),
   the actual probability of having {self.disease_name} is only {prob:.1%}
   
   Why? The disease is rare (base rate: {prevalence:.1%})
   
   🧠 AI Insight: This shows why AI systems must consider base rates.
   High test accuracy ≠ high probability of disease when disease is rare.
   
   This is the "base rate fallacy" - one of the most important concepts
   in AI decision-making and risk assessment.
        """
    
    def _explain_negative_result(self, prob: float, specificity: float):
        """Generate human-readable explanation for negative result"""
        return f"""
🏥 Medical Diagnosis Analysis:
   Disease: {self.disease_name}
   Test Result: NEGATIVE
   
   Probability of having {self.disease_name}: {prob:.1%}
   Test specificity (true negative rate): {specificity:.1%}
   
   ✅ Good news: Very low probability of disease given negative test.
        """
    
    def compare_test_qualities(self):
        """Show how test accuracy affects diagnosis"""
        print(f"\n{'='*60}")
        print(f"Impact of Test Quality on Diagnosis")
        print(f"Disease: {self.disease_name} (Prevalence: {self.prevalence:.1%})")
        print(f"{'='*60}\n")
        
        test_scenarios = [
            ("Basic Test", 0.80, 0.90),
            ("Good Test", 0.95, 0.95),
            ("Excellent Test", 0.99, 0.99)
        ]
        
        for name, sensitivity, specificity in test_scenarios:
            result = self.diagnose(test_positive=True, 
                                 sensitivity=sensitivity, 
                                 specificity=specificity)
            print(f"{name}:")
            print(f"  Sensitivity: {sensitivity:.1%}, Specificity: {specificity:.1%}")
            print(f"  P(Disease|Positive) = {result['disease_probability']:.1%}\n")


class BayesianUpdateVisualizer:
    """Visualize how beliefs update with evidence"""
    
    def visualize_belief_update(self, 
                               prior: float,
                               likelihoods: List[Tuple[str, float, float]],
                               save_path: str = 'bayesian_update.png'):
        """
        Show how probability updates with each piece of evidence
        
        Args:
            prior: Initial belief P(A)
            likelihoods: List of (evidence_name, P(E|A), P(E|not A))
        """
        beliefs = [prior]
        evidence_names = ['Prior']
        
        current_p_a = prior
        
        for name, p_e_given_a, p_e_given_not_a in likelihoods:
            # Apply Bayes' Theorem
            p_e = p_e_given_a * current_p_a + p_e_given_not_a * (1 - current_p_a)
            current_p_a = (p_e_given_a * current_p_a) / p_e
            
            beliefs.append(current_p_a)
            evidence_names.append(name)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot belief evolution
        plt.subplot(1, 2, 1)
        plt.plot(range(len(beliefs)), beliefs, marker='o', linewidth=2, markersize=10)
        plt.xticks(range(len(beliefs)), evidence_names, rotation=45, ha='right')
        plt.ylabel('Probability', fontsize=12)
        plt.xlabel('Evidence', fontsize=12)
        plt.title('Bayesian Belief Update Process', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision Threshold')
        plt.legend()
        
        # Plot belief changes
        plt.subplot(1, 2, 2)
        changes = [0] + [beliefs[i] - beliefs[i-1] for i in range(1, len(beliefs))]
        colors = ['green' if c >= 0 else 'red' for c in changes]
        plt.bar(range(len(changes)), changes, color=colors, alpha=0.7)
        plt.xticks(range(len(changes)), evidence_names, rotation=45, ha='right')
        plt.ylabel('Change in Probability', fontsize=12)
        plt.xlabel('Evidence', fontsize=12)
        plt.title('Impact of Each Evidence', fontsize=14, fontweight='bold')
        plt.axhline(y=0, color='black', linewidth=0.5)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {save_path}")
        plt.close()


def demonstrate_conditional_probability():
    """Interactive demonstration of conditional probability"""
    print("\n" + "="*70)
    print("PART 1: CONDITIONAL PROBABILITY BASICS")
    print("="*70)
    
    cp = ConditionalProbability()
    cp.user_behavior_example()


def demonstrate_spam_filter():
    """Interactive spam filter demonstration"""
    print("\n\n" + "="*70)
    print("PART 2: BAYESIAN SPAM FILTER (PRODUCTION AI EXAMPLE)")
    print("="*70)
    
    filter = BayesianSpamFilter()
    
    # Test emails
    test_emails = [
        {
            'subject': 'Urgent: Free Prize - Click to Win!',
            'words': ['urgent', 'free', 'prize', 'click', 'win']
        },
        {
            'subject': 'Project Report - Review Deadline',
            'words': ['project', 'report', 'review', 'deadline']
        },
        {
            'subject': 'Meeting Tomorrow - Report Attached',
            'words': ['meeting', 'report', 'attached']
        },
        {
            'subject': 'Congratulations! Free Offer Inside',
            'words': ['congratulations', 'free', 'offer']
        }
    ]
    
    for i, email in enumerate(test_emails, 1):
        print(f"\n{'─'*60}")
        print(f"Email #{i}: {email['subject']}")
        print(f"{'─'*60}")
        result = filter.calculate_posterior(email['words'])
        filter.explain_classification(result)


def demonstrate_medical_diagnosis():
    """Demonstrate medical diagnosis with Bayes' Theorem"""
    print("\n\n" + "="*70)
    print("PART 3: MEDICAL AI DIAGNOSIS (BASE RATE IMPORTANCE)")
    print("="*70)
    
    # Rare disease example
    diagnosis = BayesianMedicalDiagnosis(
        disease_name="Rare Genetic Disorder",
        prevalence=0.001  # 0.1% of population
    )
    
    print("\n📋 Scenario: Testing for a rare disease")
    result = diagnosis.diagnose(
        test_positive=True,
        sensitivity=0.99,  # 99% true positive rate
        specificity=0.95   # 95% true negative rate
    )
    
    print(result['explanation'])
    
    # Compare test qualities
    diagnosis.compare_test_qualities()


def demonstrate_belief_updates():
    """Show how AI updates beliefs with new evidence"""
    print("\n\n" + "="*70)
    print("PART 4: VISUALIZING BAYESIAN UPDATES")
    print("="*70)
    
    visualizer = BayesianUpdateVisualizer()
    
    print("\n📈 Scenario: AI deciding if a transaction is fraudulent")
    print("Starting belief: 5% chance of fraud (typical base rate)")
    
    # Evidence comes in sequentially
    evidence = [
        ("Unusual location", 0.7, 0.1),      # P(Evidence|Fraud), P(Evidence|Normal)
        ("High amount", 0.8, 0.15),
        ("New merchant", 0.6, 0.3),
        ("Late night", 0.7, 0.2)
    ]
    
    visualizer.visualize_belief_update(
        prior=0.05,
        likelihoods=evidence,
        save_path='fraud_detection_belief_update.png'
    )


def main():
    """Run all demonstrations"""
    print("\n" + "🎯"*35)
    print("DAY 24: CONDITIONAL PROBABILITY AND BAYES' THEOREM")
    print("Real-World AI Applications")
    print("🎯"*35)
    
    demonstrate_conditional_probability()
    demonstrate_spam_filter()
    demonstrate_medical_diagnosis()
    demonstrate_belief_updates()
    
    print("\n\n" + "="*70)
    print("✅ LESSON COMPLETE")
    print("="*70)
    print("""
🎓 Key Takeaways:
   1. Conditional probability is the foundation of AI decision-making
   2. Bayes' Theorem lets us update beliefs with new evidence
   3. Base rates matter enormously in real-world AI systems
   4. Production AI systems continuously apply Bayesian updates
   
🚀 Next Steps:
   - Experiment with different prior probabilities
   - Try building your own spam filter with custom words
   - Think about other applications: recommendation systems, 
     fraud detection, medical diagnosis, autonomous vehicles
   
💡 Remember: Every time an AI makes a decision, it's using some form
   of Bayesian reasoning to balance prior knowledge with new evidence.
    """)


if __name__ == "__main__":
    main()
EOF

# Create test_lesson.py
cat > test_lesson.py << 'EOF'
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
EOF

# Create README.md
cat > README.md << 'EOF'
# Day 24: Conditional Probability and Bayes' Theorem

## 🎯 Learning Objectives

By the end of this lesson, you will:
- Understand conditional probability and its role in AI decision-making
- Apply Bayes' Theorem to real-world AI problems
- Build a production-style spam filter using Naive Bayes
- Understand why base rates matter in AI systems
- See how Bayesian reasoning powers modern AI applications

## 🚀 Quick Start

### 1. Setup Environment

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run the Lesson

```bash
python lesson_code.py
```

This will demonstrate:
- Conditional probability in e-commerce analytics
- Bayesian spam filtering (like Gmail)
- Medical diagnosis AI with base rate consideration
- Visual belief update process

### 3. Test Your Understanding

```bash
python test_lesson.py
```

All tests should pass if you've understood the concepts correctly.

## 📚 What You'll Build

### 1. **Bayesian Spam Filter**
Real implementation of email classification using Naive Bayes:
```python
filter = BayesianSpamFilter()
result = filter.classify(['urgent', 'free', 'prize'])
# Classifies email as spam with probability score
```

### 2. **Medical Diagnosis AI**
Shows the critical importance of base rates:
```python
diagnosis = BayesianMedicalDiagnosis(disease="Rare Genetic Disorder", prevalence=0.001)
result = diagnosis.diagnose(test_positive=True, sensitivity=0.99, specificity=0.95)
# Shows why even 99% accurate test doesn't mean 99% probability
```

### 3. **Belief Update Visualizer**
See how AI updates beliefs as evidence arrives:
```python
visualizer = BayesianUpdateVisualizer()
visualizer.visualize_belief_update(prior=0.05, likelihoods=[...])
# Creates visualization of Bayesian inference process
```

## 🔑 Key Concepts

### Conditional Probability
```
P(A|B) = P(A and B) / P(B)
```
"Probability of A, given that B happened"

### Bayes' Theorem
```
P(A|B) = P(B|A) × P(A) / P(B)
```
"Update your belief in A based on evidence B"

### Naive Bayes (for spam filtering)
Assumes independence between features (words):
```
P(Spam|Words) = P(Words|Spam) × P(Spam) / P(Words)
P(Words|Spam) ≈ P(W1|Spam) × P(W2|Spam) × ...
```

## 💡 Real-World Applications

1. **Email Filtering**: Gmail's spam filter uses Naive Bayes
2. **Medical Diagnosis**: AI health assistants use Bayesian reasoning
3. **Fraud Detection**: Credit card systems update fraud probability with each transaction detail
4. **Recommendation Systems**: Netflix/YouTube update preferences as you watch
5. **Autonomous Vehicles**: Sensor fusion uses Bayesian updates for object detection
6. **Natural Language Processing**: Language models are essentially Bayesian inference engines

## 🧪 Experiments to Try

1. **Modify spam filter**: Add your own words and probabilities
2. **Change base rates**: See how disease prevalence affects diagnosis
3. **Test accuracy trade-offs**: Compare different test sensitivity/specificity
4. **Sequential updates**: Add more evidence and watch beliefs evolve

## 📊 Expected Outputs

- `fraud_detection_belief_update.png`: Visualization of belief updates
- Classification reports for test emails
- Medical diagnosis explanations
- Test results showing your understanding

## 🎓 Connection to AI/ML

This lesson is foundational because:

1. **Machine Learning is Bayesian**: Most ML algorithms implicitly or explicitly use Bayesian reasoning
2. **Neural Networks**: Dropout, batch normalization, Bayesian neural networks
3. **Reinforcement Learning**: Policy updates are Bayesian
4. **Natural Language Processing**: Language models update probability distributions
5. **Computer Vision**: Object detection confidence scores are Bayesian posteriors

## 🔗 Next Steps

Tomorrow (Day 25): **Random Variables and Probability Distributions**
- Learn how to model uncertainty systematically
- Understand Normal, Binomial, Poisson distributions
- See how these appear in neural networks and AI systems

## 📖 Additional Resources

- Read the lesson article: `lesson_article.md`
- Experiment with the code
- Try building a spam filter for your own email
- Think about other applications of Bayes' Theorem in your daily life

## ⚠️ Common Pitfalls

1. **Base Rate Fallacy**: Ignoring prior probability (prevalence)
2. **Independence Assumption**: Naive Bayes assumes features are independent (rarely true, but works anyway!)
3. **Numerical Underflow**: Use log probabilities for many multiplications
4. **Interpretation**: P(Disease|Positive Test) ≠ Test Accuracy

## 🎯 Success Criteria

You've mastered this lesson when you can:
- ✅ Explain conditional probability to a friend
- ✅ Apply Bayes' Theorem to a new problem
- ✅ Understand why base rates matter
- ✅ See Bayesian reasoning in everyday AI applications
- ✅ Pass all tests

---

**Remember**: Every sophisticated AI system uses Bayesian reasoning to transform prior knowledge and new evidence into intelligent decisions. Master this, and you understand the core logic behind modern AI.

🚀 Happy Learning!
EOF

echo "✅ setup.sh"
echo "✅ requirements.txt"
echo "✅ lesson_code.py"
echo "✅ test_lesson.py"
echo "✅ README.md"
echo ""
echo "🎉 All files generated successfully!"

