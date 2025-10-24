"""
Day 23: Introduction to Probability
Building a Simple Spam Classifier with Probability
"""

import random
from collections import Counter
from typing import Dict, List
import numpy as np


class ProbabilityBasics:
    """Learn probability fundamentals through practical examples"""
    
    @staticmethod
    def calculate_simple_probability(favorable_outcomes: int, total_outcomes: int) -> float:
        """
        Calculate basic probability: P(Event) = favorable / total
        
        Example: Rolling a 4 on a six-sided die
        P(4) = 1 favorable outcome / 6 total outcomes = 0.167
        """
        if total_outcomes == 0:
            return 0.0
        return favorable_outcomes / total_outcomes
    
    @staticmethod
    def simulate_coin_flips(num_flips: int = 1000) -> Dict[str, float]:
        """
        Simulate coin flips to demonstrate probability convergence
        As we flip more, observed probability approaches theoretical (0.5)
        """
        flips = [random.choice(['Heads', 'Tails']) for _ in range(num_flips)]
        heads_count = flips.count('Heads')
        
        return {
            'heads_probability': heads_count / num_flips,
            'tails_probability': 1 - (heads_count / num_flips),
            'total_flips': num_flips,
            'theoretical_probability': 0.5
        }
    
    @staticmethod
    def calculate_joint_probability(event_a_prob: float, event_b_prob: float, 
                                   independent: bool = True) -> float:
        """
        Calculate P(A and B)
        For independent events: P(A and B) = P(A) × P(B)
        """
        if independent:
            return event_a_prob * event_b_prob
        else:
            raise NotImplementedError("Dependent events covered in Day 24!")


class SpamClassifier:
    """
    Simple probability-based spam classifier
    Demonstrates how AI uses probability to make decisions under uncertainty
    """
    
    def __init__(self):
        self.spam_word_probs: Dict[str, float] = {}
        self.ham_word_probs: Dict[str, float] = {}
        self.p_spam: float = 0.5
        self.p_ham: float = 0.5
        
    def train(self, spam_emails: List[str], ham_emails: List[str]):
        """
        Learn probability distributions from training data
        This is the 'learning' part of machine learning!
        """
        # Calculate P(spam) and P(ham) - prior probabilities
        total_emails = len(spam_emails) + len(ham_emails)
        self.p_spam = len(spam_emails) / total_emails
        self.p_ham = len(ham_emails) / total_emails
        
        # Calculate P(word | spam) for each word
        self.spam_word_probs = self._calculate_word_probabilities(spam_emails)
        self.ham_word_probs = self._calculate_word_probabilities(ham_emails)
        
        print(f"📚 Training complete!")
        print(f"   P(spam) = {self.p_spam:.2f}")
        print(f"   P(ham) = {self.p_ham:.2f}")
        print(f"   Learned {len(self.spam_word_probs)} spam words")
        print(f"   Learned {len(self.ham_word_probs)} ham words")
    
    def _calculate_word_probabilities(self, emails: List[str]) -> Dict[str, float]:
        """Calculate P(word | email_type)"""
        all_words = []
        for email in emails:
            all_words.extend(email.lower().split())
        
        word_counts = Counter(all_words)
        total_words = len(all_words)
        
        if total_words == 0:
            return {}
        
        # Convert counts to probabilities
        return {word: count / total_words 
                for word, count in word_counts.items()}
    
    def classify(self, email_text: str) -> Dict:
        """
        Classify email using probability
        Returns probabilities for both classes, not just a binary decision
        """
        words = email_text.lower().split()
        
        # Start with prior probabilities
        spam_score = self.p_spam
        ham_score = self.p_ham
        
        # Multiply by word probabilities (simplified Naive Bayes)
        # Using small smoothing value for unseen words
        for word in words:
            spam_score *= self.spam_word_probs.get(word, 0.01)
            ham_score *= self.ham_word_probs.get(word, 0.01)
        
        # Normalize to get proper probabilities that sum to 1
        total_score = spam_score + ham_score
        
        if total_score == 0:
            spam_prob = 0.5
            ham_prob = 0.5
        else:
            spam_prob = spam_score / total_score
            ham_prob = ham_score / total_score
        
        return {
            'text': email_text,
            'spam_probability': spam_prob,
            'ham_probability': ham_prob,
            'classification': 'SPAM' if spam_prob > ham_prob else 'HAM',
            'confidence': max(spam_prob, ham_prob)
        }


class ProbabilityDistribution:
    """Visualize probability distributions - how AI thinks about uncertainty"""
    
    @staticmethod
    def create_dice_distribution() -> Dict[int, float]:
        """
        Probability mass function for a fair six-sided die
        Each outcome has equal probability (uniform distribution)
        """
        return {i: 1/6 for i in range(1, 7)}
    
    @staticmethod
    def simulate_distribution(num_samples: int = 10000) -> Dict[int, float]:
        """
        Simulate rolling a die many times
        Observed frequencies approach theoretical probabilities
        """
        rolls = [random.randint(1, 6) for _ in range(num_samples)]
        counts = Counter(rolls)
        
        return {outcome: count / num_samples 
                for outcome, count in sorted(counts.items())}


def main():
    """Run the complete Day 23 lesson"""
    
    print("=" * 60)
    print("Day 23: Introduction to Probability for AI Systems")
    print("=" * 60)
    print()
    
    # Section 1: Basic Probability
    print("📊 Section 1: Basic Probability Calculations")
    print("-" * 60)
    
    prob = ProbabilityBasics()
    
    # Die roll example
    p_rolling_four = prob.calculate_simple_probability(1, 6)
    print(f"P(rolling a 4 on a die) = {p_rolling_four:.4f}")
    
    # Coin flip simulation
    print("\n🪙 Simulating coin flips...")
    flip_results = prob.simulate_coin_flips(10000)
    print(f"After {flip_results['total_flips']} flips:")
    print(f"  Observed P(Heads) = {flip_results['heads_probability']:.4f}")
    print(f"  Theoretical P(Heads) = {flip_results['theoretical_probability']:.4f}")
    
    # Joint probability
    p_two_heads = prob.calculate_joint_probability(0.5, 0.5, independent=True)
    print(f"\nP(two heads in a row) = {p_two_heads:.4f}")
    
    print()
    
    # Section 2: Spam Classifier
    print("📧 Section 2: Building a Spam Classifier")
    print("-" * 60)
    
    # Training data
    spam_emails = [
        "win free money now click here",
        "free prize winner claim now",
        "congratulations you won money",
        "claim your free prize today",
        "winner winner free money"
    ]
    
    ham_emails = [
        "meeting scheduled for tomorrow at three",
        "project deadline is next week",
        "lunch plans for today",
        "can we schedule a call",
        "review the project proposal"
    ]
    
    # Create and train classifier
    classifier = SpamClassifier()
    classifier.train(spam_emails, ham_emails)
    
    print()
    
    # Test the classifier
    test_emails = [
        "win free prize money",
        "meeting about the project",
        "free lunch today",
        "claim your prize"
    ]
    
    print("🧪 Testing classifier on new emails:")
    print()
    
    for email in test_emails:
        result = classifier.classify(email)
        print(f"Email: '{result['text']}'")
        print(f"  → {result['classification']} "
              f"(confidence: {result['confidence']:.1%})")
        print(f"  → P(spam) = {result['spam_probability']:.3f}, "
              f"P(ham) = {result['ham_probability']:.3f}")
        print()
    
    # Section 3: Probability Distribution
    print("📈 Section 3: Probability Distributions")
    print("-" * 60)
    
    dist = ProbabilityDistribution()
    
    theoretical = dist.create_dice_distribution()
    print("Theoretical die roll distribution (uniform):")
    for outcome, prob in theoretical.items():
        print(f"  P({outcome}) = {prob:.4f}")
    
    print("\nSimulated die roll distribution (10,000 rolls):")
    observed = dist.simulate_distribution(10000)
    for outcome, prob in observed.items():
        print(f"  P({outcome}) = {prob:.4f}")
    
    print()
    print("=" * 60)
    print("✅ Day 23 Complete!")
    print("=" * 60)
    print("\n💡 Key Takeaway:")
    print("   Probability lets AI systems quantify uncertainty.")
    print("   Every AI prediction is fundamentally a probability calculation.")
    print("\n📚 Tomorrow: Conditional Probability and Bayes' Theorem")
    print("   Learn how AI updates beliefs with new evidence!")


if __name__ == "__main__":
    main()


