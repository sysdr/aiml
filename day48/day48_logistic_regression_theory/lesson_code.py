"""
Day 48: Logistic Regression Theory - Mathematical Foundations
Exploring the sigmoid function, log-loss, and probability predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import math


class SigmoidFunction:
    """
    Demonstrates the sigmoid (logistic) function and its properties.
    
    The sigmoid function σ(z) = 1 / (1 + e^(-z)) is the heart of logistic regression.
    It maps any real number to a probability between 0 and 1.
    """
    
    @staticmethod
    def compute(z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function: σ(z) = 1 / (1 + e^(-z))
        
        Args:
            z: Input values (can be scalar, array, or matrix)
            
        Returns:
            Sigmoid output in range (0, 1)
        """
        # Clip values to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        """
        Compute derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))
        
        This elegant property makes gradient computation efficient.
        """
        sig = SigmoidFunction.compute(z)
        return sig * (1 - sig)
    
    def visualize_sigmoid(self, save_path: str = 'sigmoid_curve.png'):
        """Visualize the sigmoid function and key properties"""
        z = np.linspace(-10, 10, 1000)
        sigma_z = self.compute(z)
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Sigmoid curve
        plt.subplot(1, 2, 1)
        plt.plot(z, sigma_z, 'b-', linewidth=2, label='σ(z) = 1/(1+e^(-z))')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision boundary')
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Mark key points
        key_points = [(-5, self.compute(-5)), (0, 0.5), (5, self.compute(5))]
        for z_val, sigma_val in key_points:
            plt.plot(z_val, sigma_val, 'ro', markersize=8)
            plt.text(z_val, sigma_val + 0.1, f'({z_val}, {sigma_val:.3f})', 
                    ha='center', fontsize=9)
        
        plt.xlabel('z (linear combination)', fontsize=11)
        plt.ylabel('σ(z) (probability)', fontsize=11)
        plt.title('Sigmoid Function: Linear → Probability', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-0.1, 1.1)
        
        # Plot 2: Sigmoid derivative
        plt.subplot(1, 2, 2)
        derivative = self.derivative(z)
        plt.plot(z, derivative, 'g-', linewidth=2, label="σ'(z) = σ(z)(1-σ(z))")
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        plt.xlabel('z', fontsize=11)
        plt.ylabel("σ'(z)", fontsize=11)
        plt.title('Sigmoid Derivative (Gradient)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sigmoid visualization saved to {save_path}")
        plt.close()


class LogLossCalculator:
    """
    Implements binary cross-entropy (log-loss) for classification.
    
    Log-loss penalizes confident wrong predictions exponentially,
    making it ideal for training classification models.
    """
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
        """
        Compute binary cross-entropy loss.
        
        Loss = -1/m * Σ[y*log(p) + (1-y)*log(1-p)]
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted probabilities (0 to 1)
            epsilon: Small value to prevent log(0)
            
        Returns:
            Average log-loss across all examples
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        m = len(y_true)
        loss = -1/m * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    @staticmethod
    def compute_per_example(y_true: int, y_pred: float, epsilon: float = 1e-15) -> float:
        """Compute log-loss for a single example"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def demonstrate_loss_behavior(self):
        """Show how log-loss penalizes predictions"""
        print("\n" + "="*60)
        print("LOG-LOSS BEHAVIOR DEMONSTRATION")
        print("="*60)
        
        scenarios = [
            (1, 0.99, "Correct & Confident"),
            (1, 0.90, "Correct & Fairly Confident"),
            (1, 0.70, "Correct but Uncertain"),
            (1, 0.51, "Barely Correct"),
            (1, 0.50, "Guessing"),
            (1, 0.10, "Wrong & Confident (BAD)"),
            (1, 0.01, "Wrong & Very Confident (TERRIBLE)"),
        ]
        
        print(f"\n{'True Label':<12} {'Prediction':<12} {'Scenario':<30} {'Loss':<10}")
        print("-" * 70)
        
        for y_true, y_pred, scenario in scenarios:
            loss = self.compute_per_example(y_true, y_pred)
            print(f"{y_true:<12} {y_pred:<12.2f} {scenario:<30} {loss:<10.4f}")
        
        print("\nKey Insight: Confident wrong predictions incur exponential penalties!")
        print("Loss at 0.01 is ~22x worse than loss at 0.90\n")


class DecisionBoundaryAnalyzer:
    """
    Analyzes how different thresholds affect classification decisions.
    
    In production, the 0.5 threshold is often tuned based on business costs.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def predict_class(self, probabilities: np.ndarray) -> np.ndarray:
        """Convert probabilities to class predictions using threshold"""
        return (probabilities >= self.threshold).astype(int)
    
    def analyze_threshold_impact(self, y_true: np.ndarray, probabilities: np.ndarray):
        """Show how different thresholds change predictions"""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        
        print("\n" + "="*60)
        print("THRESHOLD ANALYSIS")
        print("="*60)
        print("\nImpact of different thresholds on classification:")
        print(f"{'Threshold':<12} {'Predicted Positive':<20} {'Accuracy':<12}")
        print("-" * 50)
        
        for thresh in thresholds:
            predictions = (probabilities >= thresh).astype(int)
            num_positive = np.sum(predictions == 1)
            accuracy = np.mean(predictions == y_true)
            
            print(f"{thresh:<12.1f} {num_positive:<20} {accuracy:<12.3f}")
        
        print("\nProduction Example:")
        print("- Spam filter: Low threshold (0.3) → More emails flagged, fewer misses")
        print("- Fraud detection: High threshold (0.8) → Block only when very confident")
        print("- Medical diagnosis: Balance based on treatment cost vs. disease severity\n")


class LogisticRegressionDemo:
    """
    Demonstrates core logistic regression concepts without full training.
    Day 49 will implement the complete training algorithm.
    """
    
    def __init__(self):
        self.sigmoid = SigmoidFunction()
        self.log_loss = LogLossCalculator()
        self.decision_analyzer = DecisionBoundaryAnalyzer()
    
    def demonstrate_linear_to_probability(self):
        """Show how sigmoid converts linear predictions to probabilities"""
        print("\n" + "="*70)
        print("LINEAR PREDICTIONS → PROBABILITY CONVERSION")
        print("="*70)
        
        # Simulated linear predictions (before sigmoid)
        linear_scores = np.array([-5, -2, -0.5, 0, 0.5, 2, 5])
        probabilities = self.sigmoid.compute(linear_scores)
        
        print(f"\n{'Linear Score (z)':<20} {'Probability σ(z)':<20} {'Interpretation':<30}")
        print("-" * 70)
        
        interpretations = [
            "Very confident NEGATIVE",
            "Leaning NEGATIVE",
            "Slightly NEGATIVE",
            "Uncertain (50/50)",
            "Slightly POSITIVE",
            "Leaning POSITIVE",
            "Very confident POSITIVE"
        ]
        
        for z, p, interp in zip(linear_scores, probabilities, interpretations):
            print(f"{z:<20.1f} {p:<20.4f} {interp:<30}")
        
        print("\nReal-World Example (Email Spam Detection):")
        print("- z = -5 → 0.67% spam probability → Deliver to inbox")
        print("- z = 0 → 50% spam probability → Borderline, check other signals")
        print("- z = 5 → 99.3% spam probability → Move to spam folder\n")
    
    def compare_with_linear_regression(self):
        """Illustrate why linear regression fails for classification"""
        print("\n" + "="*70)
        print("WHY LINEAR REGRESSION FAILS FOR CLASSIFICATION")
        print("="*70)
        
        print("\nLinear Regression for Binary Classification:")
        print("Problem: Outputs can be < 0 or > 1 (meaningless as probabilities)")
        print("\nExample predictions:")
        print("- Input x₁=10 → Prediction = 2.5 (250% probability?? Impossible!)")
        print("- Input x₂=-5 → Prediction = -0.8 (negative probability?? Nonsense!)")
        
        print("\n✓ Logistic Regression Solution:")
        print("- Same input x₁=10 → Linear score z=2.5 → σ(z)=0.924 (92.4% probability)")
        print("- Same input x₂=-5 → Linear score z=-0.8 → σ(z)=0.310 (31.0% probability)")
        print("\nSigmoid guarantees valid probabilities between 0 and 1!\n")
    
    def run_complete_demo(self):
        """Execute full demonstration of logistic regression theory"""
        print("\n" + "="*70)
        print("DAY 48: LOGISTIC REGRESSION THEORY - COMPLETE DEMONSTRATION")
        print("="*70)
        
        # Demo 1: Linear to probability
        self.demonstrate_linear_to_probability()
        
        # Demo 2: Why not linear regression
        self.compare_with_linear_regression()
        
        # Demo 3: Log-loss behavior
        self.log_loss.demonstrate_loss_behavior()
        
        # Demo 4: Create sample data for threshold analysis
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        probabilities = np.random.beta(2, 2, n_samples)  # Random probabilities
        
        self.decision_analyzer.analyze_threshold_impact(y_true, probabilities)
        
        # Demo 5: Visualizations
        print("\nGenerating visualizations...")
        self.sigmoid.visualize_sigmoid()
        
        print("\n" + "="*70)
        print("✓ DEMONSTRATION COMPLETE")
        print("="*70)
        print("\nKey Takeaways:")
        print("1. Sigmoid converts unlimited linear scores to valid probabilities (0-1)")
        print("2. Log-loss penalizes confident wrong predictions exponentially")
        print("3. Classification threshold (0.5) should be tuned for your use case")
        print("4. These concepts power production AI at Google, Meta, Netflix, etc.")
        print("\nNext Step: Day 49 - Implement logistic regression from scratch!")


def main():
    """Run the complete Day 48 demonstration"""
    demo = LogisticRegressionDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
