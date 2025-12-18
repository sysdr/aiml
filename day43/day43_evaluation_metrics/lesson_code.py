"""
Day 43: Model Evaluation Metrics - Comprehensive Implementation
Building production-grade evaluation tools from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
from tabulate import tabulate


class MetricsCalculator:
    """
    Production-grade metrics calculator similar to what's used at
    Google, Meta, and Netflix for model evaluation.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Initialize with ground truth and predictions.
        
        Args:
            y_true: Actual labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        
        # Calculate confusion matrix components
        self.tp = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.tn = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.fp = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.fn = np.sum((self.y_true == 1) & (self.y_pred == 0))
        
    def accuracy(self) -> float:
        """
        Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
        
        Returns overall correctness but misleading with imbalanced data.
        """
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total
    
    def precision(self) -> float:
        """
        Calculate precision: TP / (TP + FP)
        
        "When I predict positive, how often am I right?"
        Critical for: spam filters, search results, targeted ads
        """
        denominator = self.tp + self.fp
        if denominator == 0:
            return 0.0
        return self.tp / denominator
    
    def recall(self) -> float:
        """
        Calculate recall: TP / (TP + FN)
        
        "Of all actual positives, how many did I catch?"
        Critical for: disease detection, fraud detection, safety systems
        """
        denominator = self.tp + self.fn
        if denominator == 0:
            return 0.0
        return self.tp / denominator
    
    def f1_score(self) -> float:
        """
        Calculate F1 Score: harmonic mean of precision and recall
        
        F1 = 2 * (precision * recall) / (precision + recall)
        
        Useful when you need to balance precision and recall.
        """
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def confusion_matrix(self) -> np.ndarray:
        """Return confusion matrix as 2x2 numpy array"""
        return np.array([
            [self.tn, self.fp],
            [self.fn, self.tp]
        ])
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Return all metrics in a dictionary"""
        return {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(),
            'Recall': self.recall(),
            'F1 Score': self.f1_score(),
            'True Positives': int(self.tp),
            'True Negatives': int(self.tn),
            'False Positives': int(self.fp),
            'False Negatives': int(self.fn)
        }
    
    def print_report(self, scenario_name: str = "Model Evaluation"):
        """
        Print a comprehensive evaluation report.
        
        Similar to what data scientists see in production dashboards.
        """
        print(f"\n{'='*60}")
        print(f"{scenario_name.center(60)}")
        print(f"{'='*60}\n")
        
        # Confusion Matrix
        print("Confusion Matrix:")
        print(f"                 Predicted Negative  Predicted Positive")
        print(f"Actual Negative        {self.tn:6d}              {self.fp:6d}")
        print(f"Actual Positive        {self.fn:6d}              {self.tp:6d}")
        print()
        
        # Metrics
        metrics = self.get_all_metrics()
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric:20s}: {value:6.2%}")
            else:
                print(f"  {metric:20s}: {value:6d}")
        print()
    
    def plot_confusion_matrix(self, title: str = "Confusion Matrix"):
        """
        Visualize confusion matrix with a heatmap.
        
        Production ML teams use this to quickly spot model weaknesses.
        """
        cm = self.confusion_matrix()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'],
                    cbar_kws={'label': 'Count'})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        return plt


class ScenarioSimulator:
    """
    Simulate real-world scenarios to understand metric tradeoffs.
    
    Based on actual challenges faced by Google, Meta, Tesla, etc.
    """
    
    @staticmethod
    def medical_diagnosis_scenario():
        """
        Scenario: Cancer detection system
        Ground truth: 5% of patients have cancer
        
        Two competing models:
        Model A: Conservative (high precision, lower recall)
        Model B: Aggressive (high recall, lower precision)
        """
        print("\n" + "="*70)
        print("SCENARIO 1: Medical Cancer Detection System".center(70))
        print("="*70)
        print("\nContext: Out of 1000 patients, 50 actually have cancer (5%)")
        print("Critical question: Which is worse?")
        print("  - Missing cancer cases (False Negative)")
        print("  - Unnecessary biopsies (False Positive)")
        print()
        
        # Generate realistic data
        np.random.seed(42)
        total_patients = 1000
        actual_cancer = 50
        
        # Ground truth: 50 have cancer, 950 don't
        y_true = np.array([1] * actual_cancer + [0] * (total_patients - actual_cancer))
        
        # Model A: Conservative - catches 70% of cancer, rarely false alarms
        y_pred_a = y_true.copy()
        # Miss 30% of cancers (15 false negatives)
        cancer_indices = np.where(y_true == 1)[0]
        missed = np.random.choice(cancer_indices, size=15, replace=False)
        y_pred_a[missed] = 0
        # Create 20 false positives
        healthy_indices = np.where(y_true == 0)[0]
        false_alarms = np.random.choice(healthy_indices, size=20, replace=False)
        y_pred_a[false_alarms] = 1
        
        # Model B: Aggressive - catches 95% of cancer, more false alarms
        y_pred_b = y_true.copy()
        # Miss only 5% of cancers (2-3 false negatives)
        missed_b = np.random.choice(cancer_indices, size=3, replace=False)
        y_pred_b[missed_b] = 0
        # Create 100 false positives
        false_alarms_b = np.random.choice(healthy_indices, size=100, replace=False)
        y_pred_b[false_alarms_b] = 1
        
        calc_a = MetricsCalculator(y_true, y_pred_a)
        calc_b = MetricsCalculator(y_true, y_pred_b)
        
        print("Model A (Conservative):")
        calc_a.print_report("Model A: High Precision Approach")
        
        print("\nModel B (Aggressive):")
        calc_b.print_report("Model B: High Recall Approach")
        
        print("\n💡 INSIGHT:")
        print("   Healthcare prioritizes RECALL over precision.")
        print("   Missing cancer (FN) = potential death")
        print("   False alarm (FP) = unnecessary test but saves lives")
        print("   → Model B is better despite lower precision!")
        
        return calc_a, calc_b
    
    @staticmethod
    def spam_filter_scenario():
        """
        Scenario: Email spam detection (like Gmail)
        Ground truth: 15% of emails are spam
        
        Key challenge: False positives (marking good emails as spam)
        hurt user trust more than false negatives (spam in inbox)
        """
        print("\n" + "="*70)
        print("SCENARIO 2: Email Spam Filter (Gmail-style)".center(70))
        print("="*70)
        print("\nContext: Out of 1000 emails, 150 are spam (15%)")
        print("Critical question: Which is worse?")
        print("  - Good email in spam folder (False Positive) - USER ANGRY!")
        print("  - Spam in inbox (False Negative) - annoying but tolerable")
        print()
        
        np.random.seed(43)
        total_emails = 1000
        actual_spam = 150
        
        y_true = np.array([1] * actual_spam + [0] * (total_emails - actual_spam))
        
        # Model with high precision (few false positives)
        y_pred = y_true.copy()
        
        # Miss 20% of spam (30 false negatives) - spam gets through
        spam_indices = np.where(y_true == 1)[0]
        missed_spam = np.random.choice(spam_indices, size=30, replace=False)
        y_pred[missed_spam] = 0
        
        # But only 5 false positives (good emails marked as spam)
        good_email_indices = np.where(y_true == 0)[0]
        false_spam = np.random.choice(good_email_indices, size=5, replace=False)
        y_pred[false_spam] = 1
        
        calc = MetricsCalculator(y_true, y_pred)
        calc.print_report("Gmail Spam Filter")
        
        print("\n💡 INSIGHT:")
        print("   Email filters prioritize PRECISION over recall.")
        print("   Blocking good email (FP) = user loses important message")
        print("   Missing spam (FN) = slight annoyance")
        print(f"   → {calc.precision():.1%} precision means users trust the spam folder!")
        
        return calc
    
    @staticmethod
    def fraud_detection_scenario():
        """
        Scenario: Credit card fraud detection (like Stripe)
        Ground truth: 0.5% of transactions are fraudulent
        
        Balance required: Both FP and FN have real costs
        """
        print("\n" + "="*70)
        print("SCENARIO 3: Credit Card Fraud Detection".center(70))
        print("="*70)
        print("\nContext: Out of 10,000 transactions, 50 are fraudulent (0.5%)")
        print("Cost analysis:")
        print("  - False Positive: Block legitimate customer ($50 avg) + angry customer")
        print("  - False Negative: Fraudulent charge ($250 avg) + chargebacks")
        print()
        
        np.random.seed(44)
        total_transactions = 10000
        actual_fraud = 50
        
        y_true = np.array([1] * actual_fraud + [0] * (total_transactions - actual_fraud))
        
        # Balanced model
        y_pred = y_true.copy()
        
        # Miss 10% of fraud (5 false negatives)
        fraud_indices = np.where(y_true == 1)[0]
        missed_fraud = np.random.choice(fraud_indices, size=5, replace=False)
        y_pred[missed_fraud] = 0
        
        # Block 0.5% of legitimate transactions (50 false positives)
        legit_indices = np.where(y_true == 0)[0]
        blocked_legit = np.random.choice(legit_indices, size=50, replace=False)
        y_pred[blocked_legit] = 1
        
        calc = MetricsCalculator(y_true, y_pred)
        calc.print_report("Stripe Fraud Detection")
        
        # Calculate business costs
        fp_cost = calc.fp * 50  # $50 per blocked legitimate transaction
        fn_cost = calc.fn * 250  # $250 per missed fraud
        total_cost = fp_cost + fn_cost
        
        print(f"\n💰 BUSINESS IMPACT:")
        print(f"   False Positive Cost: ${fp_cost:,} ({calc.fp} blocked customers)")
        print(f"   False Negative Cost: ${fn_cost:,} ({calc.fn} missed frauds)")
        print(f"   Total Cost: ${total_cost:,}")
        print(f"\n   → Must balance precision AND recall based on cost structure!")
        
        return calc


def compare_models(models: List[Tuple[str, MetricsCalculator]]):
    """
    Compare multiple models side by side.
    
    Similar to A/B testing dashboards at tech companies.
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON DASHBOARD".center(70))
    print("="*70 + "\n")
    
    comparison_data = []
    for name, calc in models:
        comparison_data.append([
            name,
            f"{calc.accuracy():.2%}",
            f"{calc.precision():.2%}",
            f"{calc.recall():.2%}",
            f"{calc.f1_score():.2%}"
        ])
    
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    print(tabulate(comparison_data, headers=headers, tablefmt="grid"))
    print()


def main():
    """
    Run all demonstrations to understand evaluation metrics.
    """
    print("\n" + "🎯" * 35)
    print("Day 43: Model Evaluation Metrics - Complete Demo")
    print("🎯" * 35)
    
    # Run all scenarios
    print("\n📊 Running Real-World Scenario Simulations...\n")
    
    # Scenario 1: Medical
    calc_med_a, calc_med_b = ScenarioSimulator.medical_diagnosis_scenario()
    
    # Scenario 2: Spam
    calc_spam = ScenarioSimulator.spam_filter_scenario()
    
    # Scenario 3: Fraud
    calc_fraud = ScenarioSimulator.fraud_detection_scenario()
    
    # Compare all models
    compare_models([
        ("Medical Model A", calc_med_a),
        ("Medical Model B", calc_med_b),
        ("Spam Filter", calc_spam),
        ("Fraud Detection", calc_fraud)
    ])
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS".center(70))
    print("="*70)
    print("""
1. Accuracy is misleading with imbalanced data
   → Always check precision and recall separately

2. Different problems need different metrics:
   → Healthcare: High recall (don't miss diseases)
   → Spam filters: High precision (don't block good emails)
   → Fraud detection: Balance both based on costs

3. There's always a precision-recall tradeoff
   → Increasing one usually decreases the other
   → Choose based on business consequences

4. F1 score helps when you need balance
   → Harmonic mean prevents high precision from masking low recall

5. Production ML teams:
   → Never rely on a single metric
   → Always examine the confusion matrix
   → Calculate business impact of FP vs FN
    """)
    
    print("\n✅ Day 43 Complete! You now understand how to measure what matters.")
    print("🚀 Tomorrow: Linear Regression - building predictive models!\n")


if __name__ == "__main__":
    main()
