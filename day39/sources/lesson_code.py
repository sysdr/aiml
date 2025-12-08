"""
Day 39: Supervised vs. Unsupervised Learning
Demonstrates the fundamental difference between learning with labels vs. discovering patterns
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
import matplotlib.pyplot as plt
import pandas as pd


class SupervisedLearningDemo:
    """
    Demonstrates supervised learning with a spam classifier.
    Key insight: We KNOW the correct answers (spam/not spam) for training data.
    """
    
    def __init__(self):
        self.model = MultinomialNB()
        self.accuracy = None
        
    def generate_synthetic_email_data(self, n_samples=1000):
        """
        Generate synthetic email data with labels.
        Real-world parallel: This is like Tesla's labeled driving footage
        or Gmail's human-labeled spam emails.
        """
        np.random.seed(42)
        
        # Features: word counts for "free", "win", "viagra", "meeting", "report"
        # Spam emails have high counts for promotional words
        spam_emails = np.random.poisson(lam=[8, 6, 5, 1, 1], size=(n_samples//2, 5))
        
        # Normal emails have high counts for work-related words
        normal_emails = np.random.poisson(lam=[1, 1, 0.5, 7, 6], size=(n_samples//2, 5))
        
        # Combine data
        X = np.vstack([spam_emails, normal_emails])
        
        # Labels: 1 for spam, 0 for not spam
        # THIS IS THE KEY: We have the correct answers
        y = np.array([1] * (n_samples//2) + [0] * (n_samples//2))
        
        return X, y
    
    def train_and_evaluate(self):
        """
        Train supervised learning model with labeled data.
        """
        print("=" * 60)
        print("SUPERVISED LEARNING: Email Spam Classification")
        print("=" * 60)
        
        # Generate labeled training data
        X, y = self.generate_synthetic_email_data(1000)
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining data: {len(X_train)} emails with labels")
        print(f"Test data: {len(X_test)} emails with labels")
        
        # SUPERVISED LEARNING KEY: .fit(X, y) - we provide labels (y)
        print("\nTraining model with labeled examples...")
        self.model.fit(X_train, y_train)
        
        # Make predictions on test data
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy - we can do this because we have true labels!
        self.accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {self.accuracy:.2%}")
        print("\nKey Insight: We can measure 'correctness' because")
        print("we have true labels for comparison.")
        
        # Show example predictions
        print("\n" + "-" * 60)
        print("Example Predictions:")
        print("-" * 60)
        
        for i in range(5):
            true_label = "Spam" if y_test[i] == 1 else "Not Spam"
            pred_label = "Spam" if y_pred[i] == 1 else "Not Spam"
            correct = "✓" if y_test[i] == y_pred[i] else "✗"
            
            print(f"Email {i+1}: True={true_label:10s} | Predicted={pred_label:10s} {correct}")
        
        return self.accuracy


class UnsupervisedLearningDemo:
    """
    Demonstrates unsupervised learning with customer segmentation.
    Key insight: We DON'T have labels. Algorithm discovers patterns.
    """
    
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.silhouette = None
        
    def generate_customer_data(self, n_customers=300):
        """
        Generate customer purchase data WITHOUT labels.
        Real-world parallel: This is like Spotify's user listening patterns
        or Netflix's viewing habits - no one pre-defined the customer types.
        """
        np.random.seed(42)
        
        # Three natural customer groups (but we pretend we don't know this!)
        # Budget shoppers: low spending, high frequency
        budget = np.random.normal(loc=[20, 15], scale=[5, 3], size=(n_customers//3, 2))
        
        # Premium shoppers: high spending, low frequency
        premium = np.random.normal(loc=[80, 5], scale=[10, 2], size=(n_customers//3, 2))
        
        # Regular shoppers: medium spending, medium frequency
        regular = np.random.normal(loc=[50, 10], scale=[8, 3], size=(n_customers//3, 2))
        
        # Combine - NO LABELS PROVIDED
        X = np.vstack([budget, premium, regular])
        
        return X
    
    def discover_segments(self):
        """
        Use unsupervised learning to discover customer segments.
        """
        print("\n" + "=" * 60)
        print("UNSUPERVISED LEARNING: Customer Segmentation")
        print("=" * 60)
        
        # Generate unlabeled data
        X = self.generate_customer_data(300)
        
        print(f"\nData: {len(X)} customers with purchase patterns")
        print("Features: [average_purchase_amount, purchases_per_month]")
        print("\nKey Insight: NO LABELS PROVIDED - algorithm discovers patterns")
        
        # UNSUPERVISED LEARNING KEY: .fit(X) - no labels (no y)
        print("\nDiscovering customer segments...")
        self.model.fit(X)
        
        # Get cluster assignments
        clusters = self.model.predict(X)
        
        # Silhouette score: measures how well-separated clusters are
        # (We can't measure "accuracy" - there's no ground truth!)
        self.silhouette = silhouette_score(X, clusters)
        
        print(f"\nSilhouette Score: {self.silhouette:.3f}")
        print("(Measures cluster quality: -1 to +1, higher is better)")
        
        # Analyze discovered segments
        print("\n" + "-" * 60)
        print("Discovered Customer Segments:")
        print("-" * 60)
        
        for cluster_id in range(3):
            cluster_data = X[clusters == cluster_id]
            avg_purchase = cluster_data[:, 0].mean()
            avg_frequency = cluster_data[:, 1].mean()
            n_customers = len(cluster_data)
            
            # Interpret the segment
            if avg_purchase < 35:
                segment_type = "Budget Shoppers"
            elif avg_purchase > 65:
                segment_type = "Premium Shoppers"
            else:
                segment_type = "Regular Shoppers"
            
            print(f"\nSegment {cluster_id+1} ({segment_type}):")
            print(f"  Customers: {n_customers}")
            print(f"  Avg Purchase: ${avg_purchase:.2f}")
            print(f"  Purchases/Month: {avg_frequency:.1f}")
        
        print("\n" + "-" * 60)
        print("Business Impact: Marketing can target each segment differently!")
        print("-" * 60)
        
        # Visualize clusters
        self.visualize_clusters(X, clusters)
        
        return self.silhouette
    
    def visualize_clusters(self, X, clusters):
        """
        Create visualization of discovered customer segments.
        """
        plt.figure(figsize=(10, 6))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        labels = ['Segment 1', 'Segment 2', 'Segment 3']
        
        for cluster_id in range(3):
            cluster_points = X[clusters == cluster_id]
            plt.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1],
                c=colors[cluster_id],
                label=labels[cluster_id],
                alpha=0.6,
                s=50
            )
        
        # Plot cluster centers
        centers = self.model.cluster_centers_
        plt.scatter(
            centers[:, 0], 
            centers[:, 1],
            c='black',
            marker='X',
            s=200,
            label='Cluster Centers',
            edgecolors='white',
            linewidths=2
        )
        
        plt.xlabel('Average Purchase Amount ($)', fontsize=12)
        plt.ylabel('Purchases per Month', fontsize=12)
        plt.title('Customer Segmentation (Unsupervised Learning)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('customer_segments.png', dpi=150)
        print("\n✓ Visualization saved as 'customer_segments.png'")
        plt.close()


class ComparisonFramework:
    """
    Compare supervised vs unsupervised learning approaches.
    Helps students understand when to use each.
    """
    
    @staticmethod
    def print_comparison():
        """
        Print comprehensive comparison table.
        """
        print("\n" + "=" * 80)
        print("SUPERVISED vs UNSUPERVISED LEARNING: Decision Framework")
        print("=" * 80)
        
        comparison = pd.DataFrame({
            'Aspect': [
                'Training Data',
                'Algorithm Goal',
                'Evaluation',
                'Use Case Examples',
                'Cost to Implement',
                'When to Use'
            ],
            'Supervised Learning': [
                'Requires labeled data (X, y)',
                'Learn mapping: input → output',
                'Can measure accuracy/error',
                'Spam detection, image recognition',
                'High (need labeled data)',
                'When you have correct answers'
            ],
            'Unsupervised Learning': [
                'Only needs input data (X)',
                'Discover hidden patterns',
                'No ground truth to compare',
                'Customer segmentation, anomaly detection',
                'Lower (no labeling needed)',
                'When exploring data structure'
            ]
        })
        
        print("\n" + comparison.to_string(index=False))
        
        print("\n" + "=" * 80)
        print("PRODUCTION REALITY: Companies use BOTH approaches together")
        print("=" * 80)
        print("""
Examples:
  
  Netflix: 
    • Supervised: Predict ratings based on past ratings
    • Unsupervised: Discover similar movies for recommendations
  
  Tesla:
    • Supervised: Detect objects (cars, pedestrians, signs) 
    • Unsupervised: Discover rare driving scenarios
  
  Gmail:
    • Supervised: Classify known spam patterns
    • Unsupervised: Detect emerging spam tactics
        """)


def main():
    """
    Run complete demonstration of supervised vs unsupervised learning.
    """
    print("\n" + "█" * 80)
    print("DAY 39: SUPERVISED VS. UNSUPERVISED LEARNING")
    print("Understanding AI's Two Fundamental Approaches")
    print("█" * 80)
    
    # Run supervised learning demo
    supervised = SupervisedLearningDemo()
    supervised_score = supervised.train_and_evaluate()
    
    # Run unsupervised learning demo
    unsupervised = UnsupervisedLearningDemo(n_clusters=3)
    unsupervised_score = unsupervised.discover_segments()
    
    # Show comparison framework
    ComparisonFramework.print_comparison()
    
    # Summary insights
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. API Difference:
   • Supervised: model.fit(X, y)  ← Requires labels
   • Unsupervised: model.fit(X)   ← No labels needed

2. Evaluation Difference:
   • Supervised: Can measure "correctness" (accuracy, error)
   • Unsupervised: Evaluate "usefulness" (silhouette, business value)

3. Data Requirement Difference:
   • Supervised: Need expensive labeled data
   • Unsupervised: Work with abundant unlabeled data

4. Production Reality:
   • Most AI systems combine both approaches
   • Start with unsupervised to explore, add supervised for precision
   • Use unsupervised to find patterns, supervised to act on them
    """)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
Tomorrow (Day 40): Regression vs. Classification
  • Dive deeper into supervised learning's two main types
  • Learn when to predict numbers vs. categories
  • Build both a regression and classification system

Challenge: Think of 3 ML problems at your favorite app/company.
For each, ask:
  1. Is this supervised or unsupervised?
  2. If supervised, predicting numbers or categories?
    """)
    
    print("\n✓ Lesson complete! Check 'customer_segments.png' for visualization.\n")


if __name__ == "__main__":
    main()
