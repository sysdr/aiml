"""
Day 60: Random Forests and Ensemble Methods
Production-grade implementation demonstrating ensemble learning superiority
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

class CustomerChurnPredictor:
    """
    Production-grade Random Forest classifier for customer churn prediction.
    Mirrors systems used at SaaS companies like Salesforce, HubSpot, and Intercom.
    """
    
    def __init__(self, n_estimators=200, max_depth=15, random_state=42):
        """
        Initialize ensemble models for comparison.
        
        Args:
            n_estimators: Number of trees in the forest (production: 100-500)
            max_depth: Maximum tree depth to prevent overfitting
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        
        # Single decision tree (baseline)
        self.single_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
        
        # Random Forest (ensemble)
        self.random_forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=random_state
        )
        
        self.feature_names = None
        self.is_trained = False
        
    def generate_synthetic_data(self, n_samples=5000):
        """
        Generate synthetic customer data for churn prediction.
        Simulates real-world customer behavior data.
        
        Args:
            n_samples: Number of customer records to generate
            
        Returns:
            X: Feature matrix
            y: Target labels (0=retained, 1=churned)
        """
        # Generate classification data with realistic properties
        X, y = make_classification(
            n_samples=n_samples,
            n_features=15,
            n_informative=10,
            n_redundant=3,
            n_clusters_per_class=2,
            weights=[0.7, 0.3],  # 30% churn rate (realistic)
            flip_y=0.05,  # 5% label noise
            random_state=self.random_state
        )
        
        # Create meaningful feature names
        self.feature_names = [
            'account_age_days',
            'total_purchases',
            'avg_purchase_value',
            'days_since_last_purchase',
            'support_tickets',
            'login_frequency',
            'feature_usage_score',
            'payment_failures',
            'discount_usage',
            'referral_count',
            'mobile_app_usage',
            'email_engagement',
            'plan_type',
            'contract_length',
            'satisfaction_score'
        ]
        
        # Convert to DataFrame for easier handling
        X = pd.DataFrame(X, columns=self.feature_names)
        
        return X, y
    
    def train(self, X_train, y_train):
        """
        Train both single tree and random forest models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Training models...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")
        print()
        
        # Train single decision tree
        print("Training single decision tree...")
        self.single_tree.fit(X_train, y_train)
        
        # Train random forest
        print("Training random forest ensemble...")
        self.random_forest.fit(X_train, y_train)
        
        self.is_trained = True
        print("✓ Training complete!")
        print()
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive evaluation comparing single tree vs random forest.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        print("=" * 70)
        print("MODEL EVALUATION: Single Tree vs Random Forest")
        print("=" * 70)
        print()
        
        # Single tree predictions
        tree_pred = self.single_tree.predict(X_test)
        tree_proba = self.single_tree.predict_proba(X_test)[:, 1]
        
        # Random forest predictions
        rf_pred = self.random_forest.predict(X_test)
        rf_proba = self.random_forest.predict_proba(X_test)[:, 1]
        
        # Calculate metrics for single tree
        tree_metrics = {
            'accuracy': accuracy_score(y_test, tree_pred),
            'precision': precision_recall_fscore_support(y_test, tree_pred, average='binary')[0],
            'recall': precision_recall_fscore_support(y_test, tree_pred, average='binary')[1],
            'f1': precision_recall_fscore_support(y_test, tree_pred, average='binary')[2],
            'roc_auc': roc_auc_score(y_test, tree_proba)
        }
        
        # Calculate metrics for random forest
        rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_recall_fscore_support(y_test, rf_pred, average='binary')[0],
            'recall': precision_recall_fscore_support(y_test, rf_pred, average='binary')[1],
            'f1': precision_recall_fscore_support(y_test, rf_pred, average='binary')[2],
            'roc_auc': roc_auc_score(y_test, rf_proba),
            'oob_score': self.random_forest.oob_score_
        }
        
        # Display comparison
        print("SINGLE DECISION TREE:")
        print(f"  Accuracy:  {tree_metrics['accuracy']:.4f}")
        print(f"  Precision: {tree_metrics['precision']:.4f}")
        print(f"  Recall:    {tree_metrics['recall']:.4f}")
        print(f"  F1 Score:  {tree_metrics['f1']:.4f}")
        print(f"  ROC AUC:   {tree_metrics['roc_auc']:.4f}")
        print()
        
        print("RANDOM FOREST ENSEMBLE:")
        print(f"  Accuracy:  {rf_metrics['accuracy']:.4f} ({self._improvement(tree_metrics['accuracy'], rf_metrics['accuracy'])})")
        print(f"  Precision: {rf_metrics['precision']:.4f} ({self._improvement(tree_metrics['precision'], rf_metrics['precision'])})")
        print(f"  Recall:    {rf_metrics['recall']:.4f} ({self._improvement(tree_metrics['recall'], rf_metrics['recall'])})")
        print(f"  F1 Score:  {rf_metrics['f1']:.4f} ({self._improvement(tree_metrics['f1'], rf_metrics['f1'])})")
        print(f"  ROC AUC:   {rf_metrics['roc_auc']:.4f} ({self._improvement(tree_metrics['roc_auc'], rf_metrics['roc_auc'])})")
        print(f"  OOB Score: {rf_metrics['oob_score']:.4f} (free validation metric)")
        print()
        
        # Detailed classification reports
        print("DETAILED CLASSIFICATION REPORT - Random Forest:")
        print(classification_report(y_test, rf_pred, target_names=['Retained', 'Churned']))
        
        return {
            'single_tree': tree_metrics,
            'random_forest': rf_metrics,
            'predictions': {
                'tree': tree_pred,
                'rf': rf_pred
            }
        }
    
    def _improvement(self, baseline, improved):
        """Calculate percentage improvement with sign."""
        diff = improved - baseline
        pct = (diff / baseline) * 100
        sign = '+' if diff > 0 else ''
        return f"{sign}{pct:.2f}%"
    
    def analyze_feature_importance(self):
        """
        Analyze and visualize which features matter most for predictions.
        Critical for production deployment and feature engineering.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before feature analysis")
        
        print("=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)
        print()
        
        # Extract feature importances
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.random_forest.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        print(importances.head(10).to_string(index=False))
        print()
        
        # Visualize feature importances
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        top_10 = importances.head(10)
        plt.barh(range(len(top_10)), top_10['importance'])
        plt.yticks(range(len(top_10)), top_10['feature'])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importances\n(Random Forest)')
        plt.gca().invert_yaxis()
        
        plt.subplot(1, 2, 2)
        cumsum = importances['importance'].cumsum()
        plt.plot(range(len(cumsum)), cumsum, marker='o')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Feature Importance')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        print("✓ Feature importance plot saved as 'feature_importance.png'")
        print()
        
        return importances
    
    def demonstrate_ensemble_power(self, X_test, y_test):
        """
        Demonstrate why ensembles work: individual tree diversity.
        Shows predictions from first 5 trees vs ensemble.
        """
        print("=" * 70)
        print("ENSEMBLE DIVERSITY DEMONSTRATION")
        print("=" * 70)
        print()
        
        # Get predictions from first 5 individual trees
        print("Predictions from first 5 trees for first 10 test samples:")
        print("(1 = churn predicted, 0 = retention predicted)")
        print()
        
        # Extract individual tree predictions
        tree_predictions = []
        for tree in self.random_forest.estimators_[:5]:
            preds = tree.predict(X_test[:10])
            tree_predictions.append(preds)
        
        # Display as table
        tree_df = pd.DataFrame(
            np.array(tree_predictions).T,
            columns=[f'Tree {i+1}' for i in range(5)]
        )
        tree_df['Ensemble Vote'] = self.random_forest.predict(X_test[:10])
        tree_df['True Label'] = y_test[:10]
        tree_df['Ensemble Correct'] = (tree_df['Ensemble Vote'] == tree_df['True Label']).astype(int)
        
        print(tree_df.to_string(index=True))
        print()
        print("Key Insight: Individual trees disagree, but ensemble vote is usually correct!")
        print()

def main():
    """
    Main execution demonstrating Random Forests for customer churn prediction.
    """
    print("=" * 70)
    print("DAY 60: RANDOM FORESTS AND ENSEMBLE METHODS")
    print("Customer Churn Prediction System")
    print("=" * 70)
    print()
    
    # Initialize predictor
    predictor = CustomerChurnPredictor(n_estimators=200, max_depth=15)
    
    # Generate synthetic data
    print("Generating synthetic customer data...")
    X, y = predictor.generate_synthetic_data(n_samples=5000)
    print(f"Generated {len(X)} customer records")
    print(f"Features: {X.shape[1]}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train models
    predictor.train(X_train, y_train)
    
    # Evaluate and compare
    results = predictor.evaluate(X_test, y_test)
    
    # Feature importance analysis
    importances = predictor.analyze_feature_importance()
    
    # Demonstrate ensemble diversity
    predictor.demonstrate_ensemble_power(X_test, y_test)
    
    # Production insights
    print("=" * 70)
    print("PRODUCTION DEPLOYMENT INSIGHTS")
    print("=" * 70)
    print()
    print("Random Forest Advantages in Production:")
    print("  1. Robust to noisy/missing data (handles real-world messiness)")
    print("  2. No feature scaling required (unlike neural networks)")
    print("  3. Interpretable via feature importance (regulatory compliance)")
    print("  4. Parallelizable training (scales to massive datasets)")
    print("  5. OOB score provides free validation (no separate validation set needed)")
    print()
    print("When to use Random Forests:")
    print("  ✓ Tabular data with mixed types (numerical + categorical)")
    print("  ✓ Need model interpretability (feature importance)")
    print("  ✓ Medium-sized datasets (1K-1M samples)")
    print("  ✓ Classification or regression tasks")
    print("  ✗ Image/text data (use deep learning instead)")
    print("  ✗ Need probability calibration (use calibrated classifiers)")
    print()
    print("Tomorrow: Apply Random Forests to credit card fraud detection!")
    print()

if __name__ == "__main__":
    main()
