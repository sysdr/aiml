"""
Day 59: Decision Trees with Scikit-learn
Customer Churn Prediction System

This implementation demonstrates production-ready decision tree classification
similar to systems used at Netflix, Spotify, and other streaming services.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class CustomerChurnPredictor:
    """
    Production-ready customer churn prediction using decision trees.
    
    Similar to systems used at:
    - Netflix: Predicting subscription cancellations
    - Spotify: Identifying at-risk premium users
    - Amazon Prime: Forecasting membership renewals
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 50):
        """
        Initialize churn predictor with production parameters.
        
        Args:
            max_depth: Maximum tree depth (prevents overfitting)
            min_samples_split: Minimum samples to split node (ensures statistical significance)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model = None
        self.feature_names = None
        self.best_params = None
        
    def generate_synthetic_data(self, n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic customer data mimicking streaming service behavior.
        
        Features mirror real-world metrics:
        - Usage patterns (hours watched, login frequency)
        - Engagement (content diversity, completion rate)
        - Support interactions (tickets, complaints)
        - Account age and subscription tier
        """
        np.random.seed(42)
        
        # Feature generation
        data = {
            'monthly_hours': np.random.exponential(20, n_samples),
            'login_frequency': np.random.poisson(15, n_samples),
            'content_diversity': np.random.beta(2, 5, n_samples) * 100,
            'completion_rate': np.random.beta(5, 2, n_samples) * 100,
            'support_tickets': np.random.poisson(1, n_samples),
            'account_age_months': np.random.gamma(2, 5, n_samples),
            'subscription_tier': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1]),
            'payment_failures': np.random.binomial(3, 0.1, n_samples),
            'days_since_last_login': np.random.exponential(3, n_samples),
            'device_count': np.random.poisson(2, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate churn labels based on realistic patterns
        # Churn probability increases with:
        # - Low usage, high support tickets, payment failures, inactivity
        churn_score = (
            -0.3 * (df['monthly_hours'] / 30) +
            -0.2 * (df['login_frequency'] / 20) +
            -0.15 * (df['completion_rate'] / 100) +
            0.4 * (df['support_tickets'] / 5) +
            0.3 * (df['payment_failures'] / 3) +
            0.25 * (df['days_since_last_login'] / 10) +
            -0.1 * (df['content_diversity'] / 100)
        )
        
        # Add noise for realism
        churn_score += np.random.normal(0, 0.3, n_samples)
        
        # Create imbalanced dataset (15% churn rate, realistic for streaming services)
        churn_threshold = np.percentile(churn_score, 85)
        y = (churn_score > churn_threshold).astype(int)
        
        self.feature_names = df.columns.tolist()
        
        return df, pd.Series(y, name='churned')
    
    def train(self, X: pd.DataFrame, y: pd.Series, use_grid_search: bool = False) -> Dict:
        """
        Train decision tree classifier with production best practices.
        
        Args:
            X: Feature dataframe
            y: Target labels
            use_grid_search: Whether to perform hyperparameter tuning
            
        Returns:
            Training metrics dictionary
        """
        # Split data with stratification (maintains class distribution)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if use_grid_search:
            # Production hyperparameter tuning
            param_grid = {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [20, 50, 100],
                'min_samples_leaf': [10, 20, 50],
                'class_weight': ['balanced', None]
            }
            
            grid_search = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            print("Performing grid search for optimal hyperparameters...")
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best parameters: {self.best_params}")
        else:
            # Use predefined production parameters
            self.model = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=20,
                class_weight='balanced',  # Critical for imbalanced data
                random_state=42
            )
            self.model.fit(X_train, y_train)
        
        # Cross-validation for reliability assessment
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=5, scoring='roc_auc'
        )
        
        # Test set evaluation
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': self.model.score(X_test, y_test),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance rankings.
        
        Used in production to:
        - Guide data collection priorities
        - Identify key churn indicators
        - Inform product development
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def visualize_results(self, metrics: Dict):
        """
        Create production-quality visualizations for model analysis.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[0, 0],
            cbar_kws={'label': 'Count'}
        )
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. Feature Importance
        importance_df = self.get_feature_importance()
        top_10 = importance_df.head(10)
        
        axes[0, 1].barh(range(len(top_10)), top_10['importance'], color='skyblue')
        axes[0, 1].set_yticks(range(len(top_10)))
        axes[0, 1].set_yticklabels(top_10['feature'])
        axes[0, 1].set_xlabel('Importance Score')
        axes[0, 1].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
        axes[0, 1].invert_yaxis()
        
        # 3. Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(
            metrics['y_test'], 
            metrics['y_pred_proba']
        )
        axes[1, 0].plot(recall, precision, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Tree Depth Analysis
        tree = self.model.tree_
        axes[1, 1].text(0.1, 0.8, f"Tree Statistics:", fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.65, f"Max Depth: {tree.max_depth}", fontsize=10)
        axes[1, 1].text(0.1, 0.55, f"Number of Leaves: {tree.n_leaves}", fontsize=10)
        axes[1, 1].text(0.1, 0.45, f"Number of Nodes: {tree.node_count}", fontsize=10)
        axes[1, 1].text(0.1, 0.3, f"Cross-Val ROC-AUC: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}", fontsize=10)
        axes[1, 1].text(0.1, 0.2, f"Test ROC-AUC: {metrics['test_roc_auc']:.3f}", fontsize=10)
        axes[1, 1].text(0.1, 0.1, f"Test Accuracy: {metrics['test_accuracy']:.3f}", fontsize=10)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('churn_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved visualization to churn_analysis.png")
        
        return fig
    
    def visualize_tree(self, max_depth_display: int = 3):
        """
        Visualize decision tree structure (limited depth for clarity).
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            max_depth=max_depth_display,
            feature_names=self.feature_names,
            class_names=['Retained', 'Churned'],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title('Decision Tree Structure (Limited Depth)', fontsize=16, fontweight='bold')
        plt.savefig('decision_tree_structure.png', dpi=300, bbox_inches='tight')
        print("Saved tree visualization to decision_tree_structure.png")


def compare_with_baseline(X: pd.DataFrame, y: pd.Series):
    """
    Compare decision tree against baseline models.
    
    Demonstrates why decision trees are preferred for certain problems.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'Random Baseline': DummyClassifier(strategy='stratified', random_state=42),
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=50,
            class_weight='balanced',
            random_state=42
        )
    }
    
    print("\n" + "="*70)
    print("Model Comparison: Baseline vs Production Decision Tree")
    print("="*70)
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = model.score(X_test, y_test)
        
        results.append({
            'Model': name,
            'Accuracy': f"{accuracy:.3f}",
            'ROC-AUC': f"{roc_auc:.3f}"
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("="*70)


def main():
    """
    Main execution demonstrating production-ready decision tree workflow.
    """
    print("="*70)
    print("Day 59: Decision Trees with Scikit-learn")
    print("Customer Churn Prediction System")
    print("="*70)
    
    # Initialize predictor
    predictor = CustomerChurnPredictor(max_depth=10, min_samples_split=50)
    
    # Generate synthetic data
    print("\n1. Generating synthetic customer data...")
    X, y = predictor.generate_synthetic_data(n_samples=10000)
    print(f"   Dataset: {X.shape[0]} customers, {X.shape[1]} features")
    print(f"   Churn rate: {y.mean():.1%} (imbalanced dataset)")
    
    # Train model
    print("\n2. Training decision tree classifier...")
    metrics = predictor.train(X, y, use_grid_search=False)
    
    # Display results
    print("\n3. Model Performance:")
    print(f"   Cross-Validation ROC-AUC: {metrics['cv_mean']:.3f} (± {metrics['cv_std']:.3f})")
    print(f"   Test ROC-AUC: {metrics['test_roc_auc']:.3f}")
    print(f"   Test Accuracy: {metrics['test_accuracy']:.3f}")
    
    print("\n4. Classification Report:")
    print(metrics['classification_report'])
    
    # Feature importance
    print("\n5. Top 5 Most Important Features:")
    importance_df = predictor.get_feature_importance()
    print(importance_df.head().to_string(index=False))
    
    # Visualizations
    print("\n6. Generating visualizations...")
    predictor.visualize_results(metrics)
    predictor.visualize_tree(max_depth_display=3)
    
    # Baseline comparison
    print("\n7. Comparing with baseline models...")
    compare_with_baseline(X, y)
    
    # Production insights
    print("\n" + "="*70)
    print("Production Insights:")
    print("="*70)
    print("This decision tree implementation mirrors systems used at:")
    print("  • Netflix: Predicting subscription cancellations from viewing patterns")
    print("  • Spotify: Identifying at-risk premium users for retention campaigns")
    print("  • Amazon: Forecasting Prime membership renewals")
    print("  • PayPal: Detecting fraud in imbalanced transaction data")
    print("\nKey production features demonstrated:")
    print("  ✓ Handling imbalanced datasets with class_weight='balanced'")
    print("  ✓ Cross-validation for reliable performance estimates")
    print("  ✓ Feature importance for business insights")
    print("  ✓ Hyperparameter tuning with GridSearchCV")
    print("  ✓ Production-quality visualizations for stakeholder communication")
    print("="*70)


if __name__ == "__main__":
    main()
