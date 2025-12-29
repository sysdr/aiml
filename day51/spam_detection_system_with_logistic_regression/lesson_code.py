"""
Day 51: Production-Ready Spam Detection System

This implementation demonstrates the complete spam detection pipeline
used by Gmail, Outlook, and Yahoo Mail to protect billions of users.

Architecture mirrors production systems at scale.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Tuple, Dict
import time


class SpamDetector:
    """
    Production-grade spam detection system using logistic regression.
    
    This class encapsulates the complete pipeline:
    - Data loading and preprocessing
    - Feature engineering
    - Model training
    - Evaluation and metrics
    - Deployment simulation
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize spam detector with reproducible random state."""
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath: str = 'spambase.data') -> pd.DataFrame:
        """
        Load and prepare spam dataset.
        
        The spambase dataset contains 4,601 emails with 57 features:
        - 48 word frequency features (e.g., 'free', 'money', 'click')
        - 6 character frequency features (e.g., '!', '$', '#')
        - 3 capital letter features
        - 1 label (spam=1, ham=0)
        
        This mirrors production feature extraction pipelines.
        """
        # Load column names
        with open('spambase.names', 'r') as f:
            columns = f.read().strip().split(',')
        
        # Load data
        data = pd.read_csv(filepath, header=None, names=columns)
        
        print(f"📊 Dataset loaded: {len(data)} emails")
        print(f"   - Spam: {data['is_spam'].sum()} ({data['is_spam'].mean()*100:.1f}%)")
        print(f"   - Ham: {len(data) - data['is_spam'].sum()} ({(1-data['is_spam'].mean())*100:.1f}%)")
        print(f"   - Features: {len(columns)-1}")
        
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target vector.
        
        In production, this step includes:
        - Feature scaling/normalization
        - Missing value imputation
        - Outlier detection
        - Feature selection
        """
        # Separate features and target
        X = data.drop('is_spam', axis=1).values
        y = data['is_spam'].values
        
        # Store feature names for interpretation
        self.feature_names = data.drop('is_spam', axis=1).columns.tolist()
        
        print(f"\n🔧 Features prepared:")
        print(f"   - Feature matrix shape: {X.shape}")
        print(f"   - Target vector shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """
        Split data into training and test sets.
        
        Production systems use:
        - 70% training
        - 15% validation (hyperparameter tuning)
        - 15% test (final evaluation)
        
        We use 80/20 split for simplicity.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"\n📊 Data split:")
        print(f"   - Training set: {len(self.X_train)} emails")
        print(f"   - Test set: {len(self.X_test)} emails")
        print(f"   - Train spam ratio: {self.y_train.mean()*100:.1f}%")
        print(f"   - Test spam ratio: {self.y_test.mean()*100:.1f}%")
    
    def train_model(self, C: float = 1.0, max_iter: int = 1000):
        """
        Train logistic regression classifier.
        
        Hyperparameters:
        - C: Inverse regularization strength (lower = stronger regularization)
        - max_iter: Maximum iterations for convergence
        - solver: 'liblinear' is fast for binary classification
        
        Production systems use grid search to optimize C.
        """
        print(f"\n🎯 Training logistic regression...")
        print(f"   - Regularization (C): {C}")
        print(f"   - Max iterations: {max_iter}")
        
        start_time = time.time()
        
        self.model = LogisticRegression(
            C=C,
            solver='liblinear',
            max_iter=max_iter,
            random_state=self.random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        
        # Calculate training accuracy
        train_accuracy = self.model.score(self.X_train, self.y_train)
        test_accuracy = self.model.score(self.X_test, self.y_test)
        
        print(f"\n✅ Training complete in {training_time:.2f} seconds")
        print(f"   - Training accuracy: {train_accuracy*100:.2f}%")
        print(f"   - Test accuracy: {test_accuracy*100:.2f}%")
        
        return self.model
    
    def evaluate_model(self) -> Dict:
        """
        Comprehensive model evaluation.
        
        Production metrics:
        - Precision: Of predicted spam, how many are truly spam?
        - Recall: Of actual spam, how many did we catch?
        - F1-Score: Harmonic mean of precision and recall
        - ROC-AUC: Area under ROC curve
        """
        print(f"\n📊 Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            self.y_test, 
            y_pred, 
            target_names=['Ham', 'Spam'],
            digits=3
        ))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        print(f"                 Predicted Ham    Predicted Spam")
        print(f"Actual Ham       {cm[0,0]:^13}    {cm[0,1]:^14}")
        print(f"Actual Spam      {cm[1,0]:^13}    {cm[1,1]:^14}")
        print("="*60)
        
        # Calculate ROC-AUC
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")
        
        # Save evaluation report
        with open('evaluation_report.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("SPAM DETECTION MODEL - EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(classification_report(
                self.y_test, 
                y_pred, 
                target_names=['Ham', 'Spam']
            ))
            f.write(f"\n\nROC-AUC Score: {roc_auc:.4f}\n")
            f.write(f"\nConfusion Matrix:\n{cm}\n")
        
        print("\n✅ Evaluation report saved to: evaluation_report.txt")
        
        return {
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Visualize confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam']
        )
        plt.title('Spam Detection - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        print("✅ Confusion matrix saved to: confusion_matrix.png")
        plt.close()
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float):
        """Visualize ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Spam Detection - ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300)
        print("✅ ROC curve saved to: roc_curve.png")
        plt.close()
    
    def analyze_features(self, top_n: int = 20):
        """
        Analyze most important features for spam detection.
        
        Production systems use this to:
        - Understand model behavior
        - Detect feature drift
        - Debug misclassifications
        """
        print(f"\n🔍 Analyzing top {top_n} features...")
        
        # Get feature coefficients
        coefficients = self.model.coef_[0]
        
        # Create dataframe of features and coefficients
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute coefficient
        feature_importance = feature_importance.sort_values(
            'abs_coefficient', 
            ascending=False
        )
        
        print(f"\n{'='*70}")
        print("TOP SPAM INDICATORS (Positive coefficients)")
        print(f"{'='*70}")
        spam_indicators = feature_importance[feature_importance['coefficient'] > 0].head(top_n)
        for idx, row in spam_indicators.iterrows():
            print(f"{row['feature']:40s} {row['coefficient']:>10.4f}")
        
        print(f"\n{'='*70}")
        print("TOP HAM INDICATORS (Negative coefficients)")
        print(f"{'='*70}")
        ham_indicators = feature_importance[feature_importance['coefficient'] < 0].head(top_n)
        for idx, row in ham_indicators.iterrows():
            print(f"{row['feature']:40s} {row['coefficient']:>10.4f}")
        
        return feature_importance
    
    def simulate_production_inference(self, num_emails: int = 1000):
        """
        Simulate production spam filtering at scale.
        
        Production systems process:
        - Gmail: 3+ million emails/second globally
        - Single server: ~10,000 emails/second
        - This simulation: ~100 emails/second (single thread)
        """
        print(f"\n🚀 Simulating production inference...")
        print(f"   - Processing {num_emails} emails")
        
        # Sample emails from test set
        sample_indices = np.random.choice(
            len(self.X_test), 
            size=min(num_emails, len(self.X_test)), 
            replace=False
        )
        X_sample = self.X_test[sample_indices]
        
        # Measure inference time
        start_time = time.time()
        predictions = self.model.predict(X_sample)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        emails_per_second = len(X_sample) / inference_time
        latency_per_email = (inference_time / len(X_sample)) * 1000  # ms
        
        print(f"\n✅ Production simulation complete:")
        print(f"   - Total time: {inference_time:.4f} seconds")
        print(f"   - Throughput: {emails_per_second:.0f} emails/second")
        print(f"   - Latency: {latency_per_email:.2f} ms/email")
        print(f"   - Spam detected: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.1f}%)")
        
        # Production comparison
        print(f"\n📊 Production scale comparison:")
        print(f"   - Your laptop: ~{int(emails_per_second)} emails/sec")
        print(f"   - Production server: ~10,000 emails/sec")
        print(f"   - Gmail globally: ~3,000,000 emails/sec")
        print(f"   - Scale factor: {3000000/emails_per_second:.0f}x (horizontal scaling)")
    
    def save_model(self, model_path: str = 'spam_model.pkl'):
        """Save trained model for deployment."""
        joblib.dump(self.model, model_path)
        print(f"\n💾 Model saved to: {model_path}")
        print(f"   - Load with: model = joblib.load('{model_path}')")
    
    @staticmethod
    def load_model(model_path: str = 'spam_model.pkl'):
        """Load saved model."""
        return joblib.load(model_path)


def main():
    """Execute complete spam detection pipeline."""
    print("="*70)
    print("DAY 51: PRODUCTION-READY SPAM DETECTION SYSTEM")
    print("="*70)
    print("\n🎯 Building Gmail-scale spam filtering pipeline...\n")
    
    # Initialize detector
    detector = SpamDetector(random_state=42)
    
    # Step 1: Load data
    data = detector.load_data()
    
    # Step 2: Prepare features
    X, y = detector.prepare_features(data)
    
    # Step 3: Split data
    detector.split_data(X, y)
    
    # Step 4: Train model
    detector.train_model(C=1.0, max_iter=1000)
    
    # Step 5: Evaluate model
    eval_metrics = detector.evaluate_model()
    
    # Step 6: Visualize results
    detector.plot_confusion_matrix(eval_metrics['confusion_matrix'])
    detector.plot_roc_curve(
        eval_metrics['fpr'], 
        eval_metrics['tpr'], 
        eval_metrics['roc_auc']
    )
    
    # Step 7: Analyze features
    feature_importance = detector.analyze_features(top_n=15)
    
    # Step 8: Simulate production
    detector.simulate_production_inference(num_emails=1000)
    
    # Step 9: Save model
    detector.save_model()
    
    print("\n" + "="*70)
    print("✅ SPAM DETECTION SYSTEM COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   - spam_model.pkl (trained model)")
    print("   - evaluation_report.txt (metrics)")
    print("   - confusion_matrix.png (visualization)")
    print("   - roc_curve.png (visualization)")
    print("\n🎯 Next: Run 'pytest test_lesson.py -v' to verify")


if __name__ == "__main__":
    main()
