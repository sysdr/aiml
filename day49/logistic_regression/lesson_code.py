"""
Day 49: Logistic Regression for Binary Classification
Production-ready spam classifier implementation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class SpamClassifier:
    """
    Production-ready binary classifier for spam detection.
    Similar to systems used by Gmail, Outlook, and other email providers.
    """
    
    def __init__(self, max_features=1000, random_state=42):
        """
        Initialize the spam classifier.
        
        Args:
            max_features: Maximum number of features for TF-IDF vectorization
            random_state: Random seed for reproducibility
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True
        )
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs'
        )
        self.is_trained = False
        
    def prepare_data(self, texts, labels):
        """
        Prepare text data for training.
        
        Args:
            texts: List of email texts
            labels: List of binary labels (0=ham, 1=spam)
            
        Returns:
            Tuple of (features, labels)
        """
        # Convert texts to TF-IDF features
        features = self.vectorizer.fit_transform(texts)
        return features, np.array(labels)
    
    def train(self, X_train, y_train):
        """
        Train the logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("🎯 Training spam classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("✅ Training complete!")
        
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to classify
            
        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Features to classify
            
        Returns:
            Probability scores for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        return self.model.predict_proba(X)
    
    def predict_text(self, texts):
        """
        Predict spam/ham for new text messages.
        
        Args:
            texts: List of email texts
            
        Returns:
            Predictions and probabilities
        """
        features = self.vectorizer.transform(texts)
        predictions = self.predict(features)
        probabilities = self.predict_proba(features)
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text[:50] + '...' if len(text) > 50 else text,
                'prediction': 'SPAM' if predictions[i] == 1 else 'HAM',
                'spam_probability': probabilities[i][1],
                'confidence': max(probabilities[i])
            })
        return results


class ModelEvaluator:
    """
    Comprehensive evaluation toolkit for binary classification.
    """
    
    @staticmethod
    def evaluate_model(y_true, y_pred, y_proba=None):
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
        """
        Visualize confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Where to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam']
        )
        plt.title('Confusion Matrix - Spam Classification', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {save_path}")
        plt.close()
        
    @staticmethod
    def plot_roc_curve(y_true, y_proba, save_path='roc_curve.png'):
        """
        Plot ROC curve and calculate AUC.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Where to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Spam Classification', fontsize=14, pad=20)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 ROC curve saved to {save_path}")
        plt.close()
        
    @staticmethod
    def print_classification_report(y_true, y_pred):
        """
        Print detailed classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            y_true, y_pred,
            target_names=['Ham', 'Spam'],
            digits=4
        ))


def create_sample_dataset():
    """
    Create a sample spam/ham dataset for demonstration.
    In production, this would come from real email data.
    """
    # Sample spam messages
    spam_messages = [
        "WINNER! You've won a $1000 prize! Click here now!",
        "Congratulations! You've been selected for a free iPhone!",
        "URGENT: Your account needs verification. Click immediately!",
        "Make money fast! Work from home! Guaranteed income!",
        "FREE VIAGRA! Limited time offer! Order now!",
        "You've inherited $10 million! Contact us immediately!",
        "Hot singles in your area! Click here to meet them!",
        "Lose weight fast with this one weird trick!",
        "CONGRATULATIONS! You're our lucky winner today!",
        "Get rich quick! This is not a scam! Guaranteed!",
        "FREE MONEY! No strings attached! Click now!",
        "Your PayPal account has been suspended! Verify now!",
        "AMAZING OFFER! Buy one get ten free! Limited stock!",
        "You've been pre-approved for a loan! Apply now!",
        "CLICK HERE FOR FREE PRIZES! Don't miss out!",
    ] * 10  # Repeat to get more samples
    
    # Sample legitimate messages (ham)
    ham_messages = [
        "Hi, can we schedule a meeting for tomorrow at 2pm?",
        "Thanks for your email. I'll review the document and get back to you.",
        "The project deadline is next Friday. Please submit your work by then.",
        "Could you please send me the quarterly report when you have a chance?",
        "Great job on the presentation! The client was very impressed.",
        "Reminder: Team lunch is scheduled for Thursday at noon.",
        "I've attached the files you requested. Let me know if you need anything else.",
        "The meeting has been rescheduled to next week. I'll send an updated invite.",
        "Please review the attached contract and let me know your thoughts.",
        "Thank you for your support on this project. Much appreciated!",
        "Can you help me with this issue when you get a chance?",
        "The system maintenance is scheduled for this weekend.",
        "I've completed the analysis. Here are my findings.",
        "Let's discuss the new features in our next standup.",
        "Your order has been shipped and should arrive by Friday.",
    ] * 10  # Repeat to get more samples
    
    # Combine messages and create labels
    all_messages = spam_messages + ham_messages
    labels = [1] * len(spam_messages) + [0] * len(ham_messages)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': all_messages,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    """
    Main execution function - complete spam classification pipeline.
    """
    print("="*70)
    print("DAY 49: LOGISTIC REGRESSION FOR BINARY CLASSIFICATION")
    print("Building a Production-Ready Spam Classifier")
    print("="*70)
    
    # Step 1: Create dataset
    print("\n📦 Step 1: Creating sample dataset...")
    df = create_sample_dataset()
    print(f"✅ Dataset created: {len(df)} emails")
    print(f"   - Spam emails: {sum(df['label'] == 1)}")
    print(f"   - Ham emails: {sum(df['label'] == 0)}")
    
    # Step 2: Split data
    print("\n🔀 Step 2: Splitting data into train/test sets...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['text'], df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    print(f"✅ Train set: {len(X_train_text)} emails")
    print(f"✅ Test set: {len(X_test_text)} emails")
    
    # Step 3: Initialize and prepare classifier
    print("\n🔧 Step 3: Initializing spam classifier...")
    classifier = SpamClassifier(max_features=1000)
    X_train, _ = classifier.prepare_data(X_train_text, y_train)
    X_test = classifier.vectorizer.transform(X_test_text)
    print(f"✅ Feature extraction complete")
    print(f"   - Feature dimensions: {X_train.shape[1]}")
    
    # Step 4: Train model
    print("\n🎓 Step 4: Training logistic regression model...")
    classifier.train(X_train, y_train)
    
    # Step 5: Make predictions
    print("\n🔮 Step 5: Making predictions on test set...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    print("✅ Predictions complete!")
    
    # Step 6: Evaluate model
    print("\n📊 Step 6: Evaluating model performance...")
    evaluator = ModelEvaluator()
    
    metrics = evaluator.evaluate_model(y_test, y_pred, y_proba)
    print("\nPerformance Metrics:")
    print(f"  • Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  • Precision: {metrics['precision']:.4f}")
    print(f"  • Recall:    {metrics['recall']:.4f}")
    print(f"  • F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  • ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Print detailed classification report
    evaluator.print_classification_report(y_test, y_pred)
    
    # Generate visualizations
    evaluator.plot_confusion_matrix(y_test, y_pred)
    evaluator.plot_roc_curve(y_test, y_proba)
    
    # Step 7: Test with new emails
    print("\n🧪 Step 7: Testing with new email samples...")
    new_emails = [
        "Hi, let's meet for coffee tomorrow to discuss the project.",
        "WINNER! You've been selected for a FREE vacation package!",
        "Please review the attached document at your earliest convenience.",
        "URGENT! Your bank account has been compromised! Click here NOW!",
    ]
    
    results = classifier.predict_text(new_emails)
    
    print("\nPrediction Results:")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        print(f"\nEmail {i}: {result['text']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Spam Probability: {result['spam_probability']:.4f}")
        print(f"  Confidence: {result['confidence']:.4f}")
    
    print("\n" + "="*70)
    print("✅ SPAM CLASSIFIER COMPLETE!")
    print("="*70)
    print("\n📚 Next Steps:")
    print("  1. Experiment with different thresholds for classification")
    print("  2. Try adding more sophisticated features (email headers, links, etc.)")
    print("  3. Collect real email data to improve the model")
    print("  4. Move on to Day 50: Multi-Class Classification")
    print("\n🎯 You've built a production-ready binary classifier!")


if __name__ == "__main__":
    main()
