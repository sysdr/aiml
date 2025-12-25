#!/usr/bin/env python3
"""Script to create the complete setup.sh file correctly"""

setup_sh_content = '''#!/bin/bash

# Day 49: Logistic Regression for Binary Classification - Complete Implementation Package
# This script generates all necessary files for the lesson

echo "🚀 Generating Day 49: Logistic Regression for Binary Classification files..."

# Create requirements.txt
cat > requirements.txt << 'REQEOF'
numpy==1.26.3
pandas==2.2.0
scikit-learn==1.4.0
matplotlib==3.8.2
seaborn==0.13.1
pytest==7.4.4
jupyter==1.0.0
REQEOF

# Create lesson_code.py
cat > lesson_code.py << 'LESSONEOF'
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
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {{auc:.3f}})', linewidth=2)
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
        print("\\n" + "="*60)
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
    print("\\n📦 Step 1: Creating sample dataset...")
    df = create_sample_dataset()
    print(f"✅ Dataset created: {{len(df)}} emails")
    print(f"   - Spam emails: {{sum(df['label'] == 1)}}")
    print(f"   - Ham emails: {{sum(df['label'] == 0)}}")
    
    # Step 2: Split data
    print("\\n🔀 Step 2: Splitting data into train/test sets...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['text'], df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    print(f"✅ Train set: {{len(X_train_text)}} emails")
    print(f"✅ Test set: {{len(X_test_text)}} emails")
    
    # Step 3: Initialize and prepare classifier
    print("\\n🔧 Step 3: Initializing spam classifier...")
    classifier = SpamClassifier(max_features=1000)
    X_train, _ = classifier.prepare_data(X_train_text, y_train)
    X_test = classifier.vectorizer.transform(X_test_text)
    print(f"✅ Feature extraction complete")
    print(f"   - Feature dimensions: {{X_train.shape[1]}}")
    
    # Step 4: Train model
    print("\\n🎓 Step 4: Training logistic regression model...")
    classifier.train(X_train, y_train)
    
    # Step 5: Make predictions
    print("\\n🔮 Step 5: Making predictions on test set...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    print("✅ Predictions complete!")
    
    # Step 6: Evaluate model
    print("\\n📊 Step 6: Evaluating model performance...")
    evaluator = ModelEvaluator()
    
    metrics = evaluator.evaluate_model(y_test, y_pred, y_proba)
    print("\\nPerformance Metrics:")
    print(f"  • Accuracy:  {{metrics['accuracy']:.4f}}")
    print(f"  • Precision: {{metrics['precision']:.4f}}")
    print(f"  • Recall:    {{metrics['recall']:.4f}}")
    print(f"  • F1-Score:  {{metrics['f1_score']:.4f}}")
    print(f"  • ROC-AUC:   {{metrics['roc_auc']:.4f}}")
    
    # Print detailed classification report
    evaluator.print_classification_report(y_test, y_pred)
    
    # Generate visualizations
    evaluator.plot_confusion_matrix(y_test, y_pred)
    evaluator.plot_roc_curve(y_test, y_proba)
    
    # Step 7: Test with new emails
    print("\\n🧪 Step 7: Testing with new email samples...")
    new_emails = [
        "Hi, let's meet for coffee tomorrow to discuss the project.",
        "WINNER! You've been selected for a FREE vacation package!",
        "Please review the attached document at your earliest convenience.",
        "URGENT! Your bank account has been compromised! Click here NOW!",
    ]
    
    results = classifier.predict_text(new_emails)
    
    print("\\nPrediction Results:")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        print(f"\\nEmail {{i}}: {{result['text']}}")
        print(f"  Prediction: {{result['prediction']}}")
        print(f"  Spam Probability: {{result['spam_probability']:.4f}}")
        print(f"  Confidence: {{result['confidence']:.4f}}")
    
    print("\\n" + "="*70)
    print("✅ SPAM CLASSIFIER COMPLETE!")
    print("="*70)
    print("\\n📚 Next Steps:")
    print("  1. Experiment with different thresholds for classification")
    print("  2. Try adding more sophisticated features (email headers, links, etc.)")
    print("  3. Collect real email data to improve the model")
    print("  4. Move on to Day 50: Multi-Class Classification")
    print("\\n🎯 You've built a production-ready binary classifier!")


if __name__ == "__main__":
    main()
LESSONEOF

# Create test_lesson.py
cat > test_lesson.py << 'TESTEOF'
"""
Test Suite for Day 49: Logistic Regression for Binary Classification
Comprehensive tests ensuring the classifier works correctly
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lesson_code import SpamClassifier, ModelEvaluator, create_sample_dataset


class TestSpamClassifier:
    """Test the SpamClassifier class"""
    
    def test_classifier_initialization(self):
        """Test that classifier initializes correctly"""
        classifier = SpamClassifier(max_features=500)
        assert classifier.is_trained == False
        assert classifier.vectorizer.max_features == 500
        
    def test_prepare_data(self):
        """Test data preparation"""
        classifier = SpamClassifier()
        texts = ["hello world", "spam message"]
        labels = [0, 1]
        
        features, y = classifier.prepare_data(texts, labels)
        assert features.shape[0] == 2
        assert len(y) == 2
        assert y[0] == 0 and y[1] == 1
        
    def test_training(self):
        """Test model training"""
        classifier = SpamClassifier()
        
        # Create simple dataset
        texts = ["legitimate email message"] * 50 + ["spam winner prize"] * 50
        labels = [0] * 50 + [1] * 50
        
        X, y = classifier.prepare_data(texts, labels)
        classifier.train(X, y)
        
        assert classifier.is_trained == True
        
    def test_prediction_before_training(self):
        """Test that prediction fails before training"""
        classifier = SpamClassifier()
        X = np.random.rand(10, 100)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            classifier.predict(X)
            
    def test_prediction_after_training(self):
        """Test predictions after training"""
        classifier = SpamClassifier()
        
        # Train on simple data
        texts = ["legitimate email"] * 50 + ["spam prize winner"] * 50
        labels = [0] * 50 + [1] * 50
        
        X, y = classifier.prepare_data(texts, labels)
        classifier.train(X, y)
        
        # Make predictions
        predictions = classifier.predict(X)
        assert len(predictions) == 100
        assert all(p in [0, 1] for p in predictions)
        
    def test_predict_proba(self):
        """Test probability predictions"""
        classifier = SpamClassifier()
        
        texts = ["good email"] * 50 + ["spam"] * 50
        labels = [0] * 50 + [1] * 50
        
        X, y = classifier.prepare_data(texts, labels)
        classifier.train(X, y)
        
        proba = classifier.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        
    def test_predict_text(self):
        """Test text prediction interface"""
        classifier = SpamClassifier()
        
        texts = ["hello friend"] * 50 + ["winner prize"] * 50
        labels = [0] * 50 + [1] * 50
        
        X, y = classifier.prepare_data(texts, labels)
        classifier.train(X, y)
        
        new_texts = ["normal email", "spam winner"]
        results = classifier.predict_text(new_texts)
        
        assert len(results) == 2
        assert all('prediction' in r for r in results)
        assert all('spam_probability' in r for r in results)
        assert all('confidence' in r for r in results)


class TestModelEvaluator:
    """Test the ModelEvaluator class"""
    
    def test_evaluate_model_basic(self):
        """Test basic metrics evaluation"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        
        metrics = evaluator.evaluate_model(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        
    def test_evaluate_model_with_proba(self):
        """Test evaluation with probabilities"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9]
        ])
        
        metrics = evaluator.evaluate_model(y_true, y_pred, y_proba)
        
        assert 'roc_auc' in metrics
        assert metrics['roc_auc'] == 1.0
        
    def test_metrics_range(self):
        """Test that all metrics are in valid range"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        metrics = evaluator.evaluate_model(y_true, y_pred)
        
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} out of range: {value}"


class TestDatasetCreation:
    """Test dataset creation functionality"""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation"""
        df = create_sample_dataset()
        
        assert 'text' in df.columns
        assert 'label' in df.columns
        assert len(df) > 0
        assert set(df['label'].unique()) == {0, 1}
        
    def test_dataset_balance(self):
        """Test that dataset is balanced"""
        df = create_sample_dataset()
        
        spam_count = sum(df['label'] == 1)
        ham_count = sum(df['label'] == 0)
        
        # Should be roughly balanced
        ratio = spam_count / ham_count
        assert 0.8 < ratio < 1.2
        
    def test_dataset_has_text(self):
        """Test that all entries have text"""
        df = create_sample_dataset()
        
        assert all(len(str(text)) > 0 for text in df['text'])


class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def test_full_pipeline(self):
        """Test the complete classification pipeline"""
        # Create dataset
        df = create_sample_dataset()
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df['text'], df['label'],
            test_size=0.2,
            random_state=42
        )
        
        # Initialize classifier
        classifier = SpamClassifier()
        
        # Prepare data
        X_train, _ = classifier.prepare_data(X_train_text, y_train)
        X_test = classifier.vectorizer.transform(X_test_text)
        
        # Train
        classifier.train(X_train, y_train)
        
        # Predict
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(y_test, y_pred, y_proba)
        
        # Check that model performs reasonably well
        assert metrics['accuracy'] > 0.7
        assert metrics['f1_score'] > 0.7
        
    def test_real_time_prediction(self):
        """Test real-time prediction on new data"""
        # Train classifier on sample data
        df = create_sample_dataset()
        classifier = SpamClassifier()
        
        X, y = classifier.prepare_data(df['text'], df['label'])
        classifier.train(X, y)
        
        # Test with new emails
        new_emails = [
            "Let's schedule a meeting for Monday",
            "WIN FREE MONEY NOW!!!"
        ]
        
        results = classifier.predict_text(new_emails)
        
        assert len(results) == 2
        # First should likely be ham, second spam
        assert results[0]['prediction'] == 'HAM'
        assert results[1]['prediction'] == 'SPAM'


def test_sklearn_logistic_regression():
    """Test that sklearn's LogisticRegression is available and working"""
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression()
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    model.fit(X, y)
    predictions = model.predict(X)
    
    assert len(predictions) == 100


def test_evaluation_metrics_available():
    """Test that all required metrics are available"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score
    )
    
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    
    assert accuracy_score(y_true, y_pred) == 1.0
    assert precision_score(y_true, y_pred) == 1.0
    assert recall_score(y_true, y_pred) == 1.0
    assert f1_score(y_true, y_pred) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
TESTEOF

# Create README.md
cat > README.md << 'READMEEOF'
# Day 49: Logistic Regression for Binary Classification

## Overview
Build a production-ready spam email classifier using logistic regression. Learn how to implement complete binary classification pipelines used by Gmail, Netflix, and other major tech companies.

## What You'll Learn
- Binary classification pipeline architecture
- Feature extraction with TF-IDF
- Model training and evaluation
- Confusion matrices and ROC curves
- Real-time prediction systems
- Production deployment patterns

## Quick Start

### Setup (5 minutes)
```bash
# Run setup script
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate
```

### Run the Lesson (10 minutes)
```bash
python lesson_code.py
```

Expected output:
- Dataset creation and splitting
- Feature extraction statistics
- Training progress
- Performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix visualization
- ROC curve plot
- Real-time predictions on new emails

### Run Tests (5 minutes)
```bash
pytest test_lesson.py -v
```

All 20+ tests should pass, validating:
- Classifier initialization
- Data preparation
- Model training
- Prediction accuracy
- Evaluation metrics
- Integration pipeline

## Project Structure
```
day_49/
├── lesson_code.py          # Main implementation
├── test_lesson.py          # Comprehensive tests
├── setup.sh                # Environment setup
├── requirements.txt        # Dependencies
├── README.md              # This file
├── confusion_matrix.png   # Generated after running
└── roc_curve.png         # Generated after running
```

## Key Concepts

### Binary Classification Pipeline
1. **Data Preparation**: Clean and preprocess raw text
2. **Feature Engineering**: Convert text to TF-IDF vectors
3. **Model Training**: Learn optimal weights via gradient descent
4. **Evaluation**: Measure performance with multiple metrics
5. **Prediction**: Make real-time classifications

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Of predicted spam, how many are actually spam
- **Recall**: Of actual spam, how many did we catch
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Classifier performance across all thresholds

### Real-World Applications
- **Gmail**: Spam detection (99.9% accuracy)
- **Amazon**: Fraud detection (millions of transactions/day)
- **Tesla**: Object classification (pedestrian yes/no)
- **LinkedIn**: Connection suggestions
- **YouTube**: Content moderation

## Customization Ideas
1. Adjust classification threshold based on cost/benefit
2. Add more sophisticated features (email headers, links)
3. Experiment with different vectorization techniques
4. Implement online learning for continuous improvement
5. Build ensemble models combining multiple classifiers

## Troubleshooting

**Import errors**: Ensure virtual environment is activated
```bash
source venv/bin/activate
```

**Low accuracy**: Dataset might be too small or imbalanced
- Solution: Collect more training data
- Solution: Adjust class weights in LogisticRegression

**Slow training**: Feature dimension too high
- Solution: Reduce max_features in TfidfVectorizer
- Solution: Use feature selection techniques

## Performance Benchmarks
Expected metrics on sample dataset:
- Accuracy: >90%
- Precision: >85%
- Recall: >85%
- F1-Score: >85%
- ROC-AUC: >0.90

## Next Steps
- **Day 50**: Multi-class classification with softmax
- **Day 51**: Feature engineering for text data
- **Day 52**: Hyperparameter tuning and cross-validation

## Resources
- [scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

---
**Time to Complete**: 2-3 hours  
**Difficulty**: Intermediate  
**Prerequisites**: Day 48 (Logistic Regression Theory)
READMEEOF

# Create setup.sh LAST (to avoid overwriting this script while running)
cat > setup.sh << 'SETUPEOF'
#!/bin/bash

echo "🔧 Setting up Day 49: Logistic Regression for Binary Classification..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✅ Setup complete! Activate the environment with: source venv/bin/activate"
echo "📚 Run the lesson with: python lesson_code.py"
echo "🧪 Run tests with: pytest test_lesson.py -v"
SETUPEOF

chmod +x setup.sh

echo "✅ All files generated successfully!"
echo ""
echo "📁 Generated files:"
echo "  - setup.sh"
echo "  - requirements.txt"
echo "  - lesson_code.py"
echo "  - test_lesson.py"
echo "  - README.md"
echo ""
echo "🚀 To get started:"
echo "  1. chmod +x setup.sh && ./setup.sh"
echo "  2. source venv/bin/activate"
echo "  3. python lesson_code.py"
echo "  4. pytest test_lesson.py -v"
echo ""
echo "🎯 Happy learning!"
'''

with open('setup.sh', 'w') as f:
    f.write(setup_sh_content)

import os
os.chmod('setup.sh', 0o755)
print("Created setup.sh successfully!")

