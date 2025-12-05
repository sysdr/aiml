"""
Day 38: The Machine Learning Workflow
Complete sentiment analysis pipeline demonstrating all 7 ML workflow stages
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
import json
from datetime import datetime
from pathlib import Path


class MLWorkflowPipeline:
    """
    Complete ML Pipeline following the 7-stage workflow:
    1. Problem Definition
    2. Data Collection
    3. Data Preparation
    4. Model Training
    5. Model Evaluation
    6. Deployment (save model)
    7. Monitoring (prediction tracking)
    """
    
    def __init__(self, problem_type="sentiment_analysis"):
        self.problem_type = problem_type
        self.model = None
        self.vectorizer = None
        self.metrics = {}
        self.logs = []
        
    def log_event(self, stage, message):
        """Track workflow progress for monitoring"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "stage": stage,
            "message": message
        }
        self.logs.append(log_entry)
        print(f"[{stage}] {message}")
    
    # STAGE 1: Problem Definition
    def define_problem(self):
        """
        Define what we're predicting and success criteria
        Real-world: This is documented in project specs before coding starts
        """
        self.log_event("PROBLEM_DEFINITION", "Binary classification: Positive vs Negative sentiment")
        self.log_event("PROBLEM_DEFINITION", "Success metric: F1-score > 0.80")
        self.log_event("PROBLEM_DEFINITION", "Business impact: Identify negative reviews for customer service")
        
        return {
            "task": "binary_classification",
            "target_variable": "sentiment",
            "success_threshold": 0.80,
            "evaluation_metric": "f1_score"
        }
    
    # STAGE 2: Data Collection
    def collect_data(self):
        """
        Simulate data collection from multiple sources
        Real-world: This would pull from databases, APIs, or data lakes
        """
        self.log_event("DATA_COLLECTION", "Loading product review data...")
        
        # Simulated product reviews (in production: from database/API)
        reviews = [
            # Positive reviews
            "This product is amazing! Best purchase ever.",
            "Absolutely love it. Works perfectly and exceeded expectations.",
            "Great quality and fast shipping. Highly recommend!",
            "Fantastic product. Worth every penny.",
            "Excellent! Exactly what I needed.",
            "Best product in this category. Very satisfied.",
            "Outstanding quality and great customer service.",
            "Perfect! No complaints at all.",
            "Wonderful product. My family loves it.",
            "Impressive quality. Will buy again.",
            # More positive variations
            "Superb product with excellent features.",
            "Amazing value for money. Very happy.",
            "Exceeded all my expectations. Brilliant!",
            "Great purchase. Would definitely recommend.",
            "Fantastic quality. Impressive design.",
            
            # Negative reviews
            "Terrible product. Waste of money.",
            "Awful quality. Broke after one week.",
            "Very disappointed. Does not work as advertised.",
            "Poor quality and bad customer service.",
            "Do not buy! Complete waste of money.",
            "Horrible experience. Product is defective.",
            "Worst purchase ever. Requesting refund.",
            "Cheap materials. Fell apart immediately.",
            "Terrible. Not worth a single penny.",
            "Awful product. Save your money.",
            # More negative variations
            "Very poor quality. Completely disappointed.",
            "Bad product. Does not meet expectations.",
            "Terrible experience from start to finish.",
            "Horrible quality. Broke on first use.",
            "Worst product I've ever bought.",
            
            # More varied examples for better training
            "Love this so much! Perfect for my needs.",
            "Great product but shipping was slow.",
            "Amazing quality, highly recommended to everyone!",
            "Disappointed with the quality for the price.",
            "Excellent features and very easy to use.",
            "Not happy with this purchase at all.",
            "Brilliant product! Works like a charm.",
            "Poor design and materials used.",
            "Fantastic! Better than expected.",
            "Waste of time and money unfortunately."
        ]
        
        sentiments = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Positive
            1, 1, 1, 1, 1,  # More positive
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Negative
            0, 0, 0, 0, 0,  # More negative
            1, 1, 1, 0, 1, 0, 1, 0, 1, 0   # Mixed
        ]
        
        # Create DataFrame (standard format for ML workflows)
        df = pd.DataFrame({
            'review_text': reviews,
            'sentiment': sentiments
        })
        
        self.log_event("DATA_COLLECTION", f"Collected {len(df)} reviews")
        return df
    
    # STAGE 3: Data Preparation
    def prepare_data(self, df):
        """
        Clean, transform, and split data for training
        Real-world: Handles missing values, outliers, normalization
        """
        self.log_event("DATA_PREPARATION", "Starting data preparation...")
        
        # Check for missing values (critical in real datasets)
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            self.log_event("DATA_PREPARATION", f"Found {missing_count} missing values - handling...")
            df = df.dropna()
        
        # Text preprocessing (lowercase, remove special chars if needed)
        df['review_text'] = df['review_text'].str.lower()
        
        # Split into features (X) and target (y)
        X = df['review_text']
        y = df['sentiment']
        
        # Train-test split (80-20 is standard)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.log_event("DATA_PREPARATION", f"Train set: {len(X_train)} samples")
        self.log_event("DATA_PREPARATION", f"Test set: {len(X_test)} samples")
        
        # Feature extraction: Convert text to numerical features
        # TF-IDF is standard for text in production systems
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1
        )
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        self.log_event("DATA_PREPARATION", f"Feature dimensions: {X_train_vectorized.shape[1]}")
        
        return X_train_vectorized, X_test_vectorized, y_train, y_test
    
    # STAGE 4: Model Training
    def train_model(self, X_train, y_train):
        """
        Train ML model on prepared data
        Real-world: This stage might take hours/days for large datasets
        """
        self.log_event("MODEL_TRAINING", "Training logistic regression model...")
        
        # Logistic Regression: Fast, interpretable, production-proven
        # Used by many companies for text classification
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )
        
        # The actual training happens here
        self.model.fit(X_train, y_train)
        
        self.log_event("MODEL_TRAINING", "Model training complete")
        
        # Show feature importance (top words for each class)
        feature_names = self.vectorizer.get_feature_names_out()
        top_positive = np.argsort(self.model.coef_[0])[-5:]
        top_negative = np.argsort(self.model.coef_[0])[:5]
        
        self.log_event("MODEL_TRAINING", f"Top positive words: {[feature_names[i] for i in top_positive]}")
        self.log_event("MODEL_TRAINING", f"Top negative words: {[feature_names[i] for i in top_negative]}")
        
        return self.model
    
    # STAGE 5: Model Evaluation
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive evaluation using multiple metrics
        Real-world: This determines if model is ready for production
        """
        self.log_event("MODEL_EVALUATION", "Evaluating model performance...")
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate all standard metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Log results
        self.log_event("MODEL_EVALUATION", f"Accuracy: {accuracy:.3f}")
        self.log_event("MODEL_EVALUATION", f"Precision: {precision:.3f}")
        self.log_event("MODEL_EVALUATION", f"Recall: {recall:.3f}")
        self.log_event("MODEL_EVALUATION", f"F1-Score: {f1:.3f}")
        
        # Confusion matrix (shows where model makes mistakes)
        cm = confusion_matrix(y_test, y_pred)
        self.log_event("MODEL_EVALUATION", f"Confusion Matrix:\n{cm}")
        
        # Check if meets success criteria
        success_threshold = 0.80
        if f1 >= success_threshold:
            self.log_event("MODEL_EVALUATION", f"✅ Model meets F1 threshold ({success_threshold})")
        else:
            self.log_event("MODEL_EVALUATION", f"⚠️ Model below F1 threshold ({success_threshold})")
        
        return self.metrics
    
    # STAGE 6: Deployment (Model Saving)
    def deploy_model(self, model_dir="models"):
        """
        Save model for production use
        Real-world: This deploys to cloud servers, edge devices, etc.
        """
        self.log_event("DEPLOYMENT", "Saving model artifacts...")
        
        # Create models directory
        Path(model_dir).mkdir(exist_ok=True)
        
        # Save model and vectorizer
        model_path = f"{model_dir}/sentiment_model.pkl"
        vectorizer_path = f"{model_dir}/vectorizer.pkl"
        metrics_path = f"{model_dir}/metrics.json"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save metrics for monitoring
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.log_event("DEPLOYMENT", f"Model saved to {model_path}")
        self.log_event("DEPLOYMENT", f"Vectorizer saved to {vectorizer_path}")
        self.log_event("DEPLOYMENT", "✅ Model ready for production")
        
        return model_path
    
    # STAGE 7: Monitoring (Make Predictions)
    def predict(self, new_reviews):
        """
        Make predictions on new data with monitoring
        Real-world: This runs continuously, serving millions of predictions
        """
        self.log_event("MONITORING", f"Making predictions on {len(new_reviews)} new reviews...")
        
        # Vectorize new reviews
        X_new = self.vectorizer.transform(new_reviews)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_new)
        probabilities = self.model.predict_proba(X_new)
        
        # Format results
        results = []
        for i, (review, pred, prob) in enumerate(zip(new_reviews, predictions, probabilities)):
            sentiment = "Positive" if pred == 1 else "Negative"
            confidence = max(prob)
            
            result = {
                'review': review,
                'sentiment': sentiment,
                'confidence': confidence
            }
            results.append(result)
            
            self.log_event("MONITORING", f"Review {i+1}: {sentiment} (confidence: {confidence:.2f})")
        
        return results


def run_complete_workflow():
    """Execute the entire 7-stage ML workflow"""
    print("=" * 70)
    print("DAY 38: COMPLETE MACHINE LEARNING WORKFLOW")
    print("=" * 70)
    print()
    
    # Initialize pipeline
    pipeline = MLWorkflowPipeline()
    
    # STAGE 1: Define the problem
    print("\n" + "="*70)
    print("STAGE 1: PROBLEM DEFINITION")
    print("="*70)
    problem_spec = pipeline.define_problem()
    
    # STAGE 2: Collect data
    print("\n" + "="*70)
    print("STAGE 2: DATA COLLECTION")
    print("="*70)
    df = pipeline.collect_data()
    
    # STAGE 3: Prepare data
    print("\n" + "="*70)
    print("STAGE 3: DATA PREPARATION")
    print("="*70)
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    
    # STAGE 4: Train model
    print("\n" + "="*70)
    print("STAGE 4: MODEL TRAINING")
    print("="*70)
    model = pipeline.train_model(X_train, y_train)
    
    # STAGE 5: Evaluate model
    print("\n" + "="*70)
    print("STAGE 5: MODEL EVALUATION")
    print("="*70)
    metrics = pipeline.evaluate_model(X_test, y_test)
    
    # STAGE 6: Deploy model
    print("\n" + "="*70)
    print("STAGE 6: DEPLOYMENT")
    print("="*70)
    model_path = pipeline.deploy_model()
    
    # STAGE 7: Make predictions (monitoring)
    print("\n" + "="*70)
    print("STAGE 7: MONITORING & PREDICTION")
    print("="*70)
    
    # Test with new reviews
    new_reviews = [
        "This product is incredible! Love it so much.",
        "Terrible quality. Very disappointed with this purchase.",
        "Good value for money. Works as expected.",
        "Awful experience. Would not recommend to anyone."
    ]
    
    results = pipeline.predict(new_reviews)
    
    # Display final results
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE - FINAL RESULTS")
    print("="*70)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.3f}")
    
    print("\nSample Predictions:")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. Review: \"{result['review']}\"")
        print(f"     Prediction: {result['sentiment']} (confidence: {result['confidence']:.1%})")
    
    print("\n" + "="*70)
    print("✅ ML Workflow Pipeline Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Try modifying the reviews in the code")
    print("  2. Experiment with different models (try RandomForest)")
    print("  3. Add more data preprocessing steps")
    print("  4. Run pytest test_lesson.py to verify your understanding")
    print()


if __name__ == "__main__":
    run_complete_workflow()
