"""
Day 50: Multi-Class Classification with Logistic Regression
Building a News Article Categorizer

This implementation demonstrates:
1. One-vs-Rest (OvR) multi-class strategy
2. Softmax (Multinomial) multi-class strategy
3. Performance comparison and evaluation
4. Real-world news categorization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import time


class NewsCategorizerOvR:
    """
    One-vs-Rest Multi-Class Classifier
    Trains separate binary classifiers for each category
    """
    
    def __init__(self, max_iter=1000):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        # Wrap LogisticRegression in OneVsRestClassifier for OvR strategy
        base_lr = LogisticRegression(
            solver='liblinear',
            max_iter=max_iter,
            random_state=42
        )
        self.model = OneVsRestClassifier(base_lr)
        self.categories = {
            0: 'Technology',
            1: 'Sports',
            2: 'Politics',
            3: 'Entertainment'
        }
    
    def fit(self, X, y):
        """Train the OvR classifier"""
        print("\n=== Training One-vs-Rest Classifier ===")
        start_time = time.time()
        
        # Convert text to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # Train OvR model (trains 4 separate binary classifiers)
        self.model.fit(X_tfidf, y)
        
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        print(f"Trained {len(self.categories)} binary classifiers")
        
        return self
    
    def predict(self, X):
        """Predict categories for new articles"""
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict(X_tfidf)
    
    def predict_proba(self, X):
        """Get probability scores for all categories"""
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict_proba(X_tfidf)
    
    def get_top_features(self, category_idx, top_n=5):
        """Get most important features for a category"""
        # OvR creates separate coefficients for each class
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.model.estimators_[category_idx].coef_[0]  # OneVsRestClassifier stores estimators
        top_indices = np.argsort(coef)[-top_n:][::-1]
        
        return [(feature_names[i], coef[i]) for i in top_indices]


class NewsCategorizerSoftmax:
    """
    Softmax (Multinomial) Multi-Class Classifier
    Single model with softmax output layer
    """
    
    def __init__(self, max_iter=1000):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = LogisticRegression(
            # Softmax strategy (multinomial) - lbfgs automatically uses multinomial
            solver='lbfgs',  # Uses multinomial (softmax) for multi-class
            max_iter=max_iter,
            random_state=42
        )
        self.categories = {
            0: 'Technology',
            1: 'Sports',
            2: 'Politics',
            3: 'Entertainment'
        }
    
    def fit(self, X, y):
        """Train the Softmax classifier"""
        print("\n=== Training Softmax Classifier ===")
        start_time = time.time()
        
        # Convert text to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # Train softmax model (single model, multi-output)
        self.model.fit(X_tfidf, y)
        
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        print(f"Single model with softmax output layer")
        
        return self
    
    def predict(self, X):
        """Predict categories for new articles"""
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict(X_tfidf)
    
    def predict_proba(self, X):
        """Get probability scores (always sum to 1.0)"""
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict_proba(X_tfidf)
    
    def get_top_features(self, category_idx, top_n=5):
        """Get most important features for a category"""
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.model.coef_[category_idx]
        top_indices = np.argsort(coef)[-top_n:][::-1]
        
        return [(feature_names[i], coef[i]) for i in top_indices]


def generate_sample_data(n_samples=1000):
    """
    Generate synthetic news articles for demonstration
    Simulates real news data with category-specific vocabulary
    """
    
    # Category-specific keywords (mimics real news patterns)
    tech_words = ['software', 'ai', 'algorithm', 'startup', 'cloud', 'data', 
                  'python', 'machine learning', 'blockchain', 'app']
    sports_words = ['game', 'player', 'score', 'team', 'championship', 'league',
                    'touchdown', 'goal', 'match', 'victory']
    politics_words = ['election', 'government', 'policy', 'legislation', 'senator',
                     'congress', 'vote', 'campaign', 'law', 'president']
    entertainment_words = ['movie', 'actor', 'premiere', 'album', 'concert', 
                          'celebrity', 'director', 'award', 'series', 'performance']
    
    articles = []
    labels = []
    
    samples_per_category = n_samples // 4
    
    for category_idx, category_words in enumerate([
        tech_words, sports_words, politics_words, entertainment_words
    ]):
        for _ in range(samples_per_category):
            # Generate article with category-specific words + some noise
            n_words = np.random.randint(15, 30)
            article_words = np.random.choice(category_words, size=n_words, replace=True)
            
            # Add some generic words (noise)
            generic_words = ['the', 'and', 'for', 'with', 'new', 'today']
            noise = np.random.choice(generic_words, size=5, replace=True)
            
            article = ' '.join(list(article_words) + list(noise))
            articles.append(article)
            labels.append(category_idx)
    
    return articles, labels


def plot_confusion_matrix(y_true, y_pred, categories, title):
    """Visualize confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories.values(),
                yticklabels=categories.values())
    plt.title(title)
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150)
    print(f"Saved {title.lower().replace(' ', '_')}.png")


def compare_strategies():
    """
    Compare One-vs-Rest and Softmax strategies
    Demonstrates production-scale multi-class classification
    """
    
    print("=" * 60)
    print("Day 50: Multi-Class Classification Comparison")
    print("=" * 60)
    
    # Generate sample news data
    print("\nGenerating sample news articles...")
    articles, labels = generate_sample_data(n_samples=1000)
    print(f"Created {len(articles)} articles across 4 categories")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        articles, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Training set: {len(X_train)} articles")
    print(f"Test set: {len(X_test)} articles")
    
    # Strategy 1: One-vs-Rest
    ovr_classifier = NewsCategorizerOvR(max_iter=1000)
    ovr_classifier.fit(X_train, y_train)
    
    y_pred_ovr = ovr_classifier.predict(X_test)
    accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
    
    print(f"\nOne-vs-Rest Accuracy: {accuracy_ovr:.4f}")
    print("\nClassification Report (OvR):")
    print(classification_report(y_test, y_pred_ovr, 
                               target_names=ovr_classifier.categories.values()))
    
    # Strategy 2: Softmax
    softmax_classifier = NewsCategorizerSoftmax(max_iter=1000)
    softmax_classifier.fit(X_train, y_train)
    
    y_pred_softmax = softmax_classifier.predict(X_test)
    accuracy_softmax = accuracy_score(y_test, y_pred_softmax)
    
    print(f"\nSoftmax Accuracy: {accuracy_softmax:.4f}")
    print("\nClassification Report (Softmax):")
    print(classification_report(y_test, y_pred_softmax,
                               target_names=softmax_classifier.categories.values()))
    
    # Visualize confusion matrices
    plot_confusion_matrix(y_test, y_pred_ovr, ovr_classifier.categories,
                         "One-vs-Rest Confusion Matrix")
    plot_confusion_matrix(y_test, y_pred_softmax, softmax_classifier.categories,
                         "Softmax Confusion Matrix")
    
    # Show top features for each category
    print("\n" + "=" * 60)
    print("Top Features per Category (Softmax Model)")
    print("=" * 60)
    
    for cat_idx, cat_name in softmax_classifier.categories.items():
        print(f"\n{cat_name}:")
        top_features = softmax_classifier.get_top_features(cat_idx, top_n=5)
        for feature, score in top_features:
            print(f"  {feature}: {score:.4f}")
    
    # Demo prediction with probabilities
    print("\n" + "=" * 60)
    print("Sample Prediction with Probabilities")
    print("=" * 60)
    
    test_articles = [
        "ai machine learning algorithm data python",
        "game touchdown player score championship",
        "election government policy vote legislation",
        "movie premiere actor celebrity award"
    ]
    
    for article in test_articles:
        pred = softmax_classifier.predict([article])[0]
        probs = softmax_classifier.predict_proba([article])[0]
        
        print(f"\nArticle: '{article}'")
        print(f"Predicted: {softmax_classifier.categories[pred]}")
        print("Probabilities:")
        for idx, prob in enumerate(probs):
            print(f"  {softmax_classifier.categories[idx]}: {prob:.4f}")
        print(f"Sum of probabilities: {probs.sum():.4f}")  # Always 1.0 for softmax


if __name__ == "__main__":
    compare_strategies()
    
    print("\n" + "=" * 60)
    print("Multi-Class Classification Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("1. OvR trains separate binary classifiers (4 models)")
    print("2. Softmax trains single multi-output model")
    print("3. Both strategies achieve similar accuracy")
    print("4. Softmax guarantees probabilities sum to 1.0")
    print("5. Choice depends on problem requirements")
    print("\nProduction systems like Gmail and Netflix use these patterns")
    print("to process millions of multi-class predictions per second.")
