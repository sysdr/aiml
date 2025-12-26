#!/bin/bash

# Day 50: Multi-Class Classification with Logistic Regression
# Implementation Package Generator

echo "Generating Day 50 implementation files..."

# Create setup.sh
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 50: Multi-Class Classification environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo "Setup complete! Virtual environment ready."
echo "To activate: source venv/bin/activate"
echo "To run lesson: python lesson_code.py"
echo "To test: pytest test_lesson.py -v"
EOF

chmod +x setup.sh

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.26.4
scikit-learn==1.5.1
pandas==2.2.2
matplotlib==3.9.0
seaborn==0.13.2
pytest==8.2.2
EOF

# Create lesson_code.py
cat > lesson_code.py << 'EOF'
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
        self.model = LogisticRegression(
            multi_class='ovr',  # One-vs-Rest strategy
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
        coef = self.model.coef_[category_idx]
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
            multi_class='multinomial',  # Softmax strategy
            solver='lbfgs',  # Required for multinomial
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
EOF

# Create test_lesson.py
cat > test_lesson.py << 'EOF'
"""
Tests for Day 50: Multi-Class Classification
Verifies both OvR and Softmax implementations
"""

import pytest
import numpy as np
from lesson_code import (
    NewsCategorizerOvR,
    NewsCategorizerSoftmax,
    generate_sample_data
)


class TestMultiClassClassification:
    """Test multi-class classification implementations"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        articles, labels = generate_sample_data(n_samples=200)
        return articles, labels
    
    def test_data_generation(self, sample_data):
        """Test synthetic data generation"""
        articles, labels = sample_data
        
        assert len(articles) == 200
        assert len(labels) == 200
        assert all(isinstance(article, str) for article in articles)
        assert all(label in [0, 1, 2, 3] for label in labels)
        assert len(set(labels)) == 4  # All 4 categories present
    
    def test_ovr_classifier(self, sample_data):
        """Test One-vs-Rest classifier"""
        articles, labels = sample_data
        
        # Split data
        split_idx = int(len(articles) * 0.8)
        X_train, X_test = articles[:split_idx], articles[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        
        # Train classifier
        classifier = NewsCategorizerOvR(max_iter=500)
        classifier.fit(X_train, y_train)
        
        # Make predictions
        predictions = classifier.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1, 2, 3] for pred in predictions)
        
        # Check accuracy is reasonable
        accuracy = np.mean(predictions == y_test)
        assert accuracy > 0.5  # Should do better than random (0.25)
    
    def test_softmax_classifier(self, sample_data):
        """Test Softmax classifier"""
        articles, labels = sample_data
        
        # Split data
        split_idx = int(len(articles) * 0.8)
        X_train, X_test = articles[:split_idx], articles[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        
        # Train classifier
        classifier = NewsCategorizerSoftmax(max_iter=500)
        classifier.fit(X_train, y_train)
        
        # Make predictions
        predictions = classifier.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1, 2, 3] for pred in predictions)
        
        # Check accuracy is reasonable
        accuracy = np.mean(predictions == y_test)
        assert accuracy > 0.5  # Should do better than random (0.25)
    
    def test_probability_outputs(self, sample_data):
        """Test probability outputs for both classifiers"""
        articles, labels = sample_data
        
        # Train both classifiers
        ovr = NewsCategorizerOvR(max_iter=500)
        ovr.fit(articles[:150], labels[:150])
        
        softmax = NewsCategorizerSoftmax(max_iter=500)
        softmax.fit(articles[:150], labels[:150])
        
        # Get probability predictions
        test_article = [articles[0]]
        
        probs_ovr = ovr.predict_proba(test_article)
        probs_softmax = softmax.predict_proba(test_article)
        
        # Check shapes
        assert probs_ovr.shape == (1, 4)
        assert probs_softmax.shape == (1, 4)
        
        # Softmax probabilities should sum to 1.0
        assert np.abs(probs_softmax.sum() - 1.0) < 0.001
        
        # All probabilities should be between 0 and 1
        assert np.all(probs_ovr >= 0) and np.all(probs_ovr <= 1)
        assert np.all(probs_softmax >= 0) and np.all(probs_softmax <= 1)
    
    def test_top_features_extraction(self, sample_data):
        """Test feature importance extraction"""
        articles, labels = sample_data
        
        classifier = NewsCategorizerSoftmax(max_iter=500)
        classifier.fit(articles, labels)
        
        # Get top features for each category
        for category_idx in range(4):
            top_features = classifier.get_top_features(category_idx, top_n=5)
            
            assert len(top_features) == 5
            assert all(isinstance(feature, str) for feature, _ in top_features)
            assert all(isinstance(score, (int, float)) for _, score in top_features)
    
    def test_prediction_consistency(self, sample_data):
        """Test that predictions are deterministic"""
        articles, labels = sample_data
        
        classifier = NewsCategorizerSoftmax(max_iter=500)
        classifier.fit(articles[:150], labels[:150])
        
        test_article = [articles[160]]
        
        # Make multiple predictions
        pred1 = classifier.predict(test_article)
        pred2 = classifier.predict(test_article)
        pred3 = classifier.predict(test_article)
        
        assert pred1[0] == pred2[0] == pred3[0]


class TestProductionPatterns:
    """Test production-ready patterns"""
    
    def test_batch_prediction(self):
        """Test efficient batch prediction"""
        articles, labels = generate_sample_data(n_samples=100)
        
        classifier = NewsCategorizerSoftmax(max_iter=500)
        classifier.fit(articles[:80], labels[:80])
        
        # Batch prediction
        test_batch = articles[80:]
        predictions = classifier.predict(test_batch)
        
        assert len(predictions) == len(test_batch)
    
    def test_category_mapping(self):
        """Test category name mapping"""
        classifier = NewsCategorizerOvR()
        
        assert 0 in classifier.categories
        assert 1 in classifier.categories
        assert 2 in classifier.categories
        assert 3 in classifier.categories
        
        assert classifier.categories[0] == 'Technology'
        assert classifier.categories[1] == 'Sports'
        assert classifier.categories[2] == 'Politics'
        assert classifier.categories[3] == 'Entertainment'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

# Create README.md
cat > README.md << 'EOF'
# Day 50: Multi-Class Classification with Logistic Regression

Complete implementation of One-vs-Rest and Softmax strategies for multi-class classification.

## Quick Start

```bash
# Setup environment
bash setup.sh

# Activate virtual environment
source venv/bin/activate

# Run the news categorizer
python lesson_code.py

# Run tests
pytest test_lesson.py -v
```

## What's Included

- **One-vs-Rest (OvR) Classifier**: Trains separate binary models for each class
- **Softmax Classifier**: Single model with multi-output layer
- **Performance Comparison**: Side-by-side evaluation of both strategies
- **News Categorization**: Real-world example with 4 categories
- **Visualization**: Confusion matrices for both strategies

## Key Concepts

1. **Multi-Class Extension**: Converting binary classification to handle 3+ classes
2. **OvR Strategy**: Train N binary classifiers for N classes
3. **Softmax Strategy**: Single model outputting probabilities that sum to 1.0
4. **Feature Importance**: Understanding which words indicate each category

## Expected Output

- Training time for both strategies
- Accuracy comparison (typically 85-95% on synthetic data)
- Confusion matrices showing prediction patterns
- Top features per category
- Sample predictions with probability distributions

## Production Applications

- Gmail priority categorization
- Netflix genre classification
- Google Photos face recognition
- Amazon product categorization
- Medical diagnosis systems

## Build Time

Complete setup and execution: ~3 minutes

## Files Generated

- `one-vs-rest_confusion_matrix.png`: OvR visualization
- `softmax_confusion_matrix.png`: Softmax visualization
- Test results and performance metrics
EOF

echo "✓ setup.sh"
echo "✓ requirements.txt"
echo "✓ lesson_code.py"
echo "✓ test_lesson.py"
echo "✓ README.md"
echo ""
echo "Day 50 implementation package generated successfully!"
echo ""
echo "Next steps:"
echo "  1. bash setup.sh"
echo "  2. source venv/bin/activate"
echo "  3. python lesson_code.py"
echo "  4. pytest test_lesson.py -v"

