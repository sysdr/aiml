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
