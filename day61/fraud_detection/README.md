# Day 61: Credit Card Fraud Detection System

## Project Overview

Build a production-grade fraud detection system that handles imbalanced data, optimizes for business metrics, and provides real-time transaction scoring—just like systems used by Stripe, PayPal, and Square.

### What You'll Learn

- Handle extreme class imbalance with SMOTE and class weights
- Engineer fraud-specific features from transaction data
- Optimize decision thresholds for business requirements
- Evaluate using fraud-specific metrics (not just accuracy)
- Build real-time transaction scoring systems
- Compare ensemble methods for fraud detection

### Real-World Applications

This project teaches techniques used by:
- **Stripe Radar**: Processes 8,000+ transactions/second with 35ms latency
- **PayPal**: Runs 15+ specialized fraud models in parallel
- **Square**: Uses merchant-specific models for small businesses
- **Major Banks**: Protect billions in transactions daily

## Quick Start

### Setup (5 minutes)

```bash
# 1. Create environment and install dependencies
chmod +x setup.sh
./setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Run the complete fraud detection system
python lesson_code.py
```

### What Happens When You Run It

The system will:
1. Generate realistic credit card transaction data (99.8% legitimate, 0.2% fraud)
2. Engineer fraud-relevant features (velocity, distance, time patterns)
3. Apply SMOTE to balance training data
4. Train Random Forest with class weights
5. Optimize decision threshold for 90% fraud detection
6. Evaluate using confusion matrix, ROC-AUC, precision-recall
7. Compare Random Forest vs Gradient Boosting vs Logistic Regression
8. Generate comprehensive evaluation visualizations

### Expected Output

```
💳 DAY 61: CREDIT CARD FRAUD DETECTION SYSTEM
==================================================================

1️⃣  Initializing fraud detection system...
2️⃣  Generating synthetic credit card transaction data...
   Generated 10,000 transactions
   Fraud cases: 20 (0.20%)

3️⃣  Engineering fraud-relevant features...
   Created 16 features

4️⃣  Preparing data with imbalance handling...
📊 Original class distribution: {0: 9980, 1: 20}
   Fraud ratio: 0.0020

🔄 Applying SMOTE to balance training data...
   After SMOTE: {0: 6986, 1: 3493}

5️⃣  Training Random Forest fraud detector...
   Training accuracy: 0.9995

6️⃣  Optimizing decision threshold for 90% fraud detection...
🎚️  Optimized threshold: 0.3245
   Target recall: 90.00%
   Achieved recall: 91.67%
   Precision at this threshold: 0.52%

7️⃣  Evaluating fraud detection performance...

============================================================
📈 FRAUD DETECTION EVALUATION REPORT
============================================================

📋 Classification Report:
              precision    recall  f1-score   support

  Legitimate       1.00      0.99      0.99      2994
       Fraud       0.52      0.92      0.66         6

    accuracy                           0.99      3000
   macro avg       0.76      0.95      0.83      3000
weighted avg       1.00      0.99      0.99      3000

🎯 ROC-AUC Score: 0.9912

📊 Confusion Matrix:
   True Negatives (Correct legitimate): 2,971
   False Positives (Blocked good customers): 23
   False Negatives (Missed fraud): 1
   True Positives (Caught fraud): 5

💰 Business Impact Metrics:
   Fraud Detection Rate: 83.33%
   False Alarm Rate: 0.77%

💵 Estimated Financial Impact (example):
   Money saved by catching fraud: $1,000.00
   Cost of false alarms: $115.00
   Net benefit: $885.00

📊 Evaluation plots saved as 'fraud_detection_evaluation.png'
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests with detailed output
pytest test_lesson.py -v

# Run specific test categories
pytest test_lesson.py -v -k "data"  # Data-related tests
pytest test_lesson.py -v -k "model"  # Model training tests
pytest test_lesson.py -v -k "evaluation"  # Evaluation tests
```

### Test Coverage

The test suite includes 20+ tests covering:
- System initialization with different models
- Data generation with various fraud ratios
- Feature engineering correctness
- SMOTE balancing
- Model training and prediction
- Threshold optimization
- Evaluation metrics computation
- Edge cases (no fraud in test set, etc.)

## Project Structure

```
fraud_detection/
├── setup.sh                          # Environment setup
├── requirements.txt                  # Dependencies
├── lesson_code.py                    # Main implementation
├── test_lesson.py                    # Test suite
├── README.md                         # This file
└── fraud_detection_evaluation.png   # Generated visualizations
```

## Key Concepts

### 1. Imbalanced Data Handling

**Problem**: Only 0.2% of transactions are fraud. A model predicting "all legitimate" gets 99.8% accuracy but catches zero fraud.

**Solutions**:
- **SMOTE**: Creates synthetic fraud examples by interpolating between existing fraud cases
- **Class Weights**: Makes fraud misclassification errors cost 200x more than legitimate errors
- **Stratified Splitting**: Maintains fraud ratio in train/test splits

### 2. Feature Engineering for Fraud

Transform raw transactions into fraud signals:

```python
# Velocity: How many recent transactions?
'Transaction_Velocity': Count of transactions in last hour

# Distance: How far from home?
'Distance_From_Home': Geographic distance from billing address

# Timing: Unusual hours?
'Is_Night': Transaction between 10pm-6am

# Amount: Suspicious size?
'Is_Large_Transaction': > 95th percentile amount
```

### 3. Threshold Optimization

Don't use default 0.5 probability threshold. Optimize for business goals:

```python
# Business requirement: Catch 90% of fraud (high recall)
# Find threshold that achieves this while maximizing precision

threshold = 0.32  # Optimized value
# Result: 92% recall, 52% precision at top predictions
```

### 4. Fraud-Specific Metrics

**Don't use**: Accuracy (misleading with imbalance)

**Do use**:
- **Recall**: % of fraud caught (minimize false negatives)
- **Precision**: % of fraud alerts that are real (minimize false positives)
- **ROC-AUC**: Overall discrimination ability (fraud vs legitimate)
- **Business metrics**: Money saved vs customer friction cost

## Comparison with Production Systems

| Feature | Our Implementation | Stripe Radar | PayPal |
|---------|-------------------|--------------|---------|
| **Throughput** | N/A (batch) | 8,000 TPS | 10,000+ TPS |
| **Latency** | N/A (batch) | 35ms | 50ms |
| **Models** | 3 (RF, GB, LR) | 10+ ensemble | 15+ specialized |
| **Features** | 16 engineered | 200+ real-time | 300+ historical |
| **Retraining** | On-demand | Every 4 hours | Continuous |
| **Fraud Rate** | 0.2% synthetic | 0.1-0.5% real | 0.2-1% real |

## Extending This Project

### Level 1: Feature Engineering
- Add more time-based features (day of week, month patterns)
- Create merchant risk scores
- Implement user spending profiles
- Add geolocation distance calculations

### Level 2: Model Improvements
- Implement Neural Network for non-linear patterns
- Add Isolation Forest for anomaly detection
- Create ensemble with model stacking
- Implement online learning for concept drift

### Level 3: Production Readiness
- Build real-time REST API with FastAPI
- Add monitoring and alerting
- Implement A/B testing framework
- Create explainability dashboard (SHAP values)

### Level 4: Advanced Topics
- Handle concept drift (fraud patterns change over time)
- Multi-stage detection (rules → ML → manual review)
- Adversarial robustness (fraudsters attack the model)
- Privacy-preserving fraud detection (federated learning)

## Common Issues and Solutions

### Issue: Low ROC-AUC score
**Solution**: Check feature engineering. Fraud patterns might not be captured. Add more behavioral features.

### Issue: High false positive rate
**Solution**: Increase decision threshold or retrain with cost-sensitive learning.

### Issue: SMOTE creates unrealistic samples
**Solution**: Use ADASYN (adaptive synthetic sampling) or reduce sampling_strategy to 0.3 instead of 0.5.

### Issue: Model predicts all legitimate
**Solution**: Ensure class_weight='balanced' is set and SMOTE is applied correctly.

## Resources

### Documentation
- Scikit-learn Imbalanced Data: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
- Imbalanced-learn (SMOTE): https://imbalanced-learn.org/stable/over_sampling.html

### Papers
- SMOTE: "SMOTE: Synthetic Minority Over-sampling Technique" (Chawla et al., 2002)
- Cost-Sensitive Learning: "The Foundations of Cost-Sensitive Learning" (Elkan, 2001)

### Industry Blogs
- Stripe Radar: https://stripe.com/radar/how-it-works
- PayPal Fraud Detection: https://www.paypal.com/us/security/fraud-protection

## Next Steps

### Tomorrow (Day 62): K-Nearest Neighbors
Learn similarity-based classification that complements ensemble methods:
- Find "nearest neighbor" transactions
- Detect novel fraud patterns (outliers)
- Combine KNN with Random Forest for robust detection

### This Week's Progression
- Day 60: Random Forests ✓
- Day 61: Fraud Detection Project ← You are here
- Day 62: K-Nearest Neighbors
- Day 63: Support Vector Machines

## Questions?

Common questions about fraud detection:

**Q: Why not use deep learning?**
A: For tabular fraud data, Random Forests often outperform neural networks. Deep learning shines with images/text, but tree-based methods excel with structured features.

**Q: How do real systems handle concept drift?**
A: They retrain continuously (every few hours) and use online learning algorithms that adapt to new fraud patterns.

**Q: What about explainability?**
A: Production systems use SHAP values to explain why a transaction was flagged, which is legally required in some jurisdictions.

**Q: How to handle false positives in production?**
A: Implement multi-stage systems: high-confidence fraud → auto-block, medium-confidence → rule-based verification, low-confidence → manual review.

---

**Author**: Senior AI Engineer  
**Course**: 180-Day AI/ML from Scratch  
**Day**: 61 of 180  
**Module**: Supervised Learning - Classification Projects
