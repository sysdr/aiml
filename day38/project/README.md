# Day 38: The Machine Learning Workflow

A complete implementation of the 7-stage ML workflow used in production AI systems.

## What You'll Learn

- **The 7 stages** of every ML project (problem definition → monitoring)
- **How to build** a complete ML pipeline from scratch
- **Production patterns** used at Netflix, Google, Amazon, and other tech companies
- **Why each stage matters** and what happens if you skip it

## Quick Start

### 1. Setup Environment

```bash
chmod +x setup_env.sh
./setup_env.sh
source ml_workflow_env/bin/activate
```

### 2. Run the Complete Workflow

```bash
python lesson_code.py
```

You'll see all 7 stages execute in real-time:
- ✅ Problem Definition
- ✅ Data Collection
- ✅ Data Preparation
- ✅ Model Training
- ✅ Model Evaluation
- ✅ Deployment
- ✅ Monitoring & Prediction

### 3. Verify Your Understanding

```bash
pytest test_lesson.py -v
```

## The 7-Stage ML Workflow

### Stage 1: Problem Definition
Define exactly what you're predicting and why it matters.

**In Code:**
```python
pipeline.define_problem()
# Output: Binary classification with F1-score > 0.80 target
```

### Stage 2: Data Collection
Gather the raw materials for your AI engine.

**In Code:**
```python
df = pipeline.collect_data()
# Output: DataFrame with reviews and sentiment labels
```

### Stage 3: Data Preparation
Clean, transform, and split data for training.

**In Code:**
```python
X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
# Output: Vectorized features ready for training
```

### Stage 4: Model Training
Let the algorithm discover patterns in your data.

**In Code:**
```python
model = pipeline.train_model(X_train, y_train)
# Output: Trained logistic regression model
```

### Stage 5: Model Evaluation
Measure performance on unseen data.

**In Code:**
```python
metrics = pipeline.evaluate_model(X_test, y_test)
# Output: Accuracy, Precision, Recall, F1-Score
```

### Stage 6: Deployment
Save model for production use.

**In Code:**
```python
model_path = pipeline.deploy_model()
# Output: Saved model files ready to serve predictions
```

### Stage 7: Monitoring
Make predictions and track performance.

**In Code:**
```python
results = pipeline.predict(new_reviews)
# Output: Predictions with confidence scores
```

## Code Structure

```
├── lesson_code.py      # Complete ML workflow pipeline
├── test_lesson.py      # Tests for each workflow stage
├── setup_env.sh       # Environment setup
├── requirements.txt   # Python dependencies
└── models/           # Saved model artifacts (created after running)
    ├── sentiment_model.pkl
    ├── vectorizer.pkl
    └── metrics.json
```

## Real-World Connections

This workflow structure scales from educational examples to production systems:

**Your Implementation:**
- 40 training samples
- Runs in 30 seconds on laptop
- 85%+ accuracy

**Amazon's Production:**
- 500M+ product reviews
- Processes continuously on AWS clusters
- 92%+ accuracy with deep learning

The workflow stages remain **identical**—only scale changes.

## Practice Exercises

### Exercise 1: Modify the Problem
Change the sentiment analysis to spam detection:
1. Update `collect_data()` with spam/ham examples
2. Change labels to "spam" vs "ham"
3. Adjust metrics for recall priority (catch all spam)

### Exercise 2: Improve Performance
Enhance the current model:
1. Add more training data
2. Experiment with `RandomForestClassifier` instead of `LogisticRegression`
3. Tune hyperparameters using `GridSearchCV`

### Exercise 3: Add Monitoring
Implement production monitoring:
1. Log predictions to a file
2. Track prediction confidence distribution
3. Alert when confidence drops below threshold

## Common Issues

**Issue:** Import errors
**Fix:** Make sure virtual environment is activated:
```bash
source ml_workflow_env/bin/activate
```

**Issue:** Tests fail
**Fix:** Run the main code first to ensure model trains:
```bash
python lesson_code.py
pytest test_lesson.py -v
```

## Next Steps

Tomorrow (Day 39): **Supervised vs. Unsupervised Learning**
- When to use supervised learning (labeled data)
- When to use unsupervised learning (no labels)
- How this choice affects your workflow

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [ML Workflow Best Practices](https://ml-ops.org/)
- Production ML: Netflix, Uber, Airbnb tech blogs

---

**Remember:** Every major AI system follows these exact 7 stages. Master the workflow, scale it to any problem.
