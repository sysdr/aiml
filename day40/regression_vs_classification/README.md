# Day 40: Regression vs. Classification

## Overview
Learn the fundamental difference between predicting continuous values (regression) and discrete categories (classification) by building a dual-purpose house price system.

## What You'll Build
1. **Regression Model**: Predicts exact house prices ($347,500)
2. **Classification Model**: Predicts price tiers (Budget, Mid-Range, Luxury, Ultra-Luxury)
3. **Comparison Dashboard**: Side-by-side evaluation of both approaches

## Real-World Applications
- **Netflix**: Predicting star ratings (regression) + content categorization (classification)
- **Tesla**: Steering angle prediction (regression) + object detection (classification)
- **Google**: Click-through rate (regression) + query type classification

## Quick Start

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run the Lesson
```bash
python lesson_code.py
```

Expected output:
- Training metrics for both models
- Test set evaluation
- Sample predictions
- Comparison visualization

### 3. Run Tests
```bash
pytest test_lesson.py -v
```

## Key Concepts

### Regression
- **Output**: Continuous numbers (any value in a range)
- **Example**: $347,250.83
- **Metrics**: MAE, RMSE, R²
- **Use case**: When you need the exact number

### Classification
- **Output**: Discrete categories (fixed set of options)
- **Example**: "Mid-Range"
- **Metrics**: Accuracy, Precision, Recall
- **Use case**: When you need the category

### The Fundamental Rule
> Look at the output layer to identify the problem type:
> - Linear/no activation = Regression
> - Softmax activation = Classification

## Files Created
- `lesson_code.py` - Main implementation
- `test_lesson.py` - Comprehensive tests
- `requirements.txt` - Dependencies
- `regression_vs_classification_results.png` - Visualization

## Learning Objectives
After completing this lesson, you'll:
1. ✅ Understand when to use regression vs classification
2. ✅ Know which metrics to use for each problem type
3. ✅ Build and evaluate both model types
4. ✅ Recognize problem types in real AI systems

## Next Steps
**Day 41**: Overfitting and Underfitting
- Learn why models fail on new data
- Detect overfitting in regression vs classification
- Apply production-grade regularization techniques

## Troubleshooting

### Import Error
```bash
pip install -r requirements.txt
```

### Virtual Environment Not Activated
```bash
source venv/bin/activate  # Unix/Mac
venv\Scripts\activate     # Windows
```

### Tests Failing
Ensure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

## Time Estimate
- Setup: 5 minutes
- Code execution: 3 minutes
- Understanding output: 10 minutes
- Experimentation: 15 minutes
- **Total: ~35 minutes**

## Success Criteria
- ✅ Both models train without errors
- ✅ Regression MAE < $50,000
- ✅ Classification accuracy > 70%
- ✅ All tests pass
- ✅ Understand output differences

---

**Course**: 180-Day AI and Machine Learning from Scratch  
**Module**: Core Concepts (Week 7)  
**Day**: 40 of 180
