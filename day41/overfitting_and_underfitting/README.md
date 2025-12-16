# Day 41: Overfitting and Underfitting

## Overview
Production-grade diagnostic system for detecting overfitting and underfitting in machine learning models. This tool mirrors what runs 24/7 in real ML pipelines at companies like Netflix, Spotify, and Tesla.

## What You'll Learn
- Detect when models are too simple (underfitting) or too complex (overfitting)
- Analyze bias-variance trade-off in real-time
- Generate learning curves to diagnose data needs
- Measure model stability with cross-validation
- Interpret train-test gaps like production engineers

## Prerequisites
- Python 3.11+
- Basic understanding of machine learning concepts
- Familiarity with train/test splits

## Quick Start

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run the Analysis
```bash
python lesson_code.py
```

**Expected Output:**
- Complexity analysis for polynomial degrees 1-15
- Optimal model identification (typically degree 3-4)
- Learning curve convergence analysis
- Cross-validation stability metrics
- Diagnostic visualization saved as PNG

### 3. Run Tests
```bash
pytest test_lesson.py -v
```

**Test Coverage:**
- Underfit detection (degree 1 models)
- Overfit detection (degree 15 models)
- Optimal model identification
- Train-test gap analysis
- Cross-validation variance

## Understanding the Output

### Complexity Analysis Table
```
Degree   Train R²     Test R²      Gap       Status
----------------------------------------------------------------
1        0.2453      0.2134      0.0319    🔴 UNDERFIT
3        0.8712      0.8456      0.0256    🟢 GOOD
15       0.9998      0.3421      0.6577    🔴 OVERFIT
```

**Interpretation:**
- **Degree 1-2**: Underfit (too simple, can't capture pattern)
- **Degree 3-5**: Optimal (captures pattern, generalizes well)
- **Degree 8+**: Overfit (memorizes noise, fails on new data)

### Learning Curves
Shows how performance improves with more training data:
- **Converging curves**: Model benefits from more data
- **Plateau**: Model at capacity, need different architecture
- **Large gap**: Overfitting, need regularization

### Cross-Validation Scores
Measures prediction stability across data subsets:
- **Low variance (<0.05)**: Stable, reliable model
- **High variance (>0.1)**: Unstable, likely overfitting

## Real-World Applications

### Netflix Recommendations
- Monitors complexity of personalization models
- Alerts when train-test gap exceeds threshold
- Auto-adjusts regularization based on CV variance

### Tesla Autopilot
- Tracks lane detection model stability
- Reduces complexity if variance increases
- Ensures consistent predictions across road conditions

### Google Search Ranking
- Runs learning curve analysis on ranking models
- Decides when to collect more training data
- Balances 10,000+ ranking features optimally

## Files Generated
- `lesson_code.py` - Main diagnostic system
- `test_lesson.py` - Comprehensive test suite
- `requirements.txt` - Python dependencies
- `setup.sh` - Environment setup script
- `overfitting_analysis.png` - Diagnostic plots

## Key Concepts

### Bias-Variance Trade-off
```
Total Error = Bias² + Variance + Irreducible Error

Bias: Systematic error (underfitting)
Variance: Sensitivity to training data (overfitting)
Irreducible: Noise in data itself
```

### Production Monitoring Pattern
```
1. Deploy model at conservative complexity
2. Monitor train-test gap in real-time
3. Alert when gap exceeds threshold (typically 10-15%)
4. Auto-trigger regularization or retraining
5. Repeat continuously
```

## Troubleshooting

### Issue: Tests failing
**Solution:** Ensure virtual environment is activated
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Plots not displaying
**Solution:** Install GUI backend or save to file only
```bash
# In lesson_code.py, remove plt.show() to save only
```

### Issue: Import errors
**Solution:** Verify Python version
```bash
python3 --version  # Should be 3.11+
```

## Next Steps
- **Day 42**: Learn proper data splitting (train/test/validation)
- **Day 43**: Implement cross-validation strategies
- **Day 44**: Apply regularization techniques (L1/L2)

## Resources
- Scikit-learn Model Selection: https://scikit-learn.org/stable/model_selection.html
- Bias-Variance Trade-off: https://en.wikipedia.org/wiki/Bias–variance_tradeoff
- Learning Curves Guide: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

---

**Course:** 180-Day AI and Machine Learning from Scratch  
**Module:** Week 7 - Core Concepts  
**Lesson:** Day 41 of 180
