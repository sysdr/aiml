# Day 27: Measures of Spread - Variance and Standard Deviation

Learn how to measure and understand data variability for AI/ML applications.

## 📚 What You'll Learn

- Calculate variance and standard deviation from scratch
- Detect outliers using the 3-sigma rule
- Assess data quality for machine learning
- Understand when to use sample vs population variance
- Apply feature scaling based on variance

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Run the Lesson

```bash
python lesson_code.py
```

This will:
- Analyze real-world response time data
- Compare datasets with different spreads
- Demonstrate ML feature scaling
- Generate visualization (`data_spread_analysis.png`)

### 3. Run Tests

```bash
pytest test_lesson.py -v
```

## 📊 Key Concepts Covered

### Variance
- **Formula**: σ² = Σ(x - μ)² / (n-1)
- Measures average squared deviation from mean
- Critical for understanding data consistency

### Standard Deviation
- **Formula**: σ = √variance
- Same units as original data (easier to interpret)
- Used in 68-95-99.7 rule for normal distributions

### Coefficient of Variation
- **Formula**: CV = (σ / μ) × 100%
- Relative measure of spread
- Helps compare variability across different scales

## 🎯 Real-World Applications

1. **Anomaly Detection**: Flag data points beyond 3σ
2. **Feature Scaling**: Normalize features with different variances
3. **Model Confidence**: Wider spread = less confident predictions
4. **A/B Testing**: Determine if differences are significant

## 📈 Output Examples

```
📊 DATA QUALITY REPORT: API Response Times (ms)
============================================================
Sample Size: 20
Range: [97.00, 250.00]

Central Tendency:
  Mean: 109.70
  Median: 101.00

Measures of Spread:
  Sample Variance: 1139.27
  Sample Std Dev: 33.75
  IQR: 4.50 (Q1=99.00, Q3=103.50)
  Coefficient of Variation: 30.77%

Outlier Detection (3σ rule):
  Outliers Found: 1 (5.0%)
  Outlier Values: [250.]

============================================================
ML Readiness: ⚠️  MODERATE
Recommendation: Some spread present. Consider feature scaling.
```

## 🔧 File Structure

```
day-27-variance-std/
├── setup.sh              # Environment setup
├── requirements.txt      # Python dependencies
├── lesson_code.py        # Main lesson implementation
├── test_lesson.py        # Unit tests
├── README.md            # This file
└── data_spread_analysis.png  # Generated visualization
```

## 💡 Practice Exercises

After completing the lesson, try these:

1. **Modify the threshold**: Change outlier detection from 3σ to 2σ
2. **Add your data**: Replace response_times with your own dataset
3. **Compare distributions**: Create datasets with CV of 10%, 25%, and 50%
4. **Build a monitor**: Track variance over time for drift detection

## 🎓 Learning Tips

- Focus on **why variance matters** for ML, not just calculations
- Practice interpreting CV values for different use cases
- Understand the difference between sample and population variance
- Connect today's concepts to tomorrow's correlation lesson

## 📚 Next Lesson

**Day 28: Correlation and Covariance**
- How variables move together
- Correlation vs causation
- Covariance matrices for ML

## ❓ Common Questions

**Q: When should I use sample vs population variance?**
A: Almost always use sample variance (n-1) in ML since you're working with training data, not all possible data.

**Q: What's a "good" coefficient of variation?**
A: < 15% is excellent, 15-30% is moderate, > 30% suggests you need normalization.

**Q: Why square the deviations in variance?**
A: To prevent positive and negative deviations from canceling out and to penalize larger deviations more.

## 🐛 Troubleshooting

**Import Error**: Make sure you've activated the virtual environment
```bash
source venv/bin/activate
```

**Missing Visualization**: Install matplotlib
```bash
pip install matplotlib
```

**Tests Failing**: Update NumPy to latest version
```bash
pip install --upgrade numpy
```

## 📞 Support

Having issues? Check:
- Python version is 3.11+
- All dependencies installed from requirements.txt
- Virtual environment is activated

---

**Time to Complete**: 2-3 hours
**Difficulty**: Beginner to Intermediate
**Prerequisites**: Day 26 (Mean, Median, Mode)
