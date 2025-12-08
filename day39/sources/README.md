# Day 39: Supervised vs. Unsupervised Learning

Learn the fundamental difference between AI's two main approaches: learning with a teacher (supervised) vs. discovering patterns independently (unsupervised).

## Quick Start

### 1. Setup Environment
```bash
chmod +x setup_venv.sh
./setup_venv.sh
source venv/bin/activate
```

### 2. Run the Lesson
```bash
python lesson_code.py
```

### 3. Run Tests
```bash
pytest test_lesson.py -v
```

## What You'll Learn

- **Supervised Learning**: Train models with labeled data (spam classifier)
- **Unsupervised Learning**: Discover patterns in unlabeled data (customer segmentation)
- **Decision Framework**: When to use each approach in production AI systems
- **Real-World Examples**: How Netflix, Tesla, Google, and Spotify use both

## Key Insights

### The API Difference
```python
# Supervised Learning - requires labels (y)
model.fit(X, y)  

# Unsupervised Learning - no labels needed
model.fit(X)
```

### When to Use Each

**Use Supervised Learning When:**
- You have labeled training data
- You need precise predictions
- You can define "correct" answers
- Examples: spam detection, fraud detection, medical diagnosis

**Use Unsupervised Learning When:**
- You have abundant unlabeled data
- You want to discover hidden patterns
- No pre-defined categories exist
- Examples: customer segmentation, anomaly detection, recommendation systems

## Files Generated

- `lesson_code.py` - Complete implementation with both approaches
- `test_lesson.py` - Comprehensive test suite
- `customer_segments.png` - Visualization of unsupervised learning results
- `requirements.txt` - All dependencies
- `README.md` - This file

## Real-World Connections

### Netflix
- **Supervised**: Predict your rating for new shows based on past ratings
- **Unsupervised**: Group similar movies for recommendations

### Tesla
- **Supervised**: Detect objects (cars, pedestrians, signs) from labeled footage
- **Unsupervised**: Discover rare driving scenarios in unlabeled data

### Gmail
- **Supervised**: Classify known spam patterns from labeled emails
- **Unsupervised**: Detect emerging spam tactics in new data

## Expected Output

The lesson demonstrates:
1. Training a spam classifier with 80% accuracy on labeled data
2. Discovering 3 customer segments from unlabeled purchase data
3. Visual comparison showing cluster separation
4. Decision framework for choosing the right approach

## Next Steps

**Day 40: Regression vs. Classification**
- Dive deeper into supervised learning
- Learn when to predict numbers vs. categories
- Build both types of systems

## Time Investment

- Setup: 5 minutes
- Main lesson: 30-45 minutes
- Experiments: 30-60 minutes
- Total: 2-3 hours

## Success Criteria

✓ Understand supervised vs. unsupervised at a conceptual level  
✓ Know which approach to use for different problems  
✓ Can implement both using scikit-learn  
✓ Recognize how production systems combine both approaches  

## Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt --upgrade
```

**Tests failing?**
```bash
pytest test_lesson.py -v --tb=short
```

**Visualization not appearing?**
- Check that matplotlib is installed
- File saves as 'customer_segments.png' in current directory

## Questions to Consider

Before tomorrow's lesson, think about:
1. Which ML problems at your favorite apps use supervised learning?
2. Which use unsupervised learning?
3. Can you identify problems that use both?

---

**Course**: 180-Day AI/ML from Scratch  
**Module**: Week 7 - Core Concepts  
**Previous**: Day 38 - The Machine Learning Workflow  
**Next**: Day 40 - Regression vs. Classification
