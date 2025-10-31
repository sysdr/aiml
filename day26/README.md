# Day 26: Descriptive Statistics (Mean, Median, Mode)

## Quick Start

```bash
# 1. Run setup (creates virtual environment and installs dependencies)
chmod +x setup.sh
./setup.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run the lesson
python lesson_code.py

# 4. Test your understanding
python test_lesson.py
```

## What You'll Learn

- Calculate mean, median, and mode in Python
- Detect data skewness and outliers
- Understand how these metrics guide AI model selection
- Build a data profiler used in production AI systems

## Files Included

- `lesson_code.py` - Main implementation with real-world examples
- `test_lesson.py` - Comprehensive test suite to verify understanding
- `setup.sh` - Automated environment setup
- `requirements.txt` - Python dependencies

## Learning Objectives

After completing this lesson, you'll be able to:

1. Calculate descriptive statistics on any dataset
2. Interpret what mean vs median differences reveal about data
3. Detect data quality issues before model training
4. Profile datasets like production AI engineers

## Real-World Applications

- **User Behavior Analysis**: Understanding typical vs power user patterns
- **Fraud Detection**: Identifying outlier transactions
- **Model Performance Monitoring**: Tracking latency and accuracy
- **Feature Engineering**: Deciding on data transformations

## Time Estimate

2-3 hours including:
- Reading main article (30 min)
- Running code examples (45 min)
- Completing tests (45 min)
- Experimentation (30 min)

## Next Lesson

Day 27: Measures of Spread (Variance and Standard Deviation)

## Need Help?

Common issues:
- **Import errors**: Run `setup.sh` again
- **Python version**: Requires Python 3.11+
- **Test failures**: Review lesson_code.py comments

## Key Takeaway

Descriptive statistics are your first conversation with data. Before training any AI model, always calculate mean, median, and mode. When they disagree, your data has a story to tell - listen carefully!
