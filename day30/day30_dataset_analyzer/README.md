# Day 30: ML Dataset Analyzer - Project Day

🎉 **Congratulations!** You've reached the final day of Week 4-5: Probability & Statistics for Data Science.

## Overview

This project brings together all the statistical concepts you've learned:
- **Day 26**: Descriptive Statistics & Measures of Central Tendency
- **Day 27**: Measures of Spread & Outlier Detection
- **Day 28**: Correlation & Covariance
- **Day 29**: Central Limit Theorem

You'll build a production-ready **ML Dataset Analyzer** that profiles datasets before they enter machine learning pipelines—the same type of tool used at Google, Meta, and Netflix.

## What You'll Build

A comprehensive dataset analyzer that:
1. ✅ Profiles numeric features with 15+ statistical measures
2. ✅ Detects data quality issues (missing data, outliers, imbalance)
3. ✅ Analyzes feature correlations for multicollinearity
4. ✅ Tests normality assumptions for ML algorithms
5. ✅ Calculates ML readiness score (0-100)
6. ✅ Generates professional HTML reports with visualizations

## Quick Start

### 1. Setup Environment

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run the Analyzer

```bash
python lesson_code.py
```

This will:
- Create sample datasets (clean and messy)
- Analyze all datasets
- Generate visualizations in `plots/` directory
- Create `analysis_report.html` with comprehensive results

### 3. View Results

Open `analysis_report.html` in your browser to see:
- ML readiness score
- Feature statistical profiles
- Data quality issues
- Correlation analysis
- Distribution plots

### 4. Run Tests

```bash
pytest test_lesson.py -v
```

## Project Structure

```
day30_dataset_analyzer/
├── lesson_code.py          # Main analyzer implementation
├── test_lesson.py          # Comprehensive test suite
├── setup.sh                # Environment setup script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── clean_dataset.csv      # Sample clean data (generated)
├── messy_dataset.csv      # Sample messy data (generated)
├── analysis_report.html   # Generated analysis report
└── plots/                 # Generated visualizations
    ├── correlation_heatmap.png
    ├── feature_distributions.png
    └── missing_data.png
```

## Key Features

### 1. Feature Profiling
- Central tendency: mean, median, mode
- Spread: std, variance, IQR, range
- Distribution shape: skewness, kurtosis
- Outlier detection using 1.5×IQR rule
- Missing data analysis

### 2. Quality Issue Detection
- High missing data (>30%)
- Excessive outliers (>10%)
- Zero variance features
- Highly skewed distributions (|skewness| > 2)
- Class imbalance (>90% majority class)

### 3. Correlation Analysis
- Full correlation matrix
- Multicollinearity detection (correlation > 0.8)
- Redundant feature identification

### 4. ML Readiness Score
Comprehensive 0-100 score based on:
- Missing data percentage
- Outlier prevalence
- Feature variance
- Multicollinearity
- Class balance

### 5. Professional Reporting
- HTML reports with interactive visualizations
- Color-coded quality indicators
- Actionable recommendations
- Export-ready format

## How to Use with Your Own Data

```python
import pandas as pd
from lesson_code import MLDatasetAnalyzer

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Create analyzer (specify target column if doing supervised learning)
analyzer = MLDatasetAnalyzer(df, target_column='your_target')

# Run full analysis
analyzer.run_full_analysis(
    generate_viz=True,   # Create plots
    generate_html=True   # Generate report
)

# Access results programmatically
readiness = analyzer.analysis_results['ml_readiness']
print(f"ML Readiness Score: {readiness['score']:.1f}/100")
```

## Real-World Applications

This analyzer implements techniques used by:

**Google (TensorFlow Data Validation)**
- Automatic schema validation
- Distribution monitoring
- Anomaly detection

**Amazon (SageMaker Data Wrangler)**
- Feature profiling
- Quality scoring
- Transformation suggestions

**Uber (Data Quality Platform)**
- Continuous monitoring
- Alert systems
- Pipeline validation

## Learning Objectives Achieved

✅ Applied descriptive statistics to real datasets  
✅ Implemented outlier detection using IQR method  
✅ Calculated and interpreted correlations  
✅ Tested statistical assumptions (normality)  
✅ Built production-ready analysis pipeline  
✅ Generated professional data reports  

## Next Steps

**Tomorrow (Day 31)**: Introduction to NumPy

You'll learn why NumPy is 50-100x faster than regular Python for numerical operations—the foundation of every ML library.

## Common Issues & Solutions

**Issue**: "ModuleNotFoundError: No module named 'pandas'"
**Solution**: Make sure you activated the virtual environment:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Issue**: Plots not showing
**Solution**: Plots are saved to the `plots/` directory. Check there or set `generate_viz=True` in `run_full_analysis()`.

**Issue**: Tests failing
**Solution**: Make sure sample datasets exist:
```bash
python -c "from lesson_code import create_sample_datasets; create_sample_datasets()"
```

## Key Takeaways

1. **Statistics is diagnostic medicine for data**: Just like doctors run tests before treatment, data scientists profile data before ML.

2. **Quality over quantity**: A small, clean dataset beats a large, messy one every time.

3. **Automation saves careers**: Manual data profiling is error-prone. Automated analyzers catch issues 24/7.

4. **Real ML is 80% data work**: The glamorous part (training models) only works if you nail the unglamorous part (data quality).

## Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [SciPy Stats Module](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Google's Rules of ML: Data](https://developers.google.com/machine-learning/guides/rules-of-ml)

---

**You've completed Week 4-5! 🎉**

You now understand the statistical foundations of AI/ML. Tomorrow, we accelerate with NumPy.
