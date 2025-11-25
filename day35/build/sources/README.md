# Day 35: Data Cleaning and Handling Missing Data

Production-ready data cleaning toolkit that handles real-world messy data the same way Netflix, Google, and Tesla clean billions of events daily.

## What You'll Learn

- Detect and analyze missing data patterns in DataFrames
- Apply professional imputation strategies (mean, median, mode, forward fill, KNN)
- Build production-grade cleaning pipelines with proper validation
- Understand why data scientists spend 80% of their time on data quality

## Quick Start

```bash
# 1. Setup environment
chmod +x setup.sh
./setup.sh
source venv/bin/activate

# 2. Run the main lesson
python lesson_code.py

# 3. Run tests to verify understanding
pytest test_lesson.py -v
```

## What's Included

- **lesson_code.py** - Complete data cleaning library with:
  - MissingDataDetector - Analyzes missing data patterns
  - DataCleaner - Production cleaning pipeline
  - Multiple imputation strategies
  - Validation suite
  
- **test_lesson.py** - Comprehensive test suite verifying:
  - All imputation strategies work correctly
  - Pipeline handles edge cases
  - Data types are preserved
  - Chaining operations works
  
- **requirements.txt** - All dependencies with exact versions

## Key Features

### Missing Data Detection
```python
detector = MissingDataDetector(df)
report = detector.generate_report()
# Shows which columns have missing data and recommends strategies
```

### Production Cleaning Pipeline
```python
cleaner = DataCleaner(df)
cleaned_df = (cleaner
              .drop_high_missing_columns(threshold=0.7)
              .fill_numeric_median(['age', 'income'])
              .fill_categorical_mode(['country'])
              .get_cleaned_data())
```

### Comprehensive Validation
```python
validation = cleaner.validate_cleaning()
# Tracks all changes, verifies data integrity
```

## Real-World Applications

**Netflix**: Processes 200B+ events daily with 30-40% missing optional fields. Uses forward fill for time series, KNN for demographics.

**Tesla Autopilot**: Handles 30% corrupted sensor readings. Forward fill for missing frames, median imputation for outlier sensors.

**Google Ads**: Billions of events with 40-50% partial data. Different strategies per feature importance.

## Learning Outcomes

After this lesson you can:
- ✅ Identify missing data types (MCAR, MAR, MNAR)
- ✅ Choose appropriate imputation strategies
- ✅ Build production cleaning pipelines
- ✅ Validate data quality with confidence
- ✅ Understand why cleaning matters more than algorithms

## Next Steps

Tomorrow: **Project Day - Exploratory Data Analysis (EDA)**

You'll apply everything learned this week:
- DataFrames manipulation
- Data cleaning techniques
- Missing data handling
- Generate professional analysis with visualizations

## Common Issues

**Issue**: Tests failing with "method 'ffill' not recognized"
**Fix**: Upgrade pandas: `pip install --upgrade pandas`

**Issue**: Mean/median imputation creates infinite values
**Fix**: Check for outliers first, use median instead of mean

## Performance Notes

The cleaning pipeline is optimized for datasets up to 10M rows. For larger data:
- Use Dask for parallel processing
- Implement column-wise processing
- Consider sampling for strategy selection

## Additional Resources

- Pandas documentation on missing data
- sklearn.impute for advanced strategies
- Production data quality monitoring tools

---

**Time to complete**: 2-3 hours
**Difficulty**: Intermediate
**Prerequisites**: Day 34 (DataFrames operations)
