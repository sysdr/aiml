# Day 34: DataFrame Indexing, Slicing, and Filtering

## Overview

Learn the DataFrame operations that power every production AI system—from Netflix recommendations to Tesla's self-driving data pipelines.

## What You'll Build

A smart content recommendation filter that demonstrates real-world data selection techniques used by major tech companies.

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
- Demonstrations of indexing techniques (.loc, .iloc, .at)
- Data slicing examples
- Boolean filtering patterns
- Production recommendation filter
- Data quality analysis
- Performance comparisons
- Visualization saved as `filtering_analysis.png`

### 3. Run Tests

```bash
pytest test_lesson.py -v
```

All tests should pass, demonstrating mastery of:
- Label-based indexing
- Position-based indexing
- Fast scalar access
- Column/row slicing
- Single and multiple condition filtering
- Combined operations

## Key Learning Objectives

1. **Indexing**: Direct access to data using labels or positions
2. **Slicing**: Extracting continuous segments of data
3. **Filtering**: Conditional data selection with boolean logic
4. **Performance**: Understanding when to use each technique
5. **Real-world**: Building production-grade data filters

## Core Concepts Covered

### Indexing Methods

- `.loc[]` - Label-based indexing (most readable)
- `.iloc[]` - Position-based indexing (most flexible)
- `.at[]` - Fast scalar access (best performance)
- `.iat[]` - Fast position-based scalar access

### Slicing Patterns

- Column selection: `df['column']` or `df[['col1', 'col2']]`
- Row ranges: `df[start:end]` or `df.iloc[start:end]`
- Combined: `df.loc[rows, columns]`

### Filtering Techniques

- Single condition: `df[df['column'] > value]`
- Multiple AND: `df[(condition1) & (condition2)]`
- Multiple OR: `df[(condition1) | (condition2)]`
- Complex logic: Combine multiple operators

## Real-World Applications

This lesson demonstrates operations used in:

- **Content recommendation** (Netflix, YouTube, Spotify)
- **User segmentation** (Meta, Google Ads)
- **Data quality control** (all ML pipelines)
- **Feature selection** (model training)
- **A/B testing** (experiment analysis)

## Files Generated

- `lesson_code.py` - Main implementation with ContentRecommendationEngine
- `test_lesson.py` - Comprehensive test suite
- `filtering_analysis.png` - Visual analysis of filtering impact
- `requirements.txt` - Python dependencies
- `setup.sh` - Environment setup script

## Performance Notes

The lesson includes performance comparisons showing:
- `.at[]` is 5-10x faster than `.loc[]` for scalar access
- Boolean indexing is highly optimized in pandas
- Combined operations can be chained efficiently

## Next Steps

Tomorrow (Day 35): Data Cleaning and Handling Missing Data

You'll learn:
- Detecting and handling missing values
- Removing duplicates
- Handling outliers
- Data type conversions
- Production data cleaning pipelines

## Troubleshooting

**Import errors?**
- Ensure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**Tests failing?**
- Check Python version: `python --version` (should be 3.11+)
- Verify pandas version: `pip show pandas` (should be 2.1.4)

**Visualization not appearing?**
- Check that matplotlib backend is properly configured
- Look for `filtering_analysis.png` in current directory

## Resources

- [Pandas Indexing Documentation](https://pandas.pydata.org/docs/user_guide/indexing.html)
- [Boolean Indexing Guide](https://pandas.pydata.org/docs/user_guide/indexing.html#boolean-indexing)
- Course repository: Next lesson preview

---

**Time to Complete**: 2-3 hours
**Difficulty**: Intermediate
**Prerequisites**: Day 33 (Introduction to Pandas)
