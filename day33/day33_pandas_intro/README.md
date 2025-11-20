# Day 33: Introduction to Pandas

The data wrangling powerhouse behind every AI system.

## Quick Start

```bash
# Setup environment
./setup.sh
source venv/bin/activate

# Run the lesson
python lesson_code.py

# Verify understanding
pytest test_lesson.py -v
```

## What You'll Learn

1. **Series**: Labeled 1D arrays for intuitive data access
2. **DataFrames**: Tables that form the foundation of ML pipelines
3. **Data Exploration**: head(), info(), describe() workflow
4. **Data Quality**: Missing value detection and handling
5. **Feature Engineering**: Creating new columns from existing data

## Files

- `lesson_code.py` - Main lesson with real ML training metrics
- `test_lesson.py` - 20 tests covering all concepts
- `ml_training_metrics.csv` - Sample dataset of model training runs

## Key Commands

```python
# The holy trinity of exploration
df.head()       # First 5 rows
df.info()       # Data types and memory
df.describe()   # Statistical summary

# Data quality
df.isnull().sum()  # Missing values per column

# Feature engineering
df['new_col'] = df['col_a'] / df['col_b']
```

## Real-World Connection

These exact operations are used daily at:
- **Netflix**: Processing viewing history for recommendations
- **Spotify**: Analyzing listening patterns
- **Uber**: Preparing ride data for surge pricing models
- **OpenAI**: Cleaning training datasets

## Next Lesson

Day 34: DataFrames - Indexing, Slicing, and Filtering
