"""
Day 33: Introduction to Pandas
The Data Wrangling Powerhouse Behind Every AI System

This lesson demonstrates how data scientists at companies like Netflix, Spotify,
and Uber use Pandas daily to prepare data for machine learning models.
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("Day 33: Introduction to Pandas")
    print("=" * 60)
    
    # ==========================================================
    # PART 1: Series - The Labeled Array
    # ==========================================================
    print("\n📊 PART 1: Pandas Series")
    print("-" * 40)
    
    # Create a Series with meaningful labels
    # Real use case: GPU utilization readings from different machines
    gpu_utilization = pd.Series(
        [78.5, 92.3, 65.8, 88.1, 71.4],
        index=['gpu_node_1', 'gpu_node_2', 'gpu_node_3', 'gpu_node_4', 'gpu_node_5'],
        name='utilization_percent'
    )
    
    print("GPU Utilization by Node:")
    print(gpu_utilization)
    
    # Access by label - much clearer than positional indexing
    print(f"\nNode 2 utilization: {gpu_utilization['gpu_node_2']}%")
    
    # Vectorized operations work just like NumPy
    print(f"Average utilization: {gpu_utilization.mean():.1f}%")
    print(f"Nodes above 80%: {(gpu_utilization > 80).sum()}")
    
    # Series arithmetic with automatic alignment
    baseline = pd.Series(
        [70, 70, 70, 70, 70],
        index=['gpu_node_1', 'gpu_node_2', 'gpu_node_3', 'gpu_node_4', 'gpu_node_5']
    )
    deviation = gpu_utilization - baseline
    print(f"\nDeviation from 70% baseline:")
    print(deviation)
    
    # ==========================================================
    # PART 2: DataFrames - The Heart of Data Science
    # ==========================================================
    print("\n\n📋 PART 2: Pandas DataFrames")
    print("-" * 40)
    
    # Create a DataFrame from dictionary
    # Real use case: Model training configurations for experiment tracking
    experiments = pd.DataFrame({
        'model_type': ['transformer', 'lstm', 'cnn', 'transformer'],
        'parameters_millions': [125, 45, 12, 350],
        'training_hours': [24, 8, 3, 72],
        'final_accuracy': [0.89, 0.82, 0.78, 0.92]
    }, index=['exp_001', 'exp_002', 'exp_003', 'exp_004'])
    
    print("Experiment Tracking DataFrame:")
    print(experiments)
    
    # Access columns (returns Series)
    print(f"\nModel types: {list(experiments['model_type'])}")
    
    # Access rows by label using .loc
    print(f"\nExperiment 001 details:")
    print(experiments.loc['exp_001'])
    
    # Multiple columns
    print("\nAccuracy vs Training Time:")
    print(experiments[['training_hours', 'final_accuracy']])
    
    # ==========================================================
    # PART 3: Loading Real Data
    # ==========================================================
    print("\n\n📂 PART 3: Loading and Exploring Data")
    print("-" * 40)
    
    # Load the CSV file
    df = pd.read_csv('ml_training_metrics.csv')
    
    # The holy trinity of data exploration
    print("First 5 rows (head):")
    print(df.head())
    
    print("\n\nDataset Info:")
    print(df.info())
    
    print("\n\nStatistical Summary:")
    print(df.describe())
    
    # ==========================================================
    # PART 4: Data Quality Checks
    # ==========================================================
    print("\n\n🔍 PART 4: Data Quality Assessment")
    print("-" * 40)
    
    # Check for missing values - critical before ML training
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")
    
    # If there are missing values, show which rows
    if df.isnull().any().any():
        print("\nRows with missing data:")
        print(df[df.isnull().any(axis=1)])
    
    # Check data types
    print("\nData types:")
    print(df.dtypes)
    
    # Unique values for categorical-like columns
    print(f"\nUnique models: {df['model_id'].unique()}")
    print(f"Number of models: {df['model_id'].nunique()}")
    
    # ==========================================================
    # PART 5: Basic Operations & Feature Engineering
    # ==========================================================
    print("\n\n⚙️ PART 5: Basic Operations")
    print("-" * 40)
    
    # Column statistics
    print(f"Average accuracy: {df['accuracy'].mean():.3f}")
    print(f"Max accuracy: {df['accuracy'].max():.3f}")
    print(f"Min loss: {df['loss'].min():.3f}")
    
    # Feature engineering - create new columns
    # This is a key ML preprocessing skill
    
    # Efficiency metric: accuracy per second of training
    df['efficiency'] = df['accuracy'] / df['training_time_sec']
    
    # Memory efficiency: accuracy per GB of memory
    df['memory_efficiency'] = df['accuracy'] / (df['gpu_memory_mb'] / 1024)
    
    # Improvement from previous epoch (for each model)
    df['accuracy_gain'] = df.groupby('model_id')['accuracy'].diff()
    
    print("\nDataFrame with engineered features:")
    print(df[['model_id', 'epoch', 'accuracy', 'efficiency', 'memory_efficiency', 'accuracy_gain']].head(10))
    
    # ==========================================================
    # PART 6: Aggregation Preview
    # ==========================================================
    print("\n\n📈 PART 6: Aggregation Preview")
    print("-" * 40)
    
    # Group by model and get final metrics
    final_results = df.groupby('model_id').agg({
        'accuracy': 'max',
        'loss': 'min',
        'training_time_sec': 'sum'
    }).round(3)
    
    print("Best results per model:")
    print(final_results)
    
    # Find best performing model
    best_model = final_results['accuracy'].idxmax()
    best_accuracy = final_results['accuracy'].max()
    print(f"\n🏆 Best model: {best_model} with accuracy {best_accuracy}")
    
    # ==========================================================
    # SUMMARY
    # ==========================================================
    print("\n\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. Series = labeled 1D array (like a column with row names)
    2. DataFrame = table of Series (like a spreadsheet)
    3. head(), info(), describe() = your first 3 commands
    4. isnull().sum() = always check for missing data
    5. Feature engineering = create new columns from existing ones
    
    These operations scale from 100 to 100 million rows!
    """)

if __name__ == "__main__":
    main()
