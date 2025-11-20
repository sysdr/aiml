"""
Flask application for ML Training Metrics Dashboard
"""
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

def load_and_process_data():
    """Load and process the ML training metrics data"""
    df = pd.read_csv('ml_training_metrics.csv')
    
    # Feature engineering (same as lesson_code.py)
    df['efficiency'] = df['accuracy'] / df['training_time_sec']
    df['memory_efficiency'] = df['accuracy'] / (df['gpu_memory_mb'] / 1024)
    df['accuracy_gain'] = df.groupby('model_id')['accuracy'].diff()
    
    return df


def sanitize_records(df: pd.DataFrame):
    """
    Convert DataFrame values to JSON-serializable Python objects.
    Replaces NaN/Inf with None so Flask's jsonify can handle the payload.
    """
    safe_df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    return safe_df.to_dict(orient='records')

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """API endpoint to get all data"""
    df = load_and_process_data()
    return jsonify(sanitize_records(df))

@app.route('/api/summary')
def get_summary():
    """API endpoint to get data summary statistics"""
    df = load_and_process_data()
    
    summary = {
        'total_records': len(df),
        'total_models': df['model_id'].nunique(),
        'avg_accuracy': float(df['accuracy'].mean()),
        'max_accuracy': float(df['accuracy'].max()),
        'min_loss': float(df['loss'].min()),
        'missing_values': int(df.isnull().sum().sum()),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    return jsonify(summary)

@app.route('/api/model_stats')
def get_model_stats():
    """API endpoint to get statistics per model"""
    df = load_and_process_data()
    
    model_stats = df.groupby('model_id').agg({
        'accuracy': ['max', 'mean'],
        'loss': ['min', 'mean'],
        'training_time_sec': 'sum',
        'epoch': 'max'
    }).round(3)
    
    model_stats.columns = ['max_accuracy', 'avg_accuracy', 'min_loss', 'avg_loss', 'total_training_time', 'total_epochs']
    model_stats = model_stats.reset_index()
    
    return jsonify(sanitize_records(model_stats))

@app.route('/api/training_progress')
def get_training_progress():
    """API endpoint to get training progress over epochs"""
    df = load_and_process_data()
    
    progress = df[['model_id', 'epoch', 'accuracy', 'loss', 'training_time_sec']]
    
    return jsonify(sanitize_records(progress))

@app.route('/api/missing_data')
def get_missing_data():
    """API endpoint to get missing data information"""
    df = load_and_process_data()
    
    missing = df.isnull().sum()
    missing_data = {
        'columns_with_missing': missing[missing > 0].to_dict(),
        'total_missing': int(missing.sum()),
        'rows_with_missing': sanitize_records(df[df.isnull().any(axis=1)]) if df.isnull().any().any() else []
    }
    
    return jsonify(missing_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

