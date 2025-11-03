#!/usr/bin/env python3
"""
Real-time Dashboard for Day 27: Measures of Spread Metrics
Run with: python dashboard.py
"""

import numpy as np
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, Response, stream_with_context
from lesson_code import DataQualityChecker

app = Flask(__name__)

# Global state for metrics
metrics_data = {
    'current_dataset': None,
    'datasets': [],
    'last_update': None,
    'is_running': False
}

# Base data for API Response Times to maintain consistency with incremental variations
base_api_times = np.array([
    102, 98, 105, 101, 99, 103, 97, 104, 100, 102,
    101, 99, 103, 98, 105, 250, 102, 99, 101, 100
])

def generate_test_data():
    """Generate continuous test data for real-time updates with smooth variations"""
    # Use time-based seed for other datasets
    current_time = int(time.time() * 2)  # Multiply by 2 for faster changes
    
    # For API Response Times, add small random variations to base data
    np.random.seed(current_time)
    api_variation = np.random.normal(0, 1, len(base_api_times))
    api_times = base_api_times + api_variation
    # Ensure no negative values and keep outlier at similar position
    api_times = np.clip(api_times, 95, None)
    api_times[15] = 240 + np.random.normal(0, 5)  # Keep outlier behavior
    
    np.random.seed(current_time)
    
    datasets = {
        'API Response Times (ms)': api_times,
        'Consistent Model Predictions': np.random.normal(100, 5, 100),
        'Typical User Behavior': np.random.normal(100, 20, 100),
        'Noisy Sensor Readings': np.random.normal(100, 50, 100)
    }
    
    return datasets

def calculate_metrics(dataset_name, data):
    """Calculate all metrics for a dataset"""
    checker = DataQualityChecker(data, dataset_name)
    outliers, outlier_count, outlier_idx = checker.detect_outliers()
    cv = checker.coefficient_of_variation()
    iqr, q25, q75 = checker.calculate_iqr()
    
    # Determine ML readiness
    if cv < 15:
        status = "EXCELLENT"
        status_icon = "✅"
        advice = "Low variance, consistent data. Great for model training!"
    elif cv < 30:
        status = "MODERATE"
        status_icon = "⚠️"
        advice = "Some spread present. Consider feature scaling."
    else:
        status = "HIGH_VARIANCE"
        status_icon = "❌"
        advice = "High variability detected. Normalization or log transform recommended."
    
    return {
        'dataset_name': dataset_name,
        'sample_size': len(data),
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(checker.mean),
        'median': float(np.median(data)),
        'sample_variance': float(checker.variance),
        'sample_std': float(checker.std),
        'population_variance': float(checker.population_var),
        'population_std': float(checker.population_std),
        'iqr': float(iqr),
        'q25': float(q25),
        'q75': float(q75),
        'cv': float(cv),
        'outlier_count': int(outlier_count),
        'outlier_percentage': float(outlier_count / len(data) * 100),
        'outliers': [float(x) for x in outliers] if len(outliers) > 0 else [],
        'ml_readiness': status,
        'ml_readiness_icon': status_icon,
        'recommendation': advice,
        'timestamp': datetime.now().isoformat()
    }

def update_metrics_thread():
    """Background thread to continuously update metrics"""
    global metrics_data
    
    while metrics_data['is_running']:
        try:
            datasets = generate_test_data()
            all_metrics = []
            
            for name, data in datasets.items():
                metrics = calculate_metrics(name, data)
                all_metrics.append(metrics)
            
            metrics_data['datasets'] = all_metrics
            metrics_data['last_update'] = datetime.now().isoformat()
            
            # Simulate new data every 5 seconds for comfortable viewing
            time.sleep(5)
        except Exception as e:
            print(f"Error updating metrics: {e}")
            time.sleep(1)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics (REST API)"""
    if not metrics_data['datasets']:
        # Initialize with default data
        datasets = generate_test_data()
        all_metrics = []
        for name, data in datasets.items():
            metrics = calculate_metrics(name, data)
            all_metrics.append(metrics)
        metrics_data['datasets'] = all_metrics
        metrics_data['last_update'] = datetime.now().isoformat()
    
    return jsonify({
        'datasets': metrics_data['datasets'],
        'last_update': metrics_data['last_update'],
        'total_datasets': len(metrics_data['datasets'])
    })

@app.route('/api/metrics/stream')
def stream_metrics():
    """Server-Sent Events stream for real-time updates"""
    def event_stream():
        while True:
            if metrics_data['datasets']:
                data = {
                    'datasets': metrics_data['datasets'],
                    'last_update': metrics_data['last_update'],
                    'timestamp': datetime.now().isoformat()
                }
                yield f"data: {json.dumps(data)}\n\n"
            time.sleep(3)  # Update every 3 seconds for comfortable, slow updates
    
    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/start', methods=['POST'])
def start_demo():
    """Start the real-time demo"""
    global metrics_data
    
    if not metrics_data['is_running']:
        metrics_data['is_running'] = True
        thread = threading.Thread(target=update_metrics_thread, daemon=True)
        thread.start()
        return jsonify({'status': 'started', 'message': 'Demo started successfully'})
    
    return jsonify({'status': 'already_running', 'message': 'Demo is already running'})

@app.route('/api/stop', methods=['POST'])
def stop_demo():
    """Stop the real-time demo"""
    global metrics_data
    metrics_data['is_running'] = False
    return jsonify({'status': 'stopped', 'message': 'Demo stopped'})

@app.route('/api/status')
def get_status():
    """Get current demo status"""
    return jsonify({
        'is_running': metrics_data['is_running'],
        'last_update': metrics_data['last_update'],
        'dataset_count': len(metrics_data['datasets'])
    })

if __name__ == '__main__':
    # Initialize with default data
    datasets = generate_test_data()
    for name, data in datasets.items():
        metrics = calculate_metrics(name, data)
        metrics_data['datasets'].append(metrics)
    metrics_data['last_update'] = datetime.now().isoformat()
    
    print("🚀 Starting Dashboard Server...")
    print("📊 Access the dashboard at: http://localhost:5000")
    print("🔄 Metrics update in real-time every 3 seconds")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

