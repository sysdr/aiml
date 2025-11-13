"""
Dashboard for Day 31: NumPy Metrics Display
Shows real-time metrics from NumPy operations
"""

import json
import os
import time
from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
import numpy as np
from lesson_code import ImagePreprocessor

app = Flask(__name__)
CORS(app)

# Global metrics storage
metrics = {
    'images_processed': 0,
    'normalize_time': 0.0,
    'standardize_time': 0.0,
    'feature_extraction_time': 0.0,
    'batch_processing_time': 0.0,
    'total_operations': 0,
    'memory_usage_mb': 0.0,
    'numpy_speedup': 0.0,
    'features_extracted': 0,
    'batches_processed': 0,
    'last_update': time.time()
}

# HTML template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Day 31: NumPy Metrics Dashboard</title>
    <meta http-equiv="refresh" content="2">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .metric-label {
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-unit {
            font-size: 0.6em;
            color: #999;
            margin-left: 5px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active { background-color: #4caf50; }
        .status-inactive { background-color: #f44336; }
        .info-box {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .info-box h2 {
            color: #667eea;
            margin-bottom: 15px;
        }
        .info-box p {
            line-height: 1.6;
            color: #555;
        }
        .zero-warning {
            color: #f44336;
            font-weight: bold;
        }
        .updated-time {
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 NumPy Performance Metrics Dashboard</h1>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">
                    <span class="status-indicator status-{{ 'active' if metrics.images_processed > 0 else 'inactive' }}"></span>
                    Images Processed
                </div>
                <div class="metric-value">
                    {{ metrics.images_processed }}
                    <span class="metric-unit">images</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Normalization Time</div>
                <div class="metric-value">
                    {{ "%.4f"|format(metrics.normalize_time) }}
                    <span class="metric-unit">seconds</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Standardization Time</div>
                <div class="metric-value">
                    {{ "%.4f"|format(metrics.standardize_time) }}
                    <span class="metric-unit">seconds</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Feature Extraction Time</div>
                <div class="metric-value">
                    {{ "%.4f"|format(metrics.feature_extraction_time) }}
                    <span class="metric-unit">seconds</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Batch Processing Time</div>
                <div class="metric-value">
                    {{ "%.4f"|format(metrics.batch_processing_time) }}
                    <span class="metric-unit">seconds</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Memory Usage</div>
                <div class="metric-value">
                    {{ "%.2f"|format(metrics.memory_usage_mb) }}
                    <span class="metric-unit">MB</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">NumPy Speedup</div>
                <div class="metric-value">
                    {{ "%.1f"|format(metrics.numpy_speedup) }}
                    <span class="metric-unit">x faster</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Features Extracted</div>
                <div class="metric-value">
                    {{ metrics.features_extracted }}
                    <span class="metric-unit">features</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Batches Processed</div>
                <div class="metric-value">
                    {{ metrics.batches_processed }}
                    <span class="metric-unit">batches</span>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Total Operations</div>
                <div class="metric-value">
                    {{ metrics.total_operations }}
                    <span class="metric-unit">ops</span>
                </div>
            </div>
        </div>
        
        <div class="info-box">
            <h2>📊 Dashboard Status</h2>
            <p>
                <strong>Status:</strong> 
                <span class="status-indicator status-{{ 'active' if metrics.images_processed > 0 else 'inactive' }}"></span>
                {% if metrics.images_processed > 0 %}
                    <span style="color: #4caf50;">Active - Metrics are being updated</span>
                {% else %}
                    <span class="zero-warning">Inactive - Run demo.py to start generating metrics</span>
                {% endif %}
            </p>
            <p style="margin-top: 10px;">
                <strong>Note:</strong> This dashboard displays real-time metrics from NumPy operations. 
                Run <code>python demo.py</code> to execute the demo and update these metrics.
            </p>
        </div>
        
        <div class="updated-time">
            Last updated: {{ time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metrics.last_update)) }}
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Render the main dashboard page"""
    return render_template_string(DASHBOARD_HTML, metrics=metrics, time=time)

@app.route('/api/metrics')
def get_metrics():
    """API endpoint to get current metrics"""
    return jsonify(metrics)

@app.route('/api/update', methods=['POST'])
def update_metrics():
    """Update metrics from demo execution"""
    global metrics
    # This will be called by demo.py
    return jsonify({'status': 'success'})

def update_metrics_from_stats(stats, num_images, memory_mb, features_count, batches_count):
    """Update global metrics from preprocessing stats"""
    global metrics
    metrics.update({
        'images_processed': num_images,
        'normalize_time': stats.get('normalize_time', 0.0),
        'standardize_time': stats.get('standardize_time', 0.0),
        'feature_extraction_time': stats.get('feature_extraction_time', 0.0),
        'batch_processing_time': stats.get('batch_processing_time', 0.0),
        'memory_usage_mb': memory_mb,
        'numpy_speedup': stats.get('numpy_speedup', 0.0),
        'features_extracted': features_count,
        'batches_processed': batches_count,
        'total_operations': sum([
            1 if stats.get('normalize_time', 0) > 0 else 0,
            1 if stats.get('standardize_time', 0) > 0 else 0,
            1 if stats.get('feature_extraction_time', 0) > 0 else 0,
            1 if stats.get('batch_processing_time', 0) > 0 else 0,
        ]),
        'last_update': time.time()
    })
    
    # Calculate speedup if both times are available
    if stats.get('normalize_time', 0) > 0 and stats.get('normalize_python_time', 0) > 0:
        metrics['numpy_speedup'] = stats['normalize_python_time'] / stats['normalize_time']

if __name__ == '__main__':
    print("="*60)
    print("Starting NumPy Metrics Dashboard")
    print("="*60)
    print("Dashboard will be available at: http://localhost:5000")
    print("API endpoint: http://localhost:5000/api/metrics")
    print("Press Ctrl+C to stop")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=False)


