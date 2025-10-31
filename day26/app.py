#!/usr/bin/env python3
"""
Dashboard server for Day 26: Descriptive Statistics
Serves metrics and runs demo calculations
"""

import json
import sys
import os
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

# Add current directory to path to import lesson_code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from lesson_code import DescriptiveStats, DataProfiler, demo_real_world_scenarios
except ImportError as e:
    print(f"Error importing lesson_code: {e}")
    print("Make sure you have run: python lesson_code.py first")
    sys.exit(1)

app = Flask(__name__, static_folder='.')
CORS(app)

# Store last computed metrics
last_metrics = {
    'total_scenarios': 3,
    'total_datapoints': 0,
    'total_outliers': 0,
    'total_skewed': 0,
    'scenario1': None,
    'scenario2': None,
    'scenario3': None,
    'demo_status': 'Not Run',
    'last_run': None
}

def compute_metrics():
    """Compute statistics for all three scenarios"""
    global last_metrics
    
    # Scenario 1: User Session Duration
    session_data = [2, 3, 3, 4, 5, 3, 2, 120, 4, 3, 150, 3, 4, 3, 2, 5, 4, 3]
    profiler1 = DataProfiler(session_data, "session_duration_minutes")
    stats1 = profiler1.stats
    skew1 = profiler1.detect_skew()
    outliers1 = profiler1.detect_outliers_iqr()
    
    # Scenario 2: Transaction Amounts
    transaction_data = [45, 67, 52, 48, 51, 49, 53, 47, 50, 48, 
                       52, 3500, 48, 51, 49, 46, 52, 48]
    profiler2 = DataProfiler(transaction_data, "transaction_amount_usd")
    stats2 = profiler2.stats
    skew2 = profiler2.detect_skew()
    outliers2 = profiler2.detect_outliers_iqr()
    
    # Scenario 3: Model Latency
    latency_data = [45, 47, 46, 48, 44, 46, 47, 45, 49, 46, 
                   48, 45, 47, 280, 46, 47, 45, 48]
    profiler3 = DataProfiler(latency_data, "response_latency_ms")
    stats3 = profiler3.stats
    skew3 = profiler3.detect_skew()
    outliers3 = profiler3.detect_outliers_iqr()
    
    # Update metrics
    mode1 = stats1.mode()
    mode2 = stats2.mode()
    mode3 = stats3.mode()
    
    last_metrics['scenario1'] = {
        'mean': round(stats1.mean(), 2),
        'median': round(stats1.median(), 2),
        'mode': ', '.join(map(str, mode1)) if mode1 else '-',
        'skew_type': skew1['skew_type'],
        'outliers': outliers1['outlier_count'],
        'count': stats1.n
    }
    
    last_metrics['scenario2'] = {
        'mean': round(stats2.mean(), 2),
        'median': round(stats2.median(), 2),
        'mode': ', '.join(map(str, mode2)) if mode2 else '-',
        'skew_type': skew2['skew_type'],
        'outliers': outliers2['outlier_count'],
        'count': stats2.n
    }
    
    last_metrics['scenario3'] = {
        'mean': round(stats3.mean(), 2),
        'median': round(stats3.median(), 2),
        'mode': ', '.join(map(str, mode3)) if mode3 else '-',
        'skew_type': skew3['skew_type'],
        'outliers': outliers3['outlier_count'],
        'count': stats3.n
    }
    
    # Compute totals
    last_metrics['total_datapoints'] = stats1.n + stats2.n + stats3.n
    last_metrics['total_outliers'] = outliers1['outlier_count'] + outliers2['outlier_count'] + outliers3['outlier_count']
    last_metrics['total_skewed'] = sum(1 for s in [skew1, skew2, skew3] if s['skew_type'] != 'symmetric')
    
    last_metrics['last_run'] = json.dumps({'timestamp': 'now'})
    
    return last_metrics

@app.route('/')
def index():
    """Serve dashboard HTML"""
    response = send_from_directory('.', 'dashboard.html')
    # Prevent caching to ensure updates are visible
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/metrics')
def get_metrics():
    """Return current metrics"""
    # Recompute to ensure fresh data
    compute_metrics()
    return jsonify(last_metrics)

@app.route('/api/run-demo', methods=['POST'])
def run_demo():
    """Execute demo and update metrics"""
    try:
        # Run the demo (prints to console)
        demo_real_world_scenarios()
        
        # Compute and update metrics
        compute_metrics()
        last_metrics['demo_status'] = 'Running'
        
        return jsonify({
            'status': 'success',
            'message': 'Demo executed successfully',
            'metrics': last_metrics
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Day 26 Dashboard Server...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("📈 API endpoint: http://localhost:5000/api/metrics")
    print("\n⚠️  Note: Run 'python lesson_code.py' first to ensure all imports work")
    print("🔄 Press Ctrl+C to stop the server\n")
    
    # Compute initial metrics
    try:
        compute_metrics()
        print("✅ Initial metrics computed successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not compute initial metrics: {e}")
        print("   Run 'python lesson_code.py' to fix this")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
