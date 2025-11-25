#!/usr/bin/env python3
"""
Real-time Dashboard for Day 35: Data Cleaning and Handling Missing Data
Run with: python dashboard.py
"""

import pandas as pd
import numpy as np
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from lesson_code import (
    MissingDataDetector,
    DataCleaner,
    generate_messy_data
)

app = Flask(__name__)

# Global state for metrics
metrics_data = {
    'cleaning_metrics': None,
    'missing_data_report': None,
    'cleaning_log': None,
    'last_update': None,
    'is_running': False
}

def generate_cleaning_metrics():
    """Generate data cleaning demonstration metrics"""
    # Use time-based seed for variation
    current_time = int(time.time() * 10) % (2**32 - 1)
    np.random.seed(current_time)
    
    # Generate messy data
    df = generate_messy_data()
    
    # Analyze missing data
    detector = MissingDataDetector(df)
    report = detector.generate_report()
    patterns = detector.visualize_patterns()
    
    # Clean the data
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                  .drop_high_missing_columns(threshold=0.7)
                  .fill_numeric_median(['age', 'income', 'session_duration', 'last_login_days'])
                  .fill_categorical_mode(['country', 'subscription_type'])
                  .get_cleaned_data())
    
    validation = cleaner.validate_cleaning()
    
    # Calculate metrics
    original_missing = df.isna().sum().sum()
    cleaned_missing = cleaned_df.isna().sum().sum()
    missing_pct_before = (original_missing / (df.shape[0] * df.shape[1])) * 100
    missing_pct_after = (cleaned_missing / (cleaned_df.shape[0] * cleaned_df.shape[1])) * 100
    
    # Get missing data breakdown by column
    missing_by_column = []
    for idx, row in report.iterrows():
        missing_by_column.append({
            'column': row['column'],
            'missing_count': int(row['missing_count']),
            'missing_pct': float(row['missing_pct']),
            'strategy': row['recommended_strategy']
        })
    
    return {
        'original_shape': {
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1])
        },
        'cleaned_shape': {
            'rows': int(cleaned_df.shape[0]),
            'columns': int(cleaned_df.shape[1])
        },
        'missing_data': {
            'before': {
                'count': int(original_missing),
                'percentage': float(round(missing_pct_before, 2))
            },
            'after': {
                'count': int(cleaned_missing),
                'percentage': float(round(missing_pct_after, 2))
            },
            'removed': int(original_missing - cleaned_missing)
        },
        'rows_affected': int(patterns['rows_affected']),
        'columns_removed': int(validation['columns_removed']),
        'rows_removed': int(validation['rows_removed']),
        'is_clean': bool(validation['is_clean']),
        'missing_by_column': missing_by_column,
        'cleaning_operations': len(validation['cleaning_log'])
    }

def calculate_all_metrics():
    """Calculate all dashboard metrics"""
    try:
        cleaning_metrics = generate_cleaning_metrics()
        
        return {
            'cleaning_metrics': cleaning_metrics,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_metrics_thread():
    """Background thread to continuously update metrics"""
    global metrics_data
    
    while metrics_data['is_running']:
        try:
            metrics = calculate_all_metrics()
            if metrics:
                metrics_data['cleaning_metrics'] = metrics['cleaning_metrics']
                metrics_data['last_update'] = datetime.now().isoformat()
            
            # Update every 3 seconds
            time.sleep(3)
        except Exception as e:
            print(f"Error updating metrics: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

# HTML template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Day 35: Data Cleaning Dashboard</title>
    <meta http-equiv="refresh" content="3">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .card h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            font-weight: 600;
            color: #666;
        }
        .metric-value {
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .status-clean {
            background: #10b981;
            color: white;
        }
        .status-dirty {
            background: #ef4444;
            color: white;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e5e7eb;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #059669);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .table-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #667eea;
            color: white;
            font-weight: 600;
        }
        tr:hover {
            background: #f9fafb;
        }
        .footer {
            text-align: center;
            color: white;
            margin-top: 30px;
            padding: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧹 Data Cleaning & Missing Data Dashboard</h1>
        
        <div class="cards">
            <div class="card">
                <h2>📊 Dataset Overview</h2>
                <div class="metric">
                    <span class="metric-label">Original Rows:</span>
                    <span class="metric-value" id="original_rows">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Original Columns:</span>
                    <span class="metric-value" id="original_cols">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cleaned Rows:</span>
                    <span class="metric-value" id="cleaned_rows">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cleaned Columns:</span>
                    <span class="metric-value" id="cleaned_cols">-</span>
                </div>
            </div>
            
            <div class="card">
                <h2>🔍 Missing Data</h2>
                <div class="metric">
                    <span class="metric-label">Before Cleaning:</span>
                    <span class="metric-value" id="missing_before">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">After Cleaning:</span>
                    <span class="metric-value" id="missing_after">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Removed:</span>
                    <span class="metric-value" style="color: #10b981;" id="missing_removed">-</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="clean_progress" style="width: 0%">0%</div>
                </div>
            </div>
            
            <div class="card">
                <h2>✅ Cleaning Status</h2>
                <div class="metric">
                    <span class="metric-label">Data Status:</span>
                    <span id="data_status">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Rows Affected:</span>
                    <span class="metric-value" id="rows_affected">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Columns Removed:</span>
                    <span class="metric-value" id="cols_removed">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cleaning Operations:</span>
                    <span class="metric-value" id="operations">-</span>
                </div>
            </div>
        </div>
        
        <div class="table-container">
            <h2 style="color: #667eea; margin-bottom: 15px;">📋 Missing Data by Column</h2>
            <table id="missing_table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Missing Count</th>
                        <th>Missing %</th>
                        <th>Recommended Strategy</th>
                    </tr>
                </thead>
                <tbody id="missing_table_body">
                    <tr><td colspan="4" style="text-align: center;">Loading...</td></tr>
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Last updated: <span id="last_update">-</span></p>
            <p>Metrics update every 3 seconds</p>
        </div>
    </div>
    
    <script>
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    if (data.cleaning_metrics) {
                        const m = data.cleaning_metrics;
                        
                        // Update overview
                        document.getElementById('original_rows').textContent = m.original_shape.rows.toLocaleString();
                        document.getElementById('original_cols').textContent = m.original_shape.columns;
                        document.getElementById('cleaned_rows').textContent = m.cleaned_shape.rows.toLocaleString();
                        document.getElementById('cleaned_cols').textContent = m.cleaned_shape.columns;
                        
                        // Update missing data
                        document.getElementById('missing_before').textContent = 
                            m.missing_data.before.count.toLocaleString() + ' (' + m.missing_data.before.percentage + '%)';
                        document.getElementById('missing_after').textContent = 
                            m.missing_data.after.count.toLocaleString() + ' (' + m.missing_data.after.percentage + '%)';
                        document.getElementById('missing_removed').textContent = 
                            m.missing_data.removed.toLocaleString();
                        
                        // Update progress bar
                        const cleanPct = 100 - m.missing_data.after.percentage;
                        const progressBar = document.getElementById('clean_progress');
                        progressBar.style.width = cleanPct + '%';
                        progressBar.textContent = cleanPct.toFixed(1) + '%';
                        
                        // Update status
                        const status = m.is_clean ? 
                            '<span class="status-badge status-clean">✅ CLEAN</span>' : 
                            '<span class="status-badge status-dirty">⚠️ HAS MISSING DATA</span>';
                        document.getElementById('data_status').innerHTML = status;
                        
                        document.getElementById('rows_affected').textContent = m.rows_affected.toLocaleString();
                        document.getElementById('cols_removed').textContent = m.columns_removed;
                        document.getElementById('operations').textContent = m.cleaning_operations;
                        
                        // Update table
                        const tbody = document.getElementById('missing_table_body');
                        tbody.innerHTML = m.missing_by_column.map(col => 
                            '<tr>' +
                            '<td><strong>' + col.column + '</strong></td>' +
                            '<td>' + col.missing_count.toLocaleString() + '</td>' +
                            '<td>' + col.missing_pct + '%</td>' +
                            '<td>' + col.strategy + '</td>' +
                            '</tr>'
                        ).join('');
                        
                        // Update timestamp
                        document.getElementById('last_update').textContent = 
                            new Date(data.last_update).toLocaleString();
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        // Update immediately and then every 3 seconds
        updateDashboard();
        setInterval(updateDashboard, 3000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics (REST API)"""
    if not metrics_data.get('cleaning_metrics'):
        # Initialize with default data
        metrics = calculate_all_metrics()
        if metrics:
            metrics_data['cleaning_metrics'] = metrics['cleaning_metrics']
            metrics_data['last_update'] = datetime.now().isoformat()
    
    return jsonify({
        'cleaning_metrics': metrics_data.get('cleaning_metrics', {}),
        'last_update': metrics_data.get('last_update', ''),
        'is_running': metrics_data['is_running']
    })

@app.route('/api/start')
def start_demo():
    """Start the demo"""
    if not metrics_data['is_running']:
        metrics_data['is_running'] = True
        thread = threading.Thread(target=update_metrics_thread, daemon=True)
        thread.start()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/api/stop')
def stop_demo():
    """Stop the demo"""
    metrics_data['is_running'] = False
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    # Initialize metrics on startup
    print("🚀 Starting Day 35: Data Cleaning Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔄 Metrics update in real-time every 3 seconds")
    print("")
    
    # Start the demo automatically
    metrics_data['is_running'] = True
    thread = threading.Thread(target=update_metrics_thread, daemon=True)
    thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

