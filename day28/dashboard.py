#!/usr/bin/env python3
"""
Real-time Dashboard for Day 28: Correlation and Covariance Metrics
Run with: python dashboard.py
"""

import numpy as np
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, Response, stream_with_context
from lesson_code import FeatureAnalyzer, generate_sample_data

app = Flask(__name__)

# Global state for metrics
metrics_data = {
    'feature_pairs': [],
    'correlation_matrix': None,
    'covariance_matrix': None,
    'last_update': None,
    'is_running': False
}

# Base data to maintain consistency with incremental variations
base_data, base_feature_names = generate_sample_data()

def generate_test_data():
    """Generate continuous test data for real-time updates with smooth variations"""
    # Use time-based seed for more noticeable variations
    # Keep seed within valid range (0 to 2**32-1)
    current_time = int(time.time() * 10) % (2**32 - 1)
    
    np.random.seed(current_time)
    n_samples = 200
    
    # Add time-based variation to make changes more noticeable
    time_variation = np.sin(current_time * 0.1) * 10  # Sinusoidal variation
    
    # Feature 1: Time spent on page (seconds) - base with noticeable variation
    time_spent = base_data[:, 0] + np.random.normal(time_variation, 5, n_samples)
    time_spent = np.clip(time_spent, 0, None)
    
    # Feature 2: Scroll depth (%) - correlated with time spent, with variation
    correlation_factor = 0.6 + np.sin(current_time * 0.05) * 0.1  # Varying correlation
    scroll_depth = correlation_factor * time_spent + np.random.normal(0, 8, n_samples)
    scroll_depth = np.clip(scroll_depth, 0, 100)
    
    # Feature 3: Number of clicks - somewhat correlated with time, varying
    click_factor = 0.03 + np.cos(current_time * 0.07) * 0.01
    num_clicks = click_factor * time_spent + np.random.normal(0, 1, n_samples)
    num_clicks = np.clip(num_clicks, 0, 15)
    
    # Feature 4: Return visits - independent with variation
    return_visits = np.random.poisson(3 + abs(np.sin(current_time * 0.03)) * 2, n_samples)
    
    # Feature 5: Social shares - weakly correlated with engagement, varying
    share_factor = 0.01 + np.sin(current_time * 0.06) * 0.005
    social_shares = share_factor * time_spent + 0.2 * num_clicks + np.random.exponential(0.5, n_samples)
    
    data = np.column_stack([
        time_spent,
        scroll_depth,
        num_clicks,
        return_visits,
        social_shares
    ])
    
    return data, base_feature_names

def calculate_metrics():
    """Calculate all correlation and covariance metrics"""
    data, feature_names = generate_test_data()
    analyzer = FeatureAnalyzer(data, feature_names)
    
    # Calculate matrices
    cov_matrix = analyzer.calculate_covariance_manual()
    corr_matrix = analyzer.calculate_correlation_manual()
    
    # Find highly correlated pairs
    high_corr_pairs = analyzer.find_highly_correlated_features(threshold=0.7)
    
    # Feature removal suggestions
    suggestions = analyzer.suggest_features_to_remove(threshold=0.9)
    
    # Get correlation statistics
    upper_triangle = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            upper_triangle.append(abs(corr_matrix[i, j]))
    
    # Create feature pair metrics
    feature_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            pair_name = f"{feature_names[i]} ↔ {feature_names[j]}"
            correlation = float(corr_matrix[i, j])
            covariance = float(cov_matrix[i, j])
            
            # Determine relationship strength
            abs_corr = abs(correlation)
            if abs_corr >= 0.9:
                strength = "VERY_STRONG"
                strength_icon = "🔴"
                advice = "Very high correlation - consider removing one feature"
            elif abs_corr >= 0.7:
                strength = "STRONG"
                strength_icon = "🟠"
                advice = "Strong correlation - monitor for redundancy"
            elif abs_corr >= 0.5:
                strength = "MODERATE"
                strength_icon = "🟡"
                advice = "Moderate correlation - features are related"
            elif abs_corr >= 0.3:
                strength = "WEAK"
                strength_icon = "🟢"
                advice = "Weak correlation - features are somewhat independent"
            else:
                strength = "VERY_WEAK"
                strength_icon = "⚪"
                advice = "Very weak correlation - features are independent"
            
            feature_pairs.append({
                'pair_name': pair_name,
                'feature1': feature_names[i],
                'feature2': feature_names[j],
                'correlation': correlation,
                'covariance': covariance,
                'abs_correlation': abs_corr,
                'strength': strength,
                'strength_icon': strength_icon,
                'advice': advice,
                'is_high_correlation': abs_corr >= 0.7
            })
    
    # Sort by absolute correlation (descending)
    feature_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    return {
        'feature_pairs': feature_pairs,
        'correlation_matrix': corr_matrix.tolist(),
        'covariance_matrix': cov_matrix.tolist(),
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_samples': data.shape[0],
        'avg_correlation': float(np.mean(upper_triangle)),
        'max_correlation': float(np.max(upper_triangle)),
        'min_correlation': float(np.min(upper_triangle)),
        'high_corr_pairs': len(high_corr_pairs),
        'features_to_keep': suggestions['keep'],
        'features_to_remove': suggestions['remove'],
        'timestamp': datetime.now().isoformat()
    }

def update_metrics_thread():
    """Background thread to continuously update metrics"""
    global metrics_data
    
    while metrics_data['is_running']:
        try:
            metrics = calculate_metrics()
            metrics_data['feature_pairs'] = metrics['feature_pairs']
            metrics_data['correlation_matrix'] = metrics['correlation_matrix']
            metrics_data['covariance_matrix'] = metrics['covariance_matrix']
            metrics_data['last_update'] = datetime.now().isoformat()
            metrics_data['current_metrics'] = metrics
            
            # Simulate new data every 3 seconds for more frequent updates
            time.sleep(3)
        except Exception as e:
            print(f"Error updating metrics: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics (REST API)"""
    if not metrics_data.get('current_metrics'):
        # Initialize with default data
        metrics = calculate_metrics()
        metrics_data['feature_pairs'] = metrics['feature_pairs']
        metrics_data['correlation_matrix'] = metrics['correlation_matrix']
        metrics_data['covariance_matrix'] = metrics['covariance_matrix']
        metrics_data['last_update'] = datetime.now().isoformat()
        metrics_data['current_metrics'] = metrics
    
    return jsonify(metrics_data['current_metrics'])

@app.route('/api/metrics/stream')
def stream_metrics():
    """Server-Sent Events stream for real-time updates"""
    def event_stream():
        while True:
            # If demo is running, use the updated metrics from the thread
            # Otherwise, recalculate with current time to show it's "live"
            if metrics_data.get('is_running') and metrics_data.get('current_metrics'):
                # Use data from background thread
                data = {
                    **metrics_data['current_metrics'],
                    'last_update': metrics_data['last_update'],
                    'timestamp': datetime.now().isoformat()
                }
            elif metrics_data.get('current_metrics'):
                # Demo not running, but still show current data (static)
                data = {
                    **metrics_data['current_metrics'],
                    'last_update': metrics_data['last_update'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # No data yet, generate initial
                metrics = calculate_metrics()
                data = {
                    **metrics,
                    'last_update': datetime.now().isoformat(),
                    'timestamp': datetime.now().isoformat()
                }
            
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(3)  # Update every 3 seconds
    
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
        'feature_pair_count': len(metrics_data.get('feature_pairs', []))
    })

if __name__ == '__main__':
    # Initialize with default data
    metrics = calculate_metrics()
    metrics_data['feature_pairs'] = metrics['feature_pairs']
    metrics_data['correlation_matrix'] = metrics['correlation_matrix']
    metrics_data['covariance_matrix'] = metrics['covariance_matrix']
    metrics_data['last_update'] = datetime.now().isoformat()
    metrics_data['current_metrics'] = metrics
    
    print("🚀 Starting Correlation & Covariance Dashboard Server...")
    print("📊 Access the dashboard at: http://localhost:5000")
    print("🔄 Metrics update in real-time every 3 seconds")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

