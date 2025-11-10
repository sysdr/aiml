#!/usr/bin/env python3
"""
Real-time Dashboard for Day 29: Central Limit Theorem Metrics
Run with: python dashboard.py
"""

import numpy as np
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify
from lesson_code import (
    CentralLimitTheoremSimulator,
    MLConfidenceCalculator,
    ABTestCalculator
)

app = Flask(__name__)

# Global state for metrics
metrics_data = {
    'clt_metrics': None,
    'model_metrics': None,
    'ab_test_metrics': None,
    'last_update': None,
    'is_running': False
}

# Initialize simulators
clt_simulator = CentralLimitTheoremSimulator(population_size=10000)

def generate_clt_metrics():
    """Generate CLT demonstration metrics"""
    # Use time-based seed for variation
    current_time = int(time.time() * 10) % (2**32 - 1)
    np.random.seed(current_time)
    
    # Run CLT demonstration with exponential distribution
    results = clt_simulator.demonstrate_clt(
        distribution_type='exponential',
        sample_size=30,
        num_samples=1000
    )
    
    # Calculate additional statistics
    sample_means = results['sample_means']
    pop_mean = results['pop_mean']
    pop_std = results['pop_std']
    sample_means_mean = results['sample_means_mean']
    sample_means_std = results['sample_means_std']
    theoretical_se = results['theoretical_se']
    
    # Calculate how close observed is to theoretical
    se_ratio = sample_means_std / theoretical_se if theoretical_se > 0 else 1.0
    mean_diff = abs(sample_means_mean - pop_mean)
    
    return {
        'population_mean': float(pop_mean),
        'population_std': float(pop_std),
        'sample_means_mean': float(sample_means_mean),
        'sample_means_std': float(sample_means_std),
        'theoretical_se': float(theoretical_se),
        'mean_difference': float(mean_diff),
        'se_ratio': float(se_ratio),
        'clt_verified': bool(mean_diff < 0.1 and 0.8 < se_ratio < 1.2),
        'sample_size': results['sample_size'],
        'num_samples': results['num_samples'],
        'distribution_type': 'exponential'
    }

def generate_model_metrics():
    """Generate ML model confidence interval metrics"""
    current_time = int(time.time() * 10) % (2**32 - 1)
    np.random.seed(current_time)
    
    test_size = 1000
    
    # Model A: Varying accuracy around 85%
    base_accuracy_a = 0.85 + np.sin(current_time * 0.1) * 0.03
    model_a_preds = np.random.binomial(1, base_accuracy_a, test_size)
    true_labels = np.ones(test_size)
    
    model_a_results = MLConfidenceCalculator.calculate_accuracy_ci(
        model_a_preds, true_labels
    )
    
    # Model B: Varying accuracy around 83%
    base_accuracy_b = 0.83 + np.cos(current_time * 0.1) * 0.02
    model_b_preds = np.random.binomial(1, base_accuracy_b, test_size)
    
    model_b_results = MLConfidenceCalculator.calculate_accuracy_ci(
        model_b_preds, true_labels
    )
    
    # Compare models
    comparison = MLConfidenceCalculator.compare_models(model_a_results, model_b_results)
    
    return {
        'model_a': {
            'accuracy': float(model_a_results['accuracy']),
            'standard_error': float(model_a_results['standard_error']),
            'ci_lower': float(model_a_results['ci_lower']),
            'ci_upper': float(model_a_results['ci_upper']),
            'margin_of_error': float(model_a_results['margin_of_error']),
            'sample_size': model_a_results['sample_size']
        },
        'model_b': {
            'accuracy': float(model_b_results['accuracy']),
            'standard_error': float(model_b_results['standard_error']),
            'ci_lower': float(model_b_results['ci_lower']),
            'ci_upper': float(model_b_results['ci_upper']),
            'margin_of_error': float(model_b_results['margin_of_error']),
            'sample_size': model_b_results['sample_size']
        },
        'comparison': {
            'difference': float(comparison['difference']),
            'intervals_overlap': bool(comparison['intervals_overlap']),
            'statistically_significant': bool(comparison['statistically_significant']),
            'p_value': float(comparison['p_value']),
            'z_statistic': float(comparison['z_statistic'])
        }
    }

def generate_ab_test_metrics():
    """Generate A/B test sample size metrics"""
    current_time = int(time.time() * 10) % (2**32 - 1)
    np.random.seed(current_time)
    
    # Varying baseline rate
    baseline_rate = 0.75 + np.sin(current_time * 0.05) * 0.05
    baseline_rate = max(0.5, min(0.95, baseline_rate))  # Clamp between 0.5 and 0.95
    
    # Different effect sizes
    effect_sizes = [0.01, 0.02, 0.03, 0.05, 0.10]
    sample_size_results = []
    
    for effect in effect_sizes:
        result = ABTestCalculator.calculate_sample_size(
            baseline_rate=baseline_rate,
            minimum_detectable_effect=effect
        )
        sample_size_results.append({
            'effect_size': float(effect),
            'effect_size_pct': float(effect * 100),
            'sample_size_per_group': result['sample_size_per_group'],
            'total_sample_size': result['total_sample_size'],
            'new_rate': float(result['new_rate'])
        })
    
    return {
        'baseline_rate': float(baseline_rate),
        'baseline_rate_pct': float(baseline_rate * 100),
        'effect_sizes': sample_size_results,
        'alpha': 0.05,
        'power': 0.80
    }

def calculate_all_metrics():
    """Calculate all dashboard metrics"""
    try:
        clt_metrics = generate_clt_metrics()
        model_metrics = generate_model_metrics()
        ab_test_metrics = generate_ab_test_metrics()
        
        return {
            'clt_metrics': clt_metrics,
            'model_metrics': model_metrics,
            'ab_test_metrics': ab_test_metrics,
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
                metrics_data['clt_metrics'] = metrics['clt_metrics']
                metrics_data['model_metrics'] = metrics['model_metrics']
                metrics_data['ab_test_metrics'] = metrics['ab_test_metrics']
                metrics_data['last_update'] = datetime.now().isoformat()
            
            # Update every 3 seconds
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
    if not metrics_data.get('clt_metrics'):
        # Initialize with default data
        metrics = calculate_all_metrics()
        if metrics:
            metrics_data['clt_metrics'] = metrics['clt_metrics']
            metrics_data['model_metrics'] = metrics['model_metrics']
            metrics_data['ab_test_metrics'] = metrics['ab_test_metrics']
            metrics_data['last_update'] = datetime.now().isoformat()
    
    return jsonify({
        'clt_metrics': metrics_data.get('clt_metrics', {}),
        'model_metrics': metrics_data.get('model_metrics', {}),
        'ab_test_metrics': metrics_data.get('ab_test_metrics', {}),
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
    print("🚀 Starting Day 29: Central Limit Theorem Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔄 Metrics update in real-time every 3 seconds")
    print("")
    
    # Start the demo automatically
    metrics_data['is_running'] = True
    thread = threading.Thread(target=update_metrics_thread, daemon=True)
    thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

