#!/usr/bin/env python3
"""
Real-time Dashboard for Day 92: PCA for Dimensionality Reduction
Run with: python dashboard.py
"""

import numpy as np
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from lesson_code import ProductionPCA, run_pca_dimensionality_reduction

app = Flask(__name__)

# Global state for metrics
metrics_data = {
    'pca_metrics': None,
    'mnist_metrics': None,
    'compression_analysis': None,
    'last_update': None,
    'is_running': False
}

def calculate_pca_metrics():
    """Calculate PCA demonstration metrics"""
    # Use time-based seed for variation
    current_time = int(time.time() * 10) % (2**32 - 1)
    np.random.seed(current_time)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=20,
        n_redundant=30,
        n_repeated=10,
        random_state=current_time
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test different variance thresholds
    thresholds = [0.80, 0.90, 0.95, 0.99]
    compression_results = []
    
    for threshold in thresholds:
        pca = ProductionPCA(variance_threshold=threshold)
        pca.fit(X_train)
        
        compression_ratio = X.shape[1] / pca.n_components_optimal
        reconstruction_error = pca.get_reconstruction_error(X_test)
        
        compression_results.append({
            'threshold': float(threshold),
            'components': int(pca.n_components_optimal),
            'compression_ratio': float(compression_ratio),
            'variance_preserved': float(pca.get_cumulative_variance()[-1]),
            'reconstruction_error': float(reconstruction_error),
            'fit_time': float(pca.fit_time),
            'transform_time': float(pca.transform_time)
        })
    
    # Main PCA metrics (95% threshold)
    pca_main = ProductionPCA(variance_threshold=0.95)
    pca_main.fit(X_train)
    X_train_reduced = pca_main.transform(X_train)
    X_test_reduced = pca_main.transform(X_test)
    
    metrics = {
        'original_dims': int(X.shape[1]),
        'reduced_dims': int(pca_main.n_components_optimal),
        'compression_ratio': float(X.shape[1] / pca_main.n_components_optimal),
        'variance_preserved': float(pca_main.get_cumulative_variance()[-1]),
        'reconstruction_error_train': float(pca_main.get_reconstruction_error(X_train)),
        'reconstruction_error_test': float(pca_main.get_reconstruction_error(X_test)),
        'fit_time': float(pca_main.fit_time),
        'transform_time': float(pca_main.transform_time),
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1])
    }
    
    return metrics, compression_results

def calculate_mnist_metrics():
    """Calculate PCA on MNIST digits metrics"""
    digits = load_digits()
    X, y = digits.data, digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pca = ProductionPCA(variance_threshold=0.95)
    pca.fit(X_train)
    
    compression_ratio = X.shape[1] / pca.n_components_optimal
    reconstruction_error = pca.get_reconstruction_error(X_test)
    
    metrics = {
        'n_images': int(X.shape[0]),
        'pixels_per_image': int(X.shape[1]),
        'n_classes': int(len(np.unique(y))),
        'optimal_components': int(pca.n_components_optimal),
        'compression_ratio': float(compression_ratio),
        'variance_preserved': float(pca.get_cumulative_variance()[-1]),
        'reconstruction_error': float(reconstruction_error),
        'fit_time': float(pca.fit_time),
        'transform_time': float(pca.transform_time)
    }
    
    return metrics

def update_metrics():
    """Update metrics in background thread"""
    while metrics_data['is_running']:
        try:
            pca_metrics, compression_analysis = calculate_pca_metrics()
            metrics_data['pca_metrics'] = pca_metrics
            metrics_data['compression_analysis'] = compression_analysis
            metrics_data['mnist_metrics'] = calculate_mnist_metrics()
            metrics_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"Error updating metrics: {e}")
        
        time.sleep(5)  # Update every 5 seconds

@app.route('/')
def index():
    """Main dashboard page"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Day 92: PCA Dimensionality Reduction Dashboard</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                border-left: 4px solid #667eea;
            }
            .metric-card h3 {
                margin-top: 0;
                color: #667eea;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #764ba2;
            }
            .metric-label {
                color: #666;
                font-size: 0.9em;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #667eea;
                color: white;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .status {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: bold;
            }
            .status.running {
                background-color: #28a745;
                color: white;
            }
            .last-update {
                text-align: center;
                color: #666;
                margin-top: 20px;
                font-style: italic;
            }
            .demo-section {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 PCA Dimensionality Reduction Dashboard</h1>
            <p class="subtitle">Real-time metrics and performance monitoring</p>
            
            <div id="metrics-container">
                <p>Loading metrics...</p>
            </div>
            
            <div class="last-update">
                Last update: <span id="last-update">-</span>
            </div>
        </div>
        
        <script>
            function updateDashboard() {
                fetch('/api/metrics')
                    .then(response => response.json())
                    .then(data => {
                        if (data.pca_metrics && data.mnist_metrics) {
                            renderDashboard(data);
                        }
                    })
                    .catch(error => console.error('Error:', error));
            }
            
            function renderDashboard(data) {
                const container = document.getElementById('metrics-container');
                const pca = data.pca_metrics;
                const mnist = data.mnist_metrics;
                const compression = data.compression_analysis || [];
                
                let html = '<div class="metrics-grid">';
                
                // PCA Main Metrics
                html += '<div class="metric-card">';
                html += '<h3>PCA Performance</h3>';
                html += `<div class="metric-value">${pca.original_dims} → ${pca.reduced_dims}</div>`;
                html += '<div class="metric-label">Dimensionality Reduction</div>';
                html += `<div class="metric-value">${pca.compression_ratio.toFixed(2)}x</div>`;
                html += '<div class="metric-label">Compression Ratio</div>';
                html += '</div>';
                
                // Variance Preserved
                html += '<div class="metric-card">';
                html += '<h3>Variance Preserved</h3>';
                html += `<div class="metric-value">${(pca.variance_preserved * 100).toFixed(2)}%</div>`;
                html += '<div class="metric-label">Information Retained</div>';
                html += `<div class="metric-value">${pca.fit_time.toFixed(4)}s</div>`;
                html += '<div class="metric-label">Fit Time</div>';
                html += '</div>';
                
                // Reconstruction Error
                html += '<div class="metric-card">';
                html += '<h3>Reconstruction Error</h3>';
                html += `<div class="metric-value">${pca.reconstruction_error_test.toFixed(6)}</div>`;
                html += '<div class="metric-label">Test MSE</div>';
                html += `<div class="metric-value">${pca.transform_time.toFixed(4)}s</div>`;
                html += '<div class="metric-label">Transform Time</div>';
                html += '</div>';
                
                // MNIST Metrics
                html += '<div class="metric-card">';
                html += '<h3>MNIST Compression</h3>';
                html += `<div class="metric-value">${mnist.pixels_per_image} → ${mnist.optimal_components}</div>`;
                html += '<div class="metric-label">Pixels to Components</div>';
                html += `<div class="metric-value">${mnist.compression_ratio.toFixed(2)}x</div>`;
                html += '<div class="metric-label">Compression Ratio</div>';
                html += '</div>';
                
                html += '</div>';
                
                // Compression Analysis Table
                html += '<h2>Compression Analysis (Varying Variance Thresholds)</h2>';
                html += '<table><thead><tr><th>Threshold</th><th>Components</th><th>Compression</th><th>Variance</th><th>Recon Error</th><th>Fit Time (s)</th></tr></thead><tbody>';
                
                compression.forEach(result => {
                    html += `<tr>`;
                    html += `<td><strong>${(result.threshold * 100).toFixed(0)}%</strong></td>`;
                    html += `<td>${result.components}</td>`;
                    html += `<td>${result.compression_ratio.toFixed(2)}x</td>`;
                    html += `<td>${(result.variance_preserved * 100).toFixed(2)}%</td>`;
                    html += `<td>${result.reconstruction_error.toFixed(6)}</td>`;
                    html += `<td>${result.fit_time.toFixed(4)}</td>`;
                    html += `</tr>`;
                });
                
                html += '</tbody></table>';
                
                // MNIST Details
                html += '<div class="demo-section">';
                html += '<h2>MNIST Digits Dataset</h2>';
                html += '<div class="metrics-grid">';
                html += '<div class="metric-card">';
                html += `<div class="metric-value">${mnist.n_images}</div>`;
                html += '<div class="metric-label">Total Images</div>';
                html += '</div>';
                html += '<div class="metric-card">';
                html += `<div class="metric-value">${mnist.n_classes}</div>`;
                html += '<div class="metric-label">Digit Classes (0-9)</div>';
                html += '</div>';
                html += '<div class="metric-card">';
                html += `<div class="metric-value">${(mnist.variance_preserved * 100).toFixed(2)}%</div>`;
                html += '<div class="metric-label">Variance Preserved</div>';
                html += '</div>';
                html += '<div class="metric-card">';
                html += `<div class="metric-value">${mnist.reconstruction_error.toFixed(6)}</div>`;
                html += '<div class="metric-label">Reconstruction Error</div>';
                html += '</div>';
                html += '</div>';
                html += '</div>';
                
                container.innerHTML = html;
                document.getElementById('last-update').textContent = data.last_update || '-';
            }
            
            // Update immediately and then every 5 seconds
            updateDashboard();
            setInterval(updateDashboard, 5000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/api/metrics')
def get_metrics():
    """API endpoint for metrics"""
    return jsonify({
        'pca_metrics': metrics_data['pca_metrics'],
        'mnist_metrics': metrics_data['mnist_metrics'],
        'compression_analysis': metrics_data['compression_analysis'],
        'last_update': metrics_data['last_update'],
        'is_running': metrics_data['is_running']
    })

@app.route('/api/demo')
def demo():
    """Demo endpoint for testing"""
    try:
        metrics = run_pca_dimensionality_reduction(n_samples=500, n_features=50)
        # Convert numpy types to native Python types for JSON serialization
        metrics_serializable = {
            'original_dims': int(metrics['original_dims']),
            'reduced_dims': int(metrics['reduced_dims']),
            'compression_ratio': float(metrics['compression_ratio']),
            'variance_preserved': float(metrics['variance_preserved']),
            'reconstruction_error_train': float(metrics['reconstruction_error_train']),
            'reconstruction_error_test': float(metrics['reconstruction_error_test']),
            'fit_time': float(metrics['fit_time']),
            'transform_time': float(metrics['transform_time'])
        }
        return jsonify({
            'status': 'success',
            'metrics': metrics_serializable,
            'message': 'PCA demo executed successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Start metrics update thread
    metrics_data['is_running'] = True
    metrics_thread = threading.Thread(target=update_metrics, daemon=True)
    metrics_thread.start()
    
    # Initial metrics calculation
    try:
        pca_metrics, compression_analysis = calculate_pca_metrics()
        metrics_data['pca_metrics'] = pca_metrics
        metrics_data['compression_analysis'] = compression_analysis
        metrics_data['mnist_metrics'] = calculate_mnist_metrics()
        metrics_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error calculating initial metrics: {e}")
    
    print("🚀 Starting Day 92 PCA Dimensionality Reduction Dashboard...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🔄 Metrics update every 5 seconds")
    print("📈 Demo endpoint: http://localhost:5000/api/demo")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
        metrics_data['is_running'] = False

