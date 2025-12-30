#!/usr/bin/env python3
"""
Real-time Dashboard for Day 59: Decision Trees Metrics
Run with: python dashboard.py
"""

import numpy as np
import pandas as pd
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from lesson_code import CustomerChurnPredictor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

app = Flask(__name__)

# Global state for metrics
metrics_data = {
    'model_metrics': None,
    'tree_info': None,
    'feature_importance': None,
    'confusion_matrix': None,
    'last_update': None,
    'is_running': False
}

# Store predictor separately (not JSON-serializable)
_predictor = None

# HTML Template for Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day 59: Decision Trees Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #00BCD4 0%, #4CAF50 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
            text-align: center;
        }
        .header h1 {
            color: #00BCD4;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-card h3 {
            color: #00BCD4;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .section {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .section h2 {
            color: #00BCD4;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        .confusion-matrix {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            max-width: 400px;
            margin: 0 auto;
        }
        .cm-cell {
            padding: 15px;
            text-align: center;
            border-radius: 8px;
            font-weight: bold;
        }
        .cm-header {
            background: #00BCD4;
            color: white;
        }
        .cm-tn { background: #4caf50; color: white; }
        .cm-fp { background: #ff9800; color: white; }
        .cm-fn { background: #ff9800; color: white; }
        .cm-tp { background: #4caf50; color: white; }
        .tree-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .tree-info-item {
            display: flex;
            justify-content: space-between;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
        }
        .feature-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .feature-item {
            display: flex;
            justify-content: space-between;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
            align-items: center;
        }
        .feature-name {
            font-weight: bold;
            color: #333;
        }
        .feature-importance {
            color: #00BCD4;
            font-weight: bold;
        }
        .btn {
            background: #00BCD4;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            margin: 0 10px;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #0097A7;
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-left: 10px;
        }
        .status.running {
            background: #4caf50;
            color: white;
        }
        .status.stopped {
            background: #f44336;
            color: white;
        }
        .timestamp {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌳 Decision Trees Dashboard</h1>
            <p>Real-time Customer Churn Prediction Metrics</p>
            <div class="controls">
                <button class="btn" id="startBtn" onclick="startDemo()">▶ Start Demo</button>
                <button class="btn" id="stopBtn" onclick="stopDemo()" disabled>⏹ Stop Demo</button>
                <span class="status stopped" id="status">Stopped</span>
            </div>
        </div>

        <div class="metrics-grid" id="metricsGrid">
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="metric-value" id="accuracy">-</div>
                <div class="metric-label">Overall Model Accuracy</div>
            </div>
            <div class="metric-card">
                <h3>Precision</h3>
                <div class="metric-value" id="precision">-</div>
                <div class="metric-label">Churn Detection Precision</div>
            </div>
            <div class="metric-card">
                <h3>Recall</h3>
                <div class="metric-value" id="recall">-</div>
                <div class="metric-label">Churn Detection Recall</div>
            </div>
            <div class="metric-card">
                <h3>F1-Score</h3>
                <div class="metric-value" id="f1score">-</div>
                <div class="metric-label">Harmonic Mean</div>
            </div>
            <div class="metric-card">
                <h3>ROC-AUC</h3>
                <div class="metric-value" id="rocAuc">-</div>
                <div class="metric-label">Area Under ROC Curve</div>
            </div>
            <div class="metric-card">
                <h3>Tree Depth</h3>
                <div class="metric-value" id="treeDepth">-</div>
                <div class="metric-label">Maximum Depth</div>
            </div>
            <div class="metric-card">
                <h3>Test Samples</h3>
                <div class="metric-value" id="testSamples">-</div>
                <div class="metric-label">Number of Test Samples</div>
            </div>
            <div class="metric-card">
                <h3>Churn Rate</h3>
                <div class="metric-value" id="churnRate">-</div>
                <div class="metric-label">Dataset Churn Percentage</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Confusion Matrix</h2>
            <div class="confusion-matrix" id="confusionMatrix">
                <div class="cm-cell cm-header"></div>
                <div class="cm-cell cm-header">Predicted Retained</div>
                <div class="cm-cell cm-header">Predicted Churned</div>
                <div class="cm-cell cm-header">Actual Retained</div>
                <div class="cm-cell cm-tn" id="tn">-</div>
                <div class="cm-cell cm-fp" id="fp">-</div>
                <div class="cm-cell cm-header">Actual Churned</div>
                <div class="cm-cell cm-fn" id="fn">-</div>
                <div class="cm-cell cm-tp" id="tp">-</div>
            </div>
        </div>

        <div class="section">
            <h2>🌳 Tree Information</h2>
            <div class="tree-info" id="treeInfo">
                <div class="tree-info-item">
                    <span>Max Depth:</span>
                    <span id="maxDepth">-</span>
                </div>
                <div class="tree-info-item">
                    <span>Min Samples Split:</span>
                    <span id="minSamples">-</span>
                </div>
                <div class="tree-info-item">
                    <span>Training Samples:</span>
                    <span id="trainSamples">-</span>
                </div>
                <div class="tree-info-item">
                    <span>CV ROC-AUC Mean:</span>
                    <span id="cvMean">-</span>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🔍 Feature Importance</h2>
            <div class="feature-list" id="featureImportance"></div>
        </div>

        <div class="timestamp" id="timestamp">Last updated: Never</div>
    </div>

    <script>
        let updateInterval = null;

        function updateDashboard(data) {
            // Update metrics
            if (data.model_metrics) {
                const m = data.model_metrics;
                document.getElementById('accuracy').textContent = (m.accuracy * 100).toFixed(2) + '%';
                document.getElementById('precision').textContent = (m.precision * 100).toFixed(2) + '%';
                document.getElementById('recall').textContent = (m.recall * 100).toFixed(2) + '%';
                document.getElementById('f1score').textContent = m.f1_score.toFixed(3);
                if (m.roc_auc !== undefined) {
                    document.getElementById('rocAuc').textContent = m.roc_auc.toFixed(3);
                }
            }

            // Update tree info
            if (data.tree_info) {
                const t = data.tree_info;
                document.getElementById('treeDepth').textContent = t.max_depth || '-';
                document.getElementById('maxDepth').textContent = t.max_depth || '-';
                document.getElementById('minSamples').textContent = t.min_samples_split || '-';
                document.getElementById('trainSamples').textContent = t.train_samples || '-';
                document.getElementById('testSamples').textContent = t.test_samples || '-';
                if (t.cv_mean !== undefined) {
                    document.getElementById('cvMean').textContent = t.cv_mean.toFixed(3) + ' ± ' + t.cv_std.toFixed(3);
                }
                if (t.churn_rate !== undefined) {
                    document.getElementById('churnRate').textContent = (t.churn_rate * 100).toFixed(1) + '%';
                }
            }

            // Update confusion matrix
            if (data.confusion_matrix) {
                const cm = data.confusion_matrix;
                document.getElementById('tn').textContent = cm[0][0];
                document.getElementById('fp').textContent = cm[0][1];
                document.getElementById('fn').textContent = cm[1][0];
                document.getElementById('tp').textContent = cm[1][1];
            }

            // Update feature importance
            if (data.feature_importance) {
                const container = document.getElementById('featureImportance');
                container.innerHTML = '';
                data.feature_importance.forEach(item => {
                    const div = document.createElement('div');
                    div.className = 'feature-item';
                    div.innerHTML = `
                        <span class="feature-name">${item.feature}</span>
                        <span class="feature-importance">${(item.importance * 100).toFixed(2)}%</span>
                    `;
                    container.appendChild(div);
                });
            }

            // Update timestamp
            if (data.last_update) {
                document.getElementById('timestamp').textContent = 'Last updated: ' + data.last_update;
            }
        }

        function fetchMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    updateDashboard(data);
                })
                .catch(error => {
                    console.error('Error fetching metrics:', error);
                });
        }

        function startDemo() {
            fetch('/api/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = false;
                        document.getElementById('status').textContent = 'Running';
                        document.getElementById('status').className = 'status running';
                        updateInterval = setInterval(fetchMetrics, 3000);
                        fetchMetrics();
                    }
                });
        }

        function stopDemo() {
            fetch('/api/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        document.getElementById('status').textContent = 'Stopped';
                        document.getElementById('status').className = 'status stopped';
                        if (updateInterval) {
                            clearInterval(updateInterval);
                            updateInterval = null;
                        }
                    }
                });
        }

        // Initial load
        fetchMetrics();
        setInterval(fetchMetrics, 3000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    return jsonify(metrics_data)

@app.route('/api/start', methods=['POST'])
def start_demo():
    """Start the demo by training the model"""
    try:
        # Initialize predictor
        predictor = CustomerChurnPredictor(max_depth=10, min_samples_split=50)
        
        # Generate data
        X, y = predictor.generate_synthetic_data(n_samples=10000)
        churn_rate = y.mean()
        
        # Train model
        metrics = predictor.train(X, y, use_grid_search=False)
        
        # Get feature importance
        importance_df = predictor.get_feature_importance()
        feature_importance_list = [
            {'feature': row['feature'], 'importance': float(row['importance'])}
            for _, row in importance_df.iterrows()
        ]
        
        # Extract confusion matrix
        cm = metrics['confusion_matrix'].tolist()
        
        # Calculate additional metrics
        y_test = metrics['y_test']
        y_pred = metrics['y_pred']
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Get tree info
        tree = predictor.model.tree_
        
        # Update metrics
        metrics_data['model_metrics'] = {
            'accuracy': float(metrics['test_accuracy']),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(metrics['test_roc_auc'])
        }
        metrics_data['tree_info'] = {
            'max_depth': int(tree.max_depth),
            'min_samples_split': int(predictor.min_samples_split),
            'train_samples': len(X) - len(metrics['y_test']),
            'test_samples': len(metrics['y_test']),
            'cv_mean': float(metrics['cv_mean']),
            'cv_std': float(metrics['cv_std']),
            'churn_rate': float(churn_rate)
        }
        metrics_data['feature_importance'] = feature_importance_list
        metrics_data['confusion_matrix'] = cm
        metrics_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics_data['is_running'] = True
        # Store predictor separately (not in JSON-serializable metrics_data)
        global _predictor
        _predictor = predictor
        
        return jsonify({'success': True, 'message': 'Demo started'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_demo():
    """Stop the demo"""
    metrics_data['is_running'] = False
    global _predictor
    _predictor = None
    return jsonify({'success': True, 'message': 'Demo stopped'})

if __name__ == '__main__':
    print("="*60)
    print("Day 59: Decision Trees Dashboard")
    print("="*60)
    print("\n📊 Dashboard starting on http://localhost:5000")
    print("🔄 Metrics will update every 3 seconds")
    print("\nPress Ctrl+C to stop")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=False)
