#!/usr/bin/env python3
"""
Real-time Dashboard for Day 51: Spam Detection Metrics
Run with: python dashboard.py
"""

import numpy as np
import pandas as pd
import json
import time
import threading
import os
import joblib
from datetime import datetime
from flask import Flask, render_template_string, jsonify, Response, stream_with_context
from lesson_code import SpamDetector

app = Flask(__name__)

# Global state for metrics
metrics_data = {
    'model_metrics': None,
    'feature_importance': None,
    'inference_stats': None,
    'last_update': None,
    'is_running': False,
    'detector': None
}

# HTML Template for Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day 51: Spam Detection Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #00b4db 0%, #00d4aa 100%);
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
            color: #00b4db;
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
            color: #00b4db;
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
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .section h2 {
            color: #00b4db;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        .confusion-matrix {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            max-width: 500px;
            margin: 20px auto;
        }
        .cm-cell {
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
        }
        .cm-header {
            background: #00b4db;
            color: white;
        }
        .cm-tp { background: #4caf50; color: white; }
        .cm-tn { background: #2196f3; color: white; }
        .cm-fp { background: #ff9800; color: white; }
        .cm-fn { background: #f44336; color: white; }
        .feature-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .feature-item {
            padding: 10px;
            margin: 5px 0;
            background: #f5f5f5;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .feature-name {
            font-weight: bold;
            color: #333;
        }
        .feature-coef {
            color: #00b4db;
            font-weight: bold;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .btn {
            background: #00b4db;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            margin: 0 10px;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #0099cc;
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
            <h1>🚀 Spam Detection Dashboard</h1>
            <p>Real-time Model Performance Metrics</p>
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
                <div class="metric-label">Spam Detection Precision</div>
            </div>
            <div class="metric-card">
                <h3>Recall</h3>
                <div class="metric-value" id="recall">-</div>
                <div class="metric-label">Spam Detection Recall</div>
            </div>
            <div class="metric-card">
                <h3>F1-Score</h3>
                <div class="metric-value" id="f1score">-</div>
                <div class="metric-label">Harmonic Mean</div>
            </div>
            <div class="metric-card">
                <h3>ROC-AUC</h3>
                <div class="metric-value" id="rocauc">-</div>
                <div class="metric-label">Area Under Curve</div>
            </div>
            <div class="metric-card">
                <h3>Throughput</h3>
                <div class="metric-value" id="throughput">-</div>
                <div class="metric-label">Emails/Second</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Confusion Matrix</h2>
            <div class="confusion-matrix" id="confusionMatrix">
                <div class="cm-cell cm-header"></div>
                <div class="cm-cell cm-header">Predicted Ham</div>
                <div class="cm-cell cm-header">Predicted Spam</div>
                <div class="cm-cell cm-header">Actual Ham</div>
                <div class="cm-cell cm-tn" id="tn">-</div>
                <div class="cm-cell cm-fp" id="fp">-</div>
                <div class="cm-cell cm-header">Actual Spam</div>
                <div class="cm-cell cm-fn" id="fn">-</div>
                <div class="cm-cell cm-tp" id="tp">-</div>
            </div>
        </div>

        <div class="section">
            <h2>🔍 Top Spam Indicators</h2>
            <div class="feature-list" id="spamFeatures"></div>
        </div>

        <div class="section">
            <h2>✅ Top Ham Indicators</h2>
            <div class="feature-list" id="hamFeatures"></div>
        </div>

        <div class="timestamp" id="timestamp">Last updated: Never</div>
    </div>

    <script>
        let eventSource = null;

        function updateDashboard(data) {
            // Update metrics
            if (data.model_metrics) {
                const m = data.model_metrics;
                document.getElementById('accuracy').textContent = (m.accuracy * 100).toFixed(2) + '%';
                document.getElementById('precision').textContent = (m.precision * 100).toFixed(2) + '%';
                document.getElementById('recall').textContent = (m.recall * 100).toFixed(2) + '%';
                document.getElementById('f1score').textContent = m.f1_score.toFixed(3);
                document.getElementById('rocauc').textContent = m.roc_auc.toFixed(4);
            }

            // Update confusion matrix
            if (data.confusion_matrix) {
                const cm = data.confusion_matrix;
                document.getElementById('tn').textContent = cm[0][0];
                document.getElementById('fp').textContent = cm[0][1];
                document.getElementById('fn').textContent = cm[1][0];
                document.getElementById('tp').textContent = cm[1][1];
            }

            // Update inference stats
            if (data.inference_stats) {
                const stats = data.inference_stats;
                document.getElementById('throughput').textContent = Math.round(stats.emails_per_second).toLocaleString();
            }

            // Update feature importance
            if (data.feature_importance) {
                const spamFeatures = data.feature_importance.spam_indicators || [];
                const hamFeatures = data.feature_importance.ham_indicators || [];
                
                const spamHtml = spamFeatures.slice(0, 10).map(f => `
                    <div class="feature-item">
                        <span class="feature-name">${f.feature}</span>
                        <span class="feature-coef">+${f.coefficient.toFixed(4)}</span>
                    </div>
                `).join('');
                document.getElementById('spamFeatures').innerHTML = spamHtml;

                const hamHtml = hamFeatures.slice(0, 10).map(f => `
                    <div class="feature-item">
                        <span class="feature-name">${f.feature}</span>
                        <span class="feature-coef">${f.coefficient.toFixed(4)}</span>
                    </div>
                `).join('');
                document.getElementById('hamFeatures').innerHTML = hamHtml;
            }

            // Update timestamp
            if (data.last_update) {
                document.getElementById('timestamp').textContent = 'Last updated: ' + new Date(data.last_update).toLocaleString();
            }
        }

        function startDemo() {
            fetch('/api/start', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    console.log('Demo started:', data);
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('status').textContent = 'Running';
                    document.getElementById('status').className = 'status running';
                    
                    // Start SSE stream
                    if (eventSource) eventSource.close();
                    eventSource = new EventSource('/api/metrics/stream');
                    eventSource.onmessage = (e) => {
                        const data = JSON.parse(e.data);
                        updateDashboard(data);
                    };
                });
        }

        function stopDemo() {
            fetch('/api/stop', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    console.log('Demo stopped:', data);
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    document.getElementById('status').textContent = 'Stopped';
                    document.getElementById('status').className = 'status stopped';
                    
                    if (eventSource) {
                        eventSource.close();
                        eventSource = null;
                    }
                });
        }

        // Load initial data
        fetch('/api/metrics')
            .then(r => r.json())
            .then(data => updateDashboard(data));

        // Auto-refresh every 3 seconds
        setInterval(() => {
            fetch('/api/metrics')
                .then(r => r.json())
                .then(data => updateDashboard(data));
        }, 3000);
    </script>
</body>
</html>
"""

def load_model_metrics():
    """Load model metrics from saved files"""
    metrics = {}
    
    # Load evaluation report
    if os.path.exists('evaluation_report.txt'):
        with open('evaluation_report.txt', 'r') as f:
            content = f.read()
            # Parse metrics (simplified parsing)
            if 'ROC-AUC Score:' in content:
                roc_auc = float(content.split('ROC-AUC Score:')[1].split()[0])
                metrics['roc_auc'] = roc_auc
    
    # Load model if available
    detector = None
    if os.path.exists('spam_model.pkl') and os.path.exists('spambase.data'):
        try:
            detector = SpamDetector(random_state=42)
            data = detector.load_data()
            X, y = detector.prepare_features(data)
            detector.split_data(X, y)
            detector.model = joblib.load('spam_model.pkl')
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            y_pred = detector.model.predict(detector.X_test)
            y_pred_proba = detector.model.predict_proba(detector.X_test)[:, 1]
            
            metrics['accuracy'] = accuracy_score(detector.y_test, y_pred)
            metrics['precision'] = precision_score(detector.y_test, y_pred)
            metrics['recall'] = recall_score(detector.y_test, y_pred)
            metrics['f1_score'] = f1_score(detector.y_test, y_pred)
            metrics['roc_auc'] = roc_auc_score(detector.y_test, y_pred_proba)
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(detector.y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Feature importance
            coefficients = detector.model.coef_[0]
            feature_names = detector.feature_names
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
            spam_indicators = feature_importance[feature_importance['coefficient'] > 0].head(15).to_dict('records')
            ham_indicators = feature_importance[feature_importance['coefficient'] < 0].head(15).to_dict('records')
            
            metrics['feature_importance'] = {
                'spam_indicators': spam_indicators,
                'ham_indicators': ham_indicators
            }
            
            metrics_data['detector'] = detector
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
    
    return metrics

def calculate_metrics():
    """Calculate current metrics"""
    global metrics_data
    
    # Load base metrics
    base_metrics = load_model_metrics()
    
    # Add inference simulation if detector is available
    inference_stats = None
    if metrics_data.get('detector'):
        detector = metrics_data['detector']
        import time
        
        # Sample emails for inference test
        sample_size = min(1000, len(detector.X_test))
        sample_indices = np.random.choice(len(detector.X_test), size=sample_size, replace=False)
        X_sample = detector.X_test[sample_indices]
        
        start_time = time.time()
        predictions = detector.model.predict(X_sample)
        inference_time = time.time() - start_time
        
        emails_per_second = sample_size / inference_time if inference_time > 0 else 0
        latency_per_email = (inference_time / sample_size) * 1000 if sample_size > 0 else 0
        
        inference_stats = {
            'emails_per_second': emails_per_second,
            'latency_ms': latency_per_email,
            'total_emails': sample_size,
            'spam_detected': int(predictions.sum())
        }
    
    return {
        'model_metrics': {
            'accuracy': base_metrics.get('accuracy', 0),
            'precision': base_metrics.get('precision', 0),
            'recall': base_metrics.get('recall', 0),
            'f1_score': base_metrics.get('f1_score', 0),
            'roc_auc': base_metrics.get('roc_auc', 0)
        },
        'confusion_matrix': base_metrics.get('confusion_matrix', [[0, 0], [0, 0]]),
        'feature_importance': base_metrics.get('feature_importance', {'spam_indicators': [], 'ham_indicators': []}),
        'inference_stats': inference_stats,
        'timestamp': datetime.now().isoformat()
    }

def update_metrics_thread():
    """Background thread to continuously update metrics"""
    global metrics_data
    
    while metrics_data['is_running']:
        try:
            metrics = calculate_metrics()
            metrics_data['model_metrics'] = metrics['model_metrics']
            metrics_data['feature_importance'] = metrics['feature_importance']
            metrics_data['inference_stats'] = metrics['inference_stats']
            metrics_data['last_update'] = datetime.now().isoformat()
            metrics_data['current_metrics'] = metrics
            
            time.sleep(3)  # Update every 3 seconds
        except Exception as e:
            print(f"Error updating metrics: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics (REST API)"""
    if not metrics_data.get('current_metrics'):
        metrics = calculate_metrics()
        metrics_data['current_metrics'] = metrics
        metrics_data['last_update'] = datetime.now().isoformat()
    
    return jsonify(metrics_data['current_metrics'])

@app.route('/api/metrics/stream')
def stream_metrics():
    """Server-Sent Events stream for real-time updates"""
    def event_stream():
        while True:
            if metrics_data.get('is_running') and metrics_data.get('current_metrics'):
                data = {
                    **metrics_data['current_metrics'],
                    'last_update': metrics_data['last_update'],
                    'timestamp': datetime.now().isoformat()
                }
            elif metrics_data.get('current_metrics'):
                data = {
                    **metrics_data['current_metrics'],
                    'last_update': metrics_data['last_update'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                metrics = calculate_metrics()
                data = {
                    **metrics,
                    'last_update': datetime.now().isoformat(),
                    'timestamp': datetime.now().isoformat()
                }
            
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(3)
    
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
        'last_update': metrics_data['last_update']
    })

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests to prevent 404 errors"""
    from flask import Response
    # Return empty 204 No Content response
    return Response(status=204)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors gracefully"""
    # For API routes, return JSON
    from flask import request
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found', 'path': request.path}), 404
    # For other routes, redirect to home
    from flask import redirect
    return redirect('/')

if __name__ == '__main__':
    import pandas as pd
    
    # Initialize with default data (with error handling)
    try:
        metrics = calculate_metrics()
        metrics_data['current_metrics'] = metrics
        metrics_data['last_update'] = datetime.now().isoformat()
        print("✅ Metrics loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not load initial metrics: {e}")
        print("   Dashboard will start with empty metrics. Run lesson_code.py first to train the model.")
        # Set default empty metrics
        metrics_data['current_metrics'] = {
            'model_metrics': {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'roc_auc': 0
            },
            'confusion_matrix': [[0, 0], [0, 0]],
            'feature_importance': {'spam_indicators': [], 'ham_indicators': []},
            'inference_stats': None,
            'timestamp': datetime.now().isoformat()
        }
        metrics_data['last_update'] = datetime.now().isoformat()
    
    print("🚀 Starting Spam Detection Dashboard Server...")
    print("📊 Access the dashboard at: http://localhost:5000")
    print("🔄 Metrics update in real-time every 3 seconds")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()

