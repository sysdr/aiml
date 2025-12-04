"""
Dashboard for Day 37: AI, ML, and Deep Learning Metrics
"""

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import json
import time
from datetime import datetime
import numpy as np
from lesson_code import AIMLIntroduction

app = Flask(__name__)
CORS(app)

# Global state
intro = AIMLIntroduction()
metrics_data = {
    'timestamp': datetime.now().isoformat(),
    'ai_metrics': {
        'total_concepts': 3,
        'concepts_covered': ['AI', 'ML', 'DL'],
        'understanding_level': 85
    },
    'ml_metrics': {
        'models_trained': 0,
        'avg_r2_score': 0.0,
        'avg_mse': 0.0,
        'predictions_made': 0
    },
    'demo_metrics': {
        'demos_run': 0,
        'last_demo_time': None,
        'success_rate': 100.0
    }
}

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Day 37: AI/ML Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-title {
            font-size: 1.2em;
            color: #667eea;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .metric-value {
            font-size: 2.5em;
            color: #333;
            font-weight: bold;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
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
        .timestamp {
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 0.9em;
        }
        .control-panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .control-title {
            font-size: 1.5em;
            color: #667eea;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .demo-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .demo-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            font-weight: 500;
        }
        .demo-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .demo-btn:active {
            transform: translateY(0);
        }
        .demo-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .status-message {
            margin-top: 15px;
            padding: 10px;
            border-radius: 5px;
            display: none;
        }
        .status-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI/ML Learning Dashboard</h1>
        
        <div class="control-panel">
            <div class="control-title">🎮 Demo Controls</div>
            <div class="demo-buttons">
                <button class="demo-btn" onclick="runSupervisedLearning()">
                    🧠 Run Supervised Learning Demo
                </button>
                <button class="demo-btn" onclick="updateDemo()">
                    📊 Update Demo Counter
                </button>
            </div>
            <div id="status-message" class="status-message"></div>
        </div>
        
        <div class="metrics-grid" id="metrics-grid">
            <!-- Metrics will be populated by JavaScript -->
        </div>
        <div class="timestamp" id="timestamp"></div>
    </div>
    <script>
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    const grid = document.getElementById('metrics-grid');
                    grid.innerHTML = '';
                    
                    // AI Metrics
                    const aiCard = createMetricCard(
                        'AI Concepts',
                        data.ai_metrics.total_concepts,
                        `Covered: ${data.ai_metrics.concepts_covered.join(', ')}`,
                        '🎯'
                    );
                    grid.appendChild(aiCard);
                    
                    // ML Metrics
                    const mlCard = createMetricCard(
                        'ML Models',
                        data.ml_metrics.models_trained,
                        `Avg R²: ${data.ml_metrics.avg_r2_score.toFixed(3)}`,
                        '🧠'
                    );
                    grid.appendChild(mlCard);
                    
                    // Demo Metrics
                    const demoCard = createMetricCard(
                        'Demos Run',
                        data.demo_metrics.demos_run,
                        `Success Rate: ${data.demo_metrics.success_rate.toFixed(1)}%`,
                        '🚀'
                    );
                    grid.appendChild(demoCard);
                    
                    // Understanding Level
                    const understandingCard = createMetricCard(
                        'Understanding',
                        data.ai_metrics.understanding_level + '%',
                        'Overall comprehension level',
                        '📊'
                    );
                    grid.appendChild(understandingCard);
                    
                    // MSE Card
                    const mseCard = createMetricCard(
                        'Model MSE',
                        data.ml_metrics.avg_mse > 0 ? 
                            '$' + data.ml_metrics.avg_mse.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",") : 
                            'N/A',
                        'Mean Squared Error',
                        '📉'
                    );
                    grid.appendChild(mseCard);
                    
                    // Predictions Card
                    const predsCard = createMetricCard(
                        'Predictions',
                        data.ml_metrics.predictions_made,
                        'Total predictions made',
                        '🎯'
                    );
                    grid.appendChild(predsCard);
                    
                    document.getElementById('timestamp').textContent = 
                        `Last updated: ${new Date(data.timestamp).toLocaleString()}`;
                })
                .catch(error => {
                    console.error('Error fetching metrics:', error);
                });
        }
        
        function createMetricCard(title, value, label, emoji) {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <div class="metric-title">${emoji} ${title}</div>
                <div class="metric-value">${value}</div>
                <div class="metric-label">${label}</div>
            `;
            return card;
        }
        
        function showStatus(message, isError = false) {
            const statusDiv = document.getElementById('status-message');
            statusDiv.textContent = message;
            statusDiv.className = 'status-message ' + (isError ? 'status-error' : 'status-success');
            statusDiv.style.display = 'block';
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }
        
        function runSupervisedLearning() {
            showStatus('Running supervised learning demo...');
            fetch('/api/run-supervised-learning', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    const r2 = data.results.r2_score.toFixed(3);
                    const mse = '$' + data.results.mse.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
                    showStatus(`✅ Demo complete! R²: ${r2}, MSE: ${mse}`);
                    updateDashboard();
                } else {
                    showStatus('❌ Demo failed: ' + data.message, true);
                }
            })
            .catch(error => {
                showStatus('❌ Error running demo: ' + error.message, true);
            });
        }
        
        function updateDemo() {
            fetch('/api/update-demo', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus('✅ Demo counter updated!');
                    updateDashboard();
                } else {
                    showStatus('❌ Failed to update demo counter', true);
                }
            })
            .catch(error => {
                showStatus('❌ Error: ' + error.message, true);
            });
        }
        
        // Update every 2 seconds
        updateDashboard();
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>
"""

@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon to prevent 404"""
    return '', 204

@app.route('/')
def dashboard():
    """Serve dashboard HTML"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    # Update metrics based on actual state
    if intro.models:
        ml_metrics = metrics_data['ml_metrics']
        ml_metrics['models_trained'] = len(intro.models)
        if intro.metrics:
            r2_scores = [m.get('r2', 0) for m in intro.metrics.values() if 'r2' in m]
            mse_scores = [m.get('mse', 0) for m in intro.metrics.values() if 'mse' in m]
            if r2_scores:
                ml_metrics['avg_r2_score'] = np.mean(r2_scores)
            if mse_scores:
                ml_metrics['avg_mse'] = np.mean(mse_scores)
    
    metrics_data['timestamp'] = datetime.now().isoformat()
    return jsonify(metrics_data)

@app.route('/api/update-demo', methods=['POST'])
def update_demo_metrics():
    """Update demo metrics"""
    metrics_data['demo_metrics']['demos_run'] += 1
    metrics_data['demo_metrics']['last_demo_time'] = datetime.now().isoformat()
    return jsonify({'status': 'success'})

@app.route('/api/run-supervised-learning', methods=['POST'])
def run_supervised_learning():
    """Run supervised learning demo and update metrics"""
    try:
        results = intro.supervised_learning_demo()
        
        # Update metrics
        ml_metrics = metrics_data['ml_metrics']
        ml_metrics['models_trained'] = len(intro.models)
        ml_metrics['predictions_made'] += 3  # We make 3 predictions in demo
        
        if intro.metrics:
            r2_scores = [m.get('r2', 0) for m in intro.metrics.values() if 'r2' in m]
            if r2_scores:
                ml_metrics['avg_r2_score'] = np.mean(r2_scores)
        
        return jsonify({
            'status': 'success',
            'results': {
                'r2_score': results['r2'],
                'mse': results['mse'],
                'predictions': results['predictions']
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Dashboard on http://localhost:5000")
    print("📊 Access dashboard at: http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/api/metrics")
    app.run(host='0.0.0.0', port=5000, debug=True)
