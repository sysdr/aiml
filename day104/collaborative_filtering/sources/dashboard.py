"""
Dashboard for Day 104: Collaborative Filtering
"""

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import json
import time
import sys
import os
from datetime import datetime
import numpy as np

# Add parent directory to path to import lesson_code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lesson_code import (
    CollaborativeFiltering,
    create_sample_dataset
)

app = Flask(__name__)
CORS(app)

# Global state
interactions_df = None
cf_user_engine = None
cf_item_engine = None

metrics_data = {
    'timestamp': datetime.now().isoformat(),
    'user_based_metrics': {
        'users_processed': 0,
        'recommendations_generated': 0,
        'avg_similarity_score': 0.0,
        'matrix_size': '0x0'
    },
    'item_based_metrics': {
        'items_processed': 0,
        'recommendations_generated': 0,
        'avg_similarity_score': 0.0,
        'matrix_size': '0x0'
    },
    'demo_metrics': {
        'demos_run': 0,
        'last_demo_time': None,
        'success_rate': 100.0
    }
}

def initialize_engines():
    """Initialize collaborative filtering engines with sample data"""
    global interactions_df, cf_user_engine, cf_item_engine
    
    if interactions_df is None:
        interactions_df = create_sample_dataset()
        
        cf_user_engine = CollaborativeFiltering(method='user-based')
        cf_user_engine.fit(interactions_df)
        
        cf_item_engine = CollaborativeFiltering(method='item-based')
        cf_item_engine.fit(interactions_df)
        
        # Update metrics
        metrics_data['user_based_metrics']['users_processed'] = len(cf_user_engine.user_item_matrix)
        metrics_data['user_based_metrics']['matrix_size'] = f"{len(cf_user_engine.user_item_matrix)}x{len(cf_user_engine.user_item_matrix.columns)}"
        metrics_data['item_based_metrics']['items_processed'] = len(cf_item_engine.item_user_matrix)
        metrics_data['item_based_metrics']['matrix_size'] = f"{len(cf_item_engine.item_user_matrix)}x{len(cf_item_engine.item_user_matrix.columns)}"

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Day 104: Collaborative Filtering Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
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
        <h1>🎯 Collaborative Filtering Dashboard</h1>
        
        <div class="control-panel">
            <div class="control-title">🎮 Demo Controls</div>
            <div class="demo-buttons">
                <button class="demo-btn" onclick="runUserBased()">
                    👥 Run User-Based CF
                </button>
                <button class="demo-btn" onclick="runItemBased()">
                    📝 Run Item-Based CF
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
                    
                    // User-Based Metrics
                    const ubCard = createMetricCard(
                        'User-Based CF',
                        data.user_based_metrics.users_processed,
                        `Matrix: ${data.user_based_metrics.matrix_size}`,
                        '👥'
                    );
                    grid.appendChild(ubCard);
                    
                    // Item-Based Metrics
                    const ibCard = createMetricCard(
                        'Item-Based CF',
                        data.item_based_metrics.items_processed,
                        `Matrix: ${data.item_based_metrics.matrix_size}`,
                        '📝'
                    );
                    grid.appendChild(ibCard);
                    
                    // User-Based Recommendations
                    const ubRecsCard = createMetricCard(
                        'User-Based Recommendations',
                        data.user_based_metrics.recommendations_generated,
                        `Avg Score: ${data.user_based_metrics.avg_similarity_score.toFixed(3)}`,
                        '🎯'
                    );
                    grid.appendChild(ubRecsCard);
                    
                    // Item-Based Recommendations
                    const ibRecsCard = createMetricCard(
                        'Item-Based Recommendations',
                        data.item_based_metrics.recommendations_generated,
                        `Avg Score: ${data.item_based_metrics.avg_similarity_score.toFixed(3)}`,
                        '🎯'
                    );
                    grid.appendChild(ibRecsCard);
                    
                    // Demo Metrics
                    const demoCard = createMetricCard(
                        'Demos Run',
                        data.demo_metrics.demos_run,
                        `Success Rate: ${data.demo_metrics.success_rate.toFixed(1)}%`,
                        '🚀'
                    );
                    grid.appendChild(demoCard);
                    
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
        
        function runUserBased() {
            showStatus('Running user-based collaborative filtering demo...');
            fetch('/api/run-user-based', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`✅ User-based CF complete! Generated ${data.results.recommendations} recommendations`);
                    updateDashboard();
                } else {
                    showStatus('❌ Demo failed: ' + data.message, true);
                }
            })
            .catch(error => {
                showStatus('❌ Error running demo: ' + error.message, true);
            });
        }
        
        function runItemBased() {
            showStatus('Running item-based collaborative filtering demo...');
            fetch('/api/run-item-based', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`✅ Item-based CF complete! Generated ${data.results.recommendations} recommendations`);
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
    initialize_engines()
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    initialize_engines()
    metrics_data['timestamp'] = datetime.now().isoformat()
    return jsonify(metrics_data)

@app.route('/api/update-demo', methods=['POST'])
def update_demo_metrics():
    """Update demo metrics"""
    metrics_data['demo_metrics']['demos_run'] += 1
    metrics_data['demo_metrics']['last_demo_time'] = datetime.now().isoformat()
    return jsonify({'status': 'success'})

@app.route('/api/run-user-based', methods=['POST'])
def run_user_based():
    """Run user-based collaborative filtering demo"""
    try:
        initialize_engines()
        result = cf_user_engine.recommend(user_id=5, top_n=5)
        
        metrics_data['user_based_metrics']['recommendations_generated'] += len(result.recommended_items)
        if result.scores:
            metrics_data['user_based_metrics']['avg_similarity_score'] = np.mean(result.scores)
        
        return jsonify({
            'status': 'success',
            'results': {
                'recommendations': len(result.recommended_items),
                'items': result.recommended_items,
                'scores': result.scores
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/run-item-based', methods=['POST'])
def run_item_based():
    """Run item-based collaborative filtering demo"""
    try:
        initialize_engines()
        result = cf_item_engine.recommend(user_id=5, top_n=5)
        
        metrics_data['item_based_metrics']['recommendations_generated'] += len(result.recommended_items)
        if result.scores:
            metrics_data['item_based_metrics']['avg_similarity_score'] = np.mean(result.scores)
        
        return jsonify({
            'status': 'success',
            'results': {
                'recommendations': len(result.recommended_items),
                'items': result.recommended_items,
                'scores': result.scores
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Collaborative Filtering Dashboard on http://localhost:5000")
    print("📊 Access dashboard at: http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/api/metrics")
    app.run(host='0.0.0.0', port=5000, debug=True)
