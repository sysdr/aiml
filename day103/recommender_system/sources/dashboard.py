"""
Dashboard for Day 103: Recommender Systems Theory
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
    CollaborativeFilteringEngine,
    ContentBasedEngine,
    HybridRecommender,
    create_sample_dataset
)

app = Flask(__name__)
CORS(app)

# Global state
interactions_df = None
item_features_df = None
cf_engine = None
cb_engine = None
hybrid_engine = None

metrics_data = {
    'timestamp': datetime.now().isoformat(),
    'collaborative_metrics': {
        'users_processed': 0,
        'recommendations_generated': 0,
        'avg_similarity_score': 0.0,
        'matrix_size': '0x0'
    },
    'content_metrics': {
        'items_processed': 0,
        'user_profiles_created': 0,
        'avg_similarity_score': 0.0,
        'features_count': 0
    },
    'hybrid_metrics': {
        'recommendations_generated': 0,
        'cf_weight': 0.6,
        'cb_weight': 0.4,
        'avg_combined_score': 0.0
    },
    'demo_metrics': {
        'demos_run': 0,
        'last_demo_time': None,
        'success_rate': 100.0
    }
}

def initialize_engines():
    """Initialize recommender engines with sample data"""
    global interactions_df, item_features_df, cf_engine, cb_engine, hybrid_engine
    
    if interactions_df is None:
        interactions_df, item_features_df = create_sample_dataset()
        
        cf_engine = CollaborativeFilteringEngine()
        cf_engine.fit(interactions_df)
        
        cb_engine = ContentBasedEngine()
        cb_engine.fit(item_features_df, interactions_df)
        
        hybrid_engine = HybridRecommender(collaborative_weight=0.6, content_weight=0.4)
        hybrid_engine.fit(interactions_df, item_features_df)
        
        # Update metrics
        metrics_data['collaborative_metrics']['users_processed'] = len(cf_engine.user_item_matrix)
        metrics_data['collaborative_metrics']['matrix_size'] = f"{len(cf_engine.user_item_matrix)}x{len(cf_engine.user_item_matrix.columns)}"
        metrics_data['content_metrics']['items_processed'] = len(item_features_df)
        metrics_data['content_metrics']['user_profiles_created'] = len(cb_engine.user_profiles)
        metrics_data['content_metrics']['features_count'] = len(item_features_df.columns) - 1

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Day 103: Recommender Systems Dashboard</title>
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
        <h1>🎯 Recommender Systems Dashboard</h1>
        
        <div class="control-panel">
            <div class="control-title">🎮 Demo Controls</div>
            <div class="demo-buttons">
                <button class="demo-btn" onclick="runCollaborativeFiltering()">
                    👥 Run Collaborative Filtering
                </button>
                <button class="demo-btn" onclick="runContentBased()">
                    📝 Run Content-Based Filtering
                </button>
                <button class="demo-btn" onclick="runHybrid()">
                    🔀 Run Hybrid Recommender
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
                    
                    // Collaborative Filtering Metrics
                    const cfCard = createMetricCard(
                        'Collaborative Filtering',
                        data.collaborative_metrics.users_processed,
                        `Matrix: ${data.collaborative_metrics.matrix_size}`,
                        '👥'
                    );
                    grid.appendChild(cfCard);
                    
                    // Content-Based Metrics
                    const cbCard = createMetricCard(
                        'Content-Based',
                        data.content_metrics.items_processed,
                        `Features: ${data.content_metrics.features_count}`,
                        '📝'
                    );
                    grid.appendChild(cbCard);
                    
                    // Hybrid Metrics
                    const hybridCard = createMetricCard(
                        'Hybrid Recommender',
                        data.hybrid_metrics.recommendations_generated,
                        `CF: ${data.hybrid_metrics.cf_weight}, CB: ${data.hybrid_metrics.cb_weight}`,
                        '🔀'
                    );
                    grid.appendChild(hybridCard);
                    
                    // User Profiles
                    const profilesCard = createMetricCard(
                        'User Profiles',
                        data.content_metrics.user_profiles_created,
                        'Content-based profiles created',
                        '👤'
                    );
                    grid.appendChild(profilesCard);
                    
                    // Recommendations Generated
                    const recsCard = createMetricCard(
                        'Recommendations',
                        data.collaborative_metrics.recommendations_generated,
                        'Total recommendations generated',
                        '🎯'
                    );
                    grid.appendChild(recsCard);
                    
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
        
        function runCollaborativeFiltering() {
            showStatus('Running collaborative filtering demo...');
            fetch('/api/run-collaborative', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`✅ Collaborative filtering complete! Generated ${data.results.recommendations} recommendations`);
                    updateDashboard();
                } else {
                    showStatus('❌ Demo failed: ' + data.message, true);
                }
            })
            .catch(error => {
                showStatus('❌ Error running demo: ' + error.message, true);
            });
        }
        
        function runContentBased() {
            showStatus('Running content-based filtering demo...');
            fetch('/api/run-content-based', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`✅ Content-based filtering complete! Generated ${data.results.recommendations} recommendations`);
                    updateDashboard();
                } else {
                    showStatus('❌ Demo failed: ' + data.message, true);
                }
            })
            .catch(error => {
                showStatus('❌ Error running demo: ' + error.message, true);
            });
        }
        
        function runHybrid() {
            showStatus('Running hybrid recommender demo...');
            fetch('/api/run-hybrid', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`✅ Hybrid recommender complete! Generated ${data.results.recommendations} recommendations`);
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

@app.route('/api/run-collaborative', methods=['POST'])
def run_collaborative():
    """Run collaborative filtering demo"""
    try:
        initialize_engines()
        result = cf_engine.recommend(user_id=5, top_n=5)
        
        metrics_data['collaborative_metrics']['recommendations_generated'] += len(result.recommended_items)
        if result.scores:
            metrics_data['collaborative_metrics']['avg_similarity_score'] = np.mean(result.scores)
        
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

@app.route('/api/run-content-based', methods=['POST'])
def run_content_based():
    """Run content-based filtering demo"""
    try:
        initialize_engines()
        user_rated_items = set(interactions_df[interactions_df['user_id'] == 5]['item_id'])
        result = cb_engine.recommend(user_id=5, top_n=5, exclude_items=user_rated_items)
        
        metrics_data['content_metrics']['items_processed'] = len(item_features_df)
        if result.scores:
            metrics_data['content_metrics']['avg_similarity_score'] = np.mean(result.scores)
        
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

@app.route('/api/run-hybrid', methods=['POST'])
def run_hybrid():
    """Run hybrid recommender demo"""
    try:
        initialize_engines()
        result = hybrid_engine.recommend(user_id=5, top_n=5)
        
        metrics_data['hybrid_metrics']['recommendations_generated'] += len(result.recommended_items)
        if result.scores:
            metrics_data['hybrid_metrics']['avg_combined_score'] = np.mean(result.scores)
        
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
    print("🚀 Starting Recommender Systems Dashboard on http://localhost:5000")
    print("📊 Access dashboard at: http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/api/metrics")
    app.run(host='0.0.0.0', port=5000, debug=True)
