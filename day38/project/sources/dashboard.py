"""
Dashboard for Day 38: Machine Learning Workflow Metrics
"""

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import json
import time
from datetime import datetime
import sys
import os

# Add parent directory to path to import lesson_code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lesson_code import MLWorkflowPipeline

app = Flask(__name__)
CORS(app)

# Global state
pipeline = None
metrics_data = {
    'timestamp': datetime.now().isoformat(),
    'workflow_metrics': {
        'stages_completed': 0,
        'total_stages': 7,
        'current_stage': None,
        'workflow_status': 'not_started'
    },
    'model_metrics': {
        'model_trained': False,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'predictions_made': 0
    },
    'data_metrics': {
        'total_samples': 0,
        'train_samples': 0,
        'test_samples': 0,
        'features': 0
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
    <title>Day 38: ML Workflow Dashboard</title>
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
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
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
        .status-pending { background-color: #ff9800; }
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
        .stage-list {
            list-style: none;
            padding: 0;
        }
        .stage-item {
            padding: 8px;
            margin: 5px 0;
            border-radius: 5px;
            background: #f5f5f5;
        }
        .stage-completed {
            background: #d4edda;
            color: #155724;
        }
        .stage-current {
            background: #fff3cd;
            color: #856404;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 ML Workflow Dashboard</h1>
        
        <div class="control-panel">
            <div class="control-title">🎮 Demo Controls</div>
            <div class="demo-buttons">
                <button class="demo-btn" onclick="runWorkflowDemo()">
                    🚀 Run Complete Workflow Demo
                </button>
                <button class="demo-btn" onclick="runPredictionDemo()">
                    🔮 Run Prediction Demo
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
        const stages = [
            'Problem Definition',
            'Data Collection',
            'Data Preparation',
            'Model Training',
            'Model Evaluation',
            'Deployment',
            'Monitoring'
        ];
        
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    const grid = document.getElementById('metrics-grid');
                    grid.innerHTML = '';
                    
                    // Workflow Progress
                    const progress = (data.workflow_metrics.stages_completed / data.workflow_metrics.total_stages) * 100;
                    const workflowCard = createMetricCard(
                        'Workflow Progress',
                        `${data.workflow_metrics.stages_completed}/${data.workflow_metrics.total_stages} Stages`,
                        data.workflow_metrics.current_stage || 'Not started',
                        '📋',
                        `<div class="progress-bar">
                            <div class="progress-fill" style="width: ${progress}%">${Math.round(progress)}%</div>
                        </div>`
                    );
                    grid.appendChild(workflowCard);
                    
                    // Model Accuracy
                    const accuracyCard = createMetricCard(
                        'Model Accuracy',
                        data.model_metrics.accuracy > 0 ? (data.model_metrics.accuracy * 100).toFixed(1) + '%' : 'N/A',
                        data.model_metrics.model_trained ? 'Model trained' : 'No model yet',
                        '🎯'
                    );
                    grid.appendChild(accuracyCard);
                    
                    // F1 Score
                    const f1Card = createMetricCard(
                        'F1 Score',
                        data.model_metrics.f1_score > 0 ? data.model_metrics.f1_score.toFixed(3) : 'N/A',
                        'Balanced metric',
                        '📊'
                    );
                    grid.appendChild(f1Card);
                    
                    // Precision
                    const precisionCard = createMetricCard(
                        'Precision',
                        data.model_metrics.precision > 0 ? data.model_metrics.precision.toFixed(3) : 'N/A',
                        'True positives / (TP + FP)',
                        '✅'
                    );
                    grid.appendChild(precisionCard);
                    
                    // Recall
                    const recallCard = createMetricCard(
                        'Recall',
                        data.model_metrics.recall > 0 ? data.model_metrics.recall.toFixed(3) : 'N/A',
                        'True positives / (TP + FN)',
                        '🔍'
                    );
                    grid.appendChild(recallCard);
                    
                    // Data Metrics
                    const dataCard = createMetricCard(
                        'Data Samples',
                        data.data_metrics.total_samples || 0,
                        `Train: ${data.data_metrics.train_samples}, Test: ${data.data_metrics.test_samples}`,
                        '📦'
                    );
                    grid.appendChild(dataCard);
                    
                    // Features
                    const featuresCard = createMetricCard(
                        'Features',
                        data.data_metrics.features || 0,
                        'Feature dimensions',
                        '🔢'
                    );
                    grid.appendChild(featuresCard);
                    
                    // Predictions
                    const predsCard = createMetricCard(
                        'Predictions Made',
                        data.model_metrics.predictions_made || 0,
                        'Total predictions',
                        '🔮'
                    );
                    grid.appendChild(predsCard);
                    
                    // Demos Run
                    const demosCard = createMetricCard(
                        'Demos Run',
                        data.demo_metrics.demos_run || 0,
                        `Success Rate: ${data.demo_metrics.success_rate.toFixed(1)}%`,
                        '🚀'
                    );
                    grid.appendChild(demosCard);
                    
                    document.getElementById('timestamp').textContent = 
                        `Last updated: ${new Date(data.timestamp).toLocaleString()}`;
                })
                .catch(error => {
                    console.error('Error fetching metrics:', error);
                });
        }
        
        function createMetricCard(title, value, label, emoji, extra = '') {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <div class="metric-title">${emoji} ${title}</div>
                <div class="metric-value">${value}</div>
                <div class="metric-label">${label}</div>
                ${extra}
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
        
        function runWorkflowDemo() {
            showStatus('Running complete workflow demo...');
            fetch('/api/run-workflow-demo', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus('✅ Workflow demo complete! Check metrics below.');
                    updateDashboard();
                } else {
                    showStatus('❌ Demo failed: ' + data.message, true);
                }
            })
            .catch(error => {
                showStatus('❌ Error running demo: ' + error.message, true);
            });
        }
        
        function runPredictionDemo() {
            showStatus('Running prediction demo...');
            fetch('/api/run-prediction-demo', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus('✅ Prediction demo complete!');
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
    global pipeline
    
    # Update metrics if pipeline exists
    if pipeline and pipeline.model is not None:
        metrics_data['model_metrics']['model_trained'] = True
        if pipeline.metrics:
            metrics_data['model_metrics']['accuracy'] = pipeline.metrics.get('accuracy', 0.0)
            metrics_data['model_metrics']['precision'] = pipeline.metrics.get('precision', 0.0)
            metrics_data['model_metrics']['recall'] = pipeline.metrics.get('recall', 0.0)
            metrics_data['model_metrics']['f1_score'] = pipeline.metrics.get('f1_score', 0.0)
    
    metrics_data['timestamp'] = datetime.now().isoformat()
    return jsonify(metrics_data)

@app.route('/api/update-demo', methods=['POST'])
def update_demo_metrics():
    """Update demo metrics"""
    metrics_data['demo_metrics']['demos_run'] += 1
    metrics_data['demo_metrics']['last_demo_time'] = datetime.now().isoformat()
    return jsonify({'status': 'success'})

@app.route('/api/run-workflow-demo', methods=['POST'])
def run_workflow_demo():
    """Run complete workflow demo and update metrics"""
    global pipeline
    try:
        pipeline = MLWorkflowPipeline()
        
        # Stage 1: Problem Definition
        pipeline.define_problem()
        metrics_data['workflow_metrics']['stages_completed'] = 1
        metrics_data['workflow_metrics']['current_stage'] = 'Data Collection'
        
        # Stage 2: Data Collection
        df = pipeline.collect_data()
        metrics_data['data_metrics']['total_samples'] = len(df)
        metrics_data['workflow_metrics']['stages_completed'] = 2
        metrics_data['workflow_metrics']['current_stage'] = 'Data Preparation'
        
        # Stage 3: Data Preparation
        X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
        metrics_data['data_metrics']['train_samples'] = len(y_train)
        metrics_data['data_metrics']['test_samples'] = len(y_test)
        metrics_data['data_metrics']['features'] = X_train.shape[1]
        metrics_data['workflow_metrics']['stages_completed'] = 3
        metrics_data['workflow_metrics']['current_stage'] = 'Model Training'
        
        # Stage 4: Model Training
        pipeline.train_model(X_train, y_train)
        metrics_data['workflow_metrics']['stages_completed'] = 4
        metrics_data['workflow_metrics']['current_stage'] = 'Model Evaluation'
        
        # Stage 5: Model Evaluation
        metrics = pipeline.evaluate_model(X_test, y_test)
        metrics_data['model_metrics']['model_trained'] = True
        metrics_data['model_metrics']['accuracy'] = metrics['accuracy']
        metrics_data['model_metrics']['precision'] = metrics['precision']
        metrics_data['model_metrics']['recall'] = metrics['recall']
        metrics_data['model_metrics']['f1_score'] = metrics['f1_score']
        metrics_data['workflow_metrics']['stages_completed'] = 5
        metrics_data['workflow_metrics']['current_stage'] = 'Deployment'
        
        # Stage 6: Deployment
        pipeline.deploy_model()
        metrics_data['workflow_metrics']['stages_completed'] = 6
        metrics_data['workflow_metrics']['current_stage'] = 'Monitoring'
        
        # Stage 7: Monitoring
        new_reviews = [
            "This product is incredible! Love it so much.",
            "Terrible quality. Very disappointed."
        ]
        results = pipeline.predict(new_reviews)
        metrics_data['model_metrics']['predictions_made'] += len(results)
        metrics_data['workflow_metrics']['stages_completed'] = 7
        metrics_data['workflow_metrics']['current_stage'] = 'Complete'
        metrics_data['workflow_metrics']['workflow_status'] = 'completed'
        
        # Update demo counter
        metrics_data['demo_metrics']['demos_run'] += 1
        metrics_data['demo_metrics']['last_demo_time'] = datetime.now().isoformat()
        
        return jsonify({
            'status': 'success',
            'results': {
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'predictions': len(results)
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/run-prediction-demo', methods=['POST'])
def run_prediction_demo():
    """Run prediction demo (requires trained model)"""
    global pipeline
    try:
        if pipeline is None or pipeline.model is None:
            # Run quick workflow first
            pipeline = MLWorkflowPipeline()
            df = pipeline.collect_data()
            X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
            pipeline.train_model(X_train, y_train)
            pipeline.evaluate_model(X_test, y_test)
        
        # Make predictions
        new_reviews = [
            "Excellent product! Very happy with purchase.",
            "Terrible quality. Would not recommend.",
            "Good value for money. Works as expected."
        ]
        results = pipeline.predict(new_reviews)
        metrics_data['model_metrics']['predictions_made'] += len(results)
        metrics_data['demo_metrics']['demos_run'] += 1
        metrics_data['demo_metrics']['last_demo_time'] = datetime.now().isoformat()
        
        return jsonify({
            'status': 'success',
            'results': {
                'predictions': len(results),
                'reviews': [r['sentiment'] for r in results]
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    print("🚀 Starting ML Workflow Dashboard on http://localhost:5000")
    print("📊 Access dashboard at: http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/api/metrics")
    app.run(host='0.0.0.0', port=5000, debug=True)

