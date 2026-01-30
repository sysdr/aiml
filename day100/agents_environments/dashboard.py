"""
Dashboard for Day 100: Agents, Environments, and Rewards
Real-time metrics visualization and monitoring
"""

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Metrics storage
metrics = {
    'episode_rewards': [],
    'episode_lengths': [],
    'total_episodes': 0,
    'mean_reward': 0.0,
    'mean_length': 0.0,
    'last_update': None
}


@app.route('/')
def index():
    """Dashboard home page"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Day 100 RL Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .metric-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #007bff;
            }
            .metric-label {
                color: #666;
                margin-top: 5px;
            }
            .chart-container {
                margin: 20px 0;
                height: 400px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Day 100: Agents, Environments, and Rewards - Dashboard</h1>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value" id="total-episodes">0</div>
                    <div class="metric-label">Total Episodes</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="mean-reward">0.00</div>
                    <div class="metric-label">Mean Reward</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="mean-length">0.00</div>
                    <div class="metric-label">Mean Episode Length</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="last-update">-</div>
                    <div class="metric-label">Last Update</div>
                </div>
            </div>
            
            <div class="chart-container">
                <canvas id="rewardsChart"></canvas>
            </div>
            
            <div class="chart-container">
                <canvas id="lengthsChart"></canvas>
            </div>
        </div>
        
        <script>
            const rewardsCtx = document.getElementById('rewardsChart').getContext('2d');
            const lengthsCtx = document.getElementById('lengthsChart').getContext('2d');
            
            const rewardsChart = new Chart(rewardsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Episode Rewards',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            const lengthsChart = new Chart(lengthsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Episode Lengths',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            function updateDashboard() {
                fetch('/api/metrics')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('total-episodes').textContent = data.total_episodes;
                        document.getElementById('mean-reward').textContent = data.mean_reward.toFixed(2);
                        document.getElementById('mean-length').textContent = data.mean_length.toFixed(2);
                        document.getElementById('last-update').textContent = 
                            data.last_update ? new Date(data.last_update).toLocaleTimeString() : '-';
                        
                        if (data.episode_rewards && data.episode_rewards.length > 0) {
                            const labels = data.episode_rewards.map((_, i) => i + 1);
                            rewardsChart.data.labels = labels;
                            rewardsChart.data.datasets[0].data = data.episode_rewards;
                            rewardsChart.update();
                        }
                        
                        if (data.episode_lengths && data.episode_lengths.length > 0) {
                            const labels = data.episode_lengths.map((_, i) => i + 1);
                            lengthsChart.data.labels = labels;
                            lengthsChart.data.datasets[0].data = data.episode_lengths;
                            lengthsChart.update();
                        }
                    })
                    .catch(error => console.error('Error fetching metrics:', error));
            }
            
            // Update every second
            setInterval(updateDashboard, 1000);
            updateDashboard();
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    # Try to load metrics from file if it exists
    metrics_file = 'metrics.json'
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                file_metrics = json.load(f)
                metrics.update(file_metrics)
        except:
            pass
    
    return jsonify(metrics)


@app.route('/api/update', methods=['POST'])
def update_metrics():
    """Update metrics (called by lesson code)"""
    # This would be called by the lesson code to update metrics
    # For now, we'll read from a metrics file
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("Starting Day 100 RL Dashboard on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
