#!/usr/bin/env python3
"""
Real-time Dashboard for Day 101: Q-Learning Algorithm
Run with: python dashboard.py
"""

import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify
import numpy as np
from lesson_code import GridWorld, QLearningAgent, train_agent, evaluate_agent

app = Flask(__name__)

# Global state for metrics
metrics_data = {
    'training_metrics': None,
    'evaluation_metrics': None,
    'q_table_stats': None,
    'policy_info': None,
    'last_update': None,
    'is_running': False
}

def calculate_qlearning_metrics():
    """Calculate Q-Learning demonstration metrics"""
    try:
        # Create environment
        env = GridWorld(size=5, obstacles=[(1, 1), (2, 3), (3, 1)])
        
        # Create agent
        agent = QLearningAgent(
            n_actions=env.n_actions,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.3,
            epsilon_decay=0.995,
            epsilon_min=0.05
        )
        
        # Train for a limited number of episodes for dashboard
        stats = train_agent(env, agent, n_episodes=2000, verbose=False)
        
        # Evaluate agent
        eval_results = evaluate_agent(env, agent, n_episodes=50)
        
        # Get policy
        policy = agent.get_policy()
        
        # Calculate Q-table statistics
        q_values = list(agent.q_table.values())
        if q_values:
            q_mean = float(np.mean(q_values))
            q_std = float(np.std(q_values))
            q_max = float(np.max(q_values))
            q_min = float(np.min(q_values))
        else:
            q_mean = q_std = q_max = q_min = 0.0
        
        # Recent training performance
        recent_rewards = stats['episode_rewards'][-100:] if len(stats['episode_rewards']) >= 100 else stats['episode_rewards']
        recent_steps = stats['episode_steps'][-100:] if len(stats['episode_steps']) >= 100 else stats['episode_steps']
        
        # Calculate convergence metrics
        if len(stats['q_value_changes']) >= 200:
            early_changes = np.mean(stats['q_value_changes'][:100])
            late_changes = np.mean(stats['q_value_changes'][-100:])
            convergence_ratio = float(late_changes / early_changes) if early_changes > 0 else 0.0
        else:
            convergence_ratio = 1.0
        
        return {
            'training': {
                'total_episodes': len(stats['episode_rewards']),
                'avg_reward': float(np.mean(recent_rewards)),
                'avg_steps': float(np.mean(recent_steps)),
                'current_epsilon': float(agent.epsilon),
                'convergence_ratio': convergence_ratio,
                'episode_rewards_trend': [float(r) for r in stats['episode_rewards'][-50:]],
                'episode_steps_trend': [float(s) for s in stats['episode_steps'][-50:]]
            },
            'evaluation': {
                'success_rate': float(eval_results['success_rate']),
                'avg_reward': float(eval_results['avg_reward']),
                'avg_steps': float(eval_results['avg_steps']),
                'std_reward': float(eval_results['std_reward']),
                'std_steps': float(eval_results['std_steps'])
            },
            'q_table': {
                'total_entries': len(agent.q_table),
                'mean_q_value': q_mean,
                'std_q_value': q_std,
                'max_q_value': q_max,
                'min_q_value': q_min,
                'unique_states': len(set(state for state, _ in agent.q_table.keys())),
                'coverage_pct': float(len(set(state for state, _ in agent.q_table.keys())) / (env.size * env.size) * 100)
            },
            'policy': {
                'policy_size': len(policy),
                'optimal_path_length': None,  # Would need to calculate
                'exploration_rate': float(agent.epsilon)
            },
            'environment': {
                'grid_size': env.size,
                'total_states': env.size * env.size,
                'num_obstacles': len(env.obstacles),
                'goal_position': list(env.goal),
                'start_position': list(env.start)
            },
            'agent_config': {
                'learning_rate': float(agent.alpha),
                'discount_factor': float(agent.gamma),
                'epsilon': float(agent.epsilon),
                'epsilon_decay': float(agent.epsilon_decay),
                'epsilon_min': float(agent.epsilon_min)
            }
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

def update_metrics():
    """Update metrics in background thread"""
    while metrics_data['is_running']:
        try:
            metrics = calculate_qlearning_metrics()
            if metrics:
                metrics_data['training_metrics'] = metrics.get('training')
                metrics_data['evaluation_metrics'] = metrics.get('evaluation')
                metrics_data['q_table_stats'] = metrics.get('q_table')
                metrics_data['policy_info'] = metrics.get('policy')
                metrics_data['environment'] = metrics.get('environment', {})
                metrics_data['agent_config'] = metrics.get('agent_config', {})
                metrics_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"Error updating metrics: {e}")
        time.sleep(5)  # Update every 5 seconds

@app.route('/')
def index():
    """Dashboard homepage"""
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Day 101: Q-Learning Algorithm Dashboard</title>
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
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
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
            font-size: 18px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 14px;
        }
        .improvement {
            color: #28a745;
            font-weight: bold;
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .section h2 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #667eea;
            color: white;
        }
        tr:hover {
            background: #f1f1f1;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .status.active {
            background: #28a745;
            color: white;
        }
        .last-update {
            text-align: right;
            color: #666;
            font-size: 12px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Day 101: Q-Learning Algorithm Dashboard</h1>
        <p class="subtitle">Real-time Reinforcement Learning Metrics & Performance</p>
        
        {% if training_metrics %}
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Training Performance</h3>
                <div class="metric-value">{{ "%.2f"|format(training_metrics.avg_reward) }}</div>
                <div class="metric-label">Average Reward (Last 100 Episodes)</div>
                <div class="metric-label">Episodes: {{ training_metrics.total_episodes }}</div>
            </div>
            
            <div class="metric-card">
                <h3>Success Rate</h3>
                <div class="metric-value">{{ "%.1f"|format(evaluation_metrics.success_rate * 100) }}%</div>
                <div class="metric-label">Goal Reaching Success</div>
                <div class="metric-label">Avg Steps: {{ "%.1f"|format(evaluation_metrics.avg_steps) }}</div>
            </div>
            
            <div class="metric-card">
                <h3>Q-Table Statistics</h3>
                <div class="metric-value">{{ q_table_stats.total_entries }}</div>
                <div class="metric-label">State-Action Pairs</div>
                <div class="metric-label">Coverage: {{ "%.1f"|format(q_table_stats.coverage_pct) }}%</div>
            </div>
            
            <div class="metric-card">
                <h3>Exploration Rate</h3>
                <div class="metric-value">{{ "%.3f"|format(training_metrics.current_epsilon) }}</div>
                <div class="metric-label">Current Epsilon (ε)</div>
                <div class="metric-label">Convergence: {{ "%.2f"|format(training_metrics.convergence_ratio) }}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Training Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Episodes</td>
                    <td>{{ training_metrics.total_episodes }}</td>
                </tr>
                <tr>
                    <td>Average Reward (Recent)</td>
                    <td>{{ "%.2f"|format(training_metrics.avg_reward) }}</td>
                </tr>
                <tr>
                    <td>Average Steps per Episode</td>
                    <td>{{ "%.2f"|format(training_metrics.avg_steps) }}</td>
                </tr>
                <tr>
                    <td>Current Exploration Rate (ε)</td>
                    <td>{{ "%.4f"|format(training_metrics.current_epsilon) }}</td>
                </tr>
                <tr>
                    <td>Convergence Ratio</td>
                    <td>{{ "%.3f"|format(training_metrics.convergence_ratio) }} 
                        {% if training_metrics.convergence_ratio < 0.5 %}
                        <span class="improvement">✓ Converging</span>
                        {% else %}
                        <span>Still Learning</span>
                        {% endif %}
                    </td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>🧪 Evaluation Results</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Success Rate</td>
                    <td><span class="improvement">{{ "%.2f"|format(evaluation_metrics.success_rate * 100) }}%</span></td>
                </tr>
                <tr>
                    <td>Average Reward</td>
                    <td>{{ "%.2f"|format(evaluation_metrics.avg_reward) }} ± {{ "%.2f"|format(evaluation_metrics.std_reward) }}</td>
                </tr>
                <tr>
                    <td>Average Steps to Goal</td>
                    <td>{{ "%.2f"|format(evaluation_metrics.avg_steps) }} ± {{ "%.2f"|format(evaluation_metrics.std_steps) }}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📈 Q-Table Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total State-Action Pairs</td>
                    <td>{{ q_table_stats.total_entries }}</td>
                </tr>
                <tr>
                    <td>Unique States Explored</td>
                    <td>{{ q_table_stats.unique_states }} / {{ environment.total_states }}</td>
                </tr>
                <tr>
                    <td>State Coverage</td>
                    <td>{{ "%.1f"|format(q_table_stats.coverage_pct) }}%</td>
                </tr>
                <tr>
                    <td>Mean Q-Value</td>
                    <td>{{ "%.2f"|format(q_table_stats.mean_q_value) }}</td>
                </tr>
                <tr>
                    <td>Max Q-Value</td>
                    <td>{{ "%.2f"|format(q_table_stats.max_q_value) }}</td>
                </tr>
                <tr>
                    <td>Min Q-Value</td>
                    <td>{{ "%.2f"|format(q_table_stats.min_q_value) }}</td>
                </tr>
                <tr>
                    <td>Q-Value Std Dev</td>
                    <td>{{ "%.2f"|format(q_table_stats.std_q_value) }}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>⚙️ Agent Configuration</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Learning Rate (α)</td>
                    <td>{{ agent_config.learning_rate }}</td>
                </tr>
                <tr>
                    <td>Discount Factor (γ)</td>
                    <td>{{ agent_config.discount_factor }}</td>
                </tr>
                <tr>
                    <td>Epsilon (ε)</td>
                    <td>{{ "%.4f"|format(agent_config.epsilon) }}</td>
                </tr>
                <tr>
                    <td>Epsilon Decay</td>
                    <td>{{ agent_config.epsilon_decay }}</td>
                </tr>
                <tr>
                    <td>Epsilon Minimum</td>
                    <td>{{ agent_config.epsilon_min }}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>🌍 Environment Configuration</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Grid Size</td>
                    <td>{{ environment.grid_size }}×{{ environment.grid_size }}</td>
                </tr>
                <tr>
                    <td>Total States</td>
                    <td>{{ environment.total_states }}</td>
                </tr>
                <tr>
                    <td>Number of Obstacles</td>
                    <td>{{ environment.num_obstacles }}</td>
                </tr>
                <tr>
                    <td>Start Position</td>
                    <td>({{ environment.start_position[0] }}, {{ environment.start_position[1] }})</td>
                </tr>
                <tr>
                    <td>Goal Position</td>
                    <td>({{ environment.goal_position[0] }}, {{ environment.goal_position[1] }})</td>
                </tr>
            </table>
        </div>
        
        {% else %}
        <div class="section">
            <h2>⏳ Initializing Metrics...</h2>
            <p>Please wait while the Q-Learning agent trains and metrics are calculated.</p>
            <p>This may take 10-30 seconds.</p>
        </div>
        {% endif %}
        
        <div class="last-update">
            {% if last_update %}
            Last updated: {{ last_update }}
            <span class="status active">● Active</span>
            {% else %}
            <span class="status">○ Waiting</span>
            {% endif %}
        </div>
    </div>
</body>
</html>
    ''', 
    training_metrics=metrics_data['training_metrics'],
    evaluation_metrics=metrics_data['evaluation_metrics'],
    q_table_stats=metrics_data['q_table_stats'],
    policy_info=metrics_data['policy_info'],
    environment=metrics_data.get('environment', {}),
    agent_config=metrics_data.get('agent_config', {}),
    last_update=metrics_data['last_update'])

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for metrics (JSON)"""
    return jsonify({
        'training': metrics_data['training_metrics'],
        'evaluation': metrics_data['evaluation_metrics'],
        'q_table': metrics_data['q_table_stats'],
        'policy': metrics_data['policy_info'],
        'last_update': metrics_data['last_update']
    })

if __name__ == '__main__':
    metrics_data['is_running'] = True
    
    # Start background thread for metrics updates
    update_thread = threading.Thread(target=update_metrics, daemon=True)
    update_thread.start()
    
    print("🚀 Starting Q-Learning Dashboard...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🔄 Metrics update every 5 seconds")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
    finally:
        metrics_data['is_running'] = False

