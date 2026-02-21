#!/usr/bin/env python3
"""
Real-time Dashboard for Day 128: Activation Functions
Run with: python dashboard.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import time
import json
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from lesson_code import (
    ReLU, LeakyReLU, Sigmoid, Tanh, Softmax,
    gradient_check, get_activation
)

app = Flask(__name__)

# Global state for metrics
metrics_data = {
    'activation_metrics': None,
    'gradient_checks': None,
    'visualizations': None,
    'last_update': None,
    'is_running': False
}

def calculate_activation_metrics():
    """Calculate activation function demonstration metrics"""
    np.random.seed(int(time.time()) % 1000)
    
    # Test data
    x_test = np.linspace(-5, 5, 100)
    x_random = np.random.randn(10, 5)
    
    activations = {
        'ReLU': ReLU(),
        'LeakyReLU': LeakyReLU(alpha=0.01),
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh(),
        'Softmax': Softmax()
    }
    
    metrics = {}
    gradient_results = {}
    
    for name, act in activations.items():
        # Forward pass metrics
        if name == 'Softmax':
            # Softmax needs 2D input
            y = act.forward(x_random)
            output_range = (y.min(), y.max())
            output_mean = y.mean()
            output_std = y.std()
            # Check if sums to 1
            sums = y.sum(axis=-1)
            sum_accuracy = np.mean(np.abs(sums - 1.0) < 1e-6)
        else:
            y = act.forward(x_test)
            output_range = (float(y.min()), float(y.max()))
            output_mean = float(y.mean())
            output_std = float(y.std())
            sum_accuracy = None
        
        # Backward pass
        grad = act.backward(np.ones_like(y))
        grad_mean = float(grad.mean())
        grad_std = float(grad.std())
        grad_max = float(grad.max())
        grad_min = float(grad.min())
        
        # Gradient check (for non-softmax)
        if name != 'Softmax':
            passed, error = gradient_check(act, x_random)
            gradient_results[name] = {
                'passed': passed,
                'max_error': float(error),
                'status': '✅ PASSED' if passed else '❌ FAILED'
            }
        else:
            gradient_results[name] = {
                'passed': True,
                'max_error': 0.0,
                'status': 'N/A (requires special handling)'
            }
        
        metrics[name] = {
            'output_range': output_range,
            'output_mean': output_mean,
            'output_std': output_std,
            'grad_mean': grad_mean,
            'grad_std': grad_std,
            'grad_range': (grad_min, grad_max),
            'sum_accuracy': sum_accuracy
        }
    
    return metrics, gradient_results

def generate_activation_plot():
    """Generate activation function visualization plot"""
    x = np.linspace(-5, 5, 300)
    
    activations_to_plot = [
        ('ReLU', ReLU(), 'tab:blue'),
        ('Sigmoid', Sigmoid(), 'tab:orange'),
        ('Tanh', Tanh(), 'tab:green'),
        ('LeakyReLU', LeakyReLU(alpha=0.01), 'tab:red'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Day 128: Activation Functions — Forward & Derivative', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (name, act, color) in enumerate(activations_to_plot):
        y = act.forward(x)
        dy = act.backward(np.ones_like(x))
        
        ax = axes[idx]
        ax2 = ax.twinx()
        
        # Forward pass
        line1 = ax.plot(x, y, color=color, linewidth=2.5, label=f'{name} — f(x)')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        
        # Derivative
        line2 = ax2.plot(x, dy, color=color, linewidth=2.5, linestyle='--', 
                        alpha=0.7, label=f"{name} — f'(x)")
        
        ax.set_title(f'{name}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Input (x)', fontsize=10)
        ax.set_ylabel('Output f(x)', fontsize=10, color=color)
        ax2.set_ylabel("Derivative f'(x)", fontsize=10, color=color)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=120, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_str

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Day 128: Activation Functions Dashboard</title>
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
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
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
        .metric-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .metric-item:last-child {
            border-bottom: none;
        }
        .metric-label {
            font-weight: 600;
            color: #555;
        }
        .metric-value {
            color: #333;
            font-weight: 500;
        }
        .gradient-status {
            padding: 5px 10px;
            border-radius: 5px;
            display: inline-block;
            font-weight: bold;
        }
        .status-passed {
            background: #d4edda;
            color: #155724;
        }
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        .visualization {
            text-align: center;
            margin: 30px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }
        .visualization img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .update-time {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
        }
        .refresh-info {
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Day 128: Activation Functions Dashboard</h1>
        <p class="subtitle">Real-time metrics and visualizations for neural network activation functions</p>
        
        <div class="visualization">
            <h2>Activation Functions Visualization</h2>
            <img src="data:image/png;base64,{{ plot_image }}" alt="Activation Functions">
        </div>
        
        <div class="metrics-grid">
            {% for name, metrics in activation_metrics.items() %}
            <div class="metric-card">
                <h3>{{ name }}</h3>
                <div class="metric-item">
                    <span class="metric-label">Output Range:</span>
                    <span class="metric-value">[{{ "%.3f"|format(metrics.output_range[0]) }}, {{ "%.3f"|format(metrics.output_range[1]) }}]</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Output Mean:</span>
                    <span class="metric-value">{{ "%.4f"|format(metrics.output_mean) }}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Output Std:</span>
                    <span class="metric-value">{{ "%.4f"|format(metrics.output_std) }}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Gradient Mean:</span>
                    <span class="metric-value">{{ "%.4f"|format(metrics.grad_mean) }}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Gradient Range:</span>
                    <span class="metric-value">[{{ "%.3f"|format(metrics.grad_range[0]) }}, {{ "%.3f"|format(metrics.grad_range[1]) }}]</span>
                </div>
                {% if metrics.sum_accuracy is not none %}
                <div class="metric-item">
                    <span class="metric-label">Sum Accuracy:</span>
                    <span class="metric-value">{{ "%.2f"|format(metrics.sum_accuracy * 100) }}%</span>
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        
        <div class="metric-card" style="margin-top: 20px;">
            <h3>Gradient Check Results</h3>
            {% for name, result in gradient_checks.items() %}
            <div class="metric-item">
                <span class="metric-label">{{ name }}:</span>
                <span class="gradient-status {{ 'status-passed' if result.passed else 'status-failed' }}">
                    {{ result.status }}
                </span>
                {% if result.max_error > 0 %}
                <span class="metric-value">(error: {{ "%.2e"|format(result.max_error) }})</span>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        
        <div class="update-time">
            <strong>Last Updated:</strong> {{ last_update }}
        </div>
        <div class="refresh-info">
            Auto-refreshing every 5 seconds...
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard page"""
    metrics, gradient_checks = calculate_activation_metrics()
    plot_image = generate_activation_plot()
    
    metrics_data['activation_metrics'] = metrics
    metrics_data['gradient_checks'] = gradient_checks
    metrics_data['visualizations'] = plot_image
    metrics_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics_data['is_running'] = True
    
    from flask import render_template_string
    return render_template_string(HTML_TEMPLATE,
                                 activation_metrics=metrics,
                                 gradient_checks=gradient_checks,
                                 plot_image=plot_image,
                                 last_update=metrics_data['last_update'])

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for metrics (JSON)"""
    metrics, gradient_checks = calculate_activation_metrics()
    # Convert gradient checks to JSON-serializable format
    gradient_checks_serializable = {
        k: {
            'passed': bool(v['passed']),
            'max_error': float(v['max_error']),
            'status': str(v['status'])
        } for k, v in gradient_checks.items()
    }
    return jsonify({
        'activation_metrics': {k: {
            'output_range': [float(v['output_range'][0]), float(v['output_range'][1])],
            'output_mean': float(v['output_mean']),
            'output_std': float(v['output_std']),
            'grad_mean': float(v['grad_mean']),
            'grad_std': float(v['grad_std']),
            'grad_range': [float(v['grad_range'][0]), float(v['grad_range'][1])],
            'sum_accuracy': float(v['sum_accuracy']) if v['sum_accuracy'] is not None else None
        } for k, v in metrics.items()},
        'gradient_checks': gradient_checks_serializable,
        'last_update': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Day 128 Activation Functions Dashboard...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🔄 Metrics update in real-time every 5 seconds")
    print("Press Ctrl+C to stop")
    print("")
    app.run(host='0.0.0.0', port=5000, debug=False)

