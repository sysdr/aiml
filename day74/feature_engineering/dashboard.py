#!/usr/bin/env python3
"""
Real-time Dashboard for Day 74: Feature Engineering
Run with: python dashboard.py
"""

import pandas as pd
import numpy as np
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from lesson_code import (
    FeatureEngineeringPipeline,
    PolynomialFeatureCreator,
    FeatureBinner,
    FeatureSelector,
    create_sample_dataset
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Global state for metrics
metrics_data = {
    'feature_metrics': None,
    'model_performance': None,
    'feature_selection': None,
    'last_update': None,
    'is_running': False
}

def calculate_feature_engineering_metrics():
    """Calculate feature engineering demonstration metrics"""
    # Use time-based seed for variation
    current_time = int(time.time() * 10) % (2**32 - 1)
    np.random.seed(current_time)
    
    # Load data
    df = create_sample_dataset()
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 1. Basic Feature Engineering
    fe_pipeline = FeatureEngineeringPipeline(
        scaling_strategy='standard',
        encoding_strategy='onehot'
    )
    X_train_basic = fe_pipeline.fit_transform(X_train)
    X_test_basic = fe_pipeline.transform(X_test)
    
    # Get feature analysis
    numeric_features = fe_pipeline.numeric_features
    categorical_features = fe_pipeline.categorical_features
    
    # 2. Polynomial Features
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train_numeric = X_train[numeric_cols].values
    X_test_numeric = X_test[numeric_cols].values
    
    poly_creator = PolynomialFeatureCreator(degree=2, interaction_only=True)
    X_train_poly = poly_creator.fit_transform(
        X_train_numeric,
        feature_names=numeric_cols.tolist()
    )
    X_test_poly = poly_creator.transform(X_test_numeric)
    
    # 3. Feature Binning
    binner = FeatureBinner(n_bins=5, strategy='quantile')
    monthly_charges_binned = binner.fit_transform(
        X_train['monthly_charges'].values,
        'monthly_charges'
    )
    unique_bins, bin_counts = np.unique(monthly_charges_binned, return_counts=True)
    bin_distribution = [
        {'bin': int(b), 'count': int(c), 'percentage': float(c/len(monthly_charges_binned)*100)}
        for b, c in zip(unique_bins, bin_counts)
    ]
    
    # 4. Model Performance
    # Baseline
    X_train_numeric_only = X_train.select_dtypes(include=['int64', 'float64']).values
    X_test_numeric_only = X_test.select_dtypes(include=['int64', 'float64']).values
    
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train_numeric_only, y_train)
    baseline_pred = baseline_model.predict(X_test_numeric_only)
    baseline_acc = float(accuracy_score(y_test, baseline_pred))
    
    # With feature engineering
    fe_model = LogisticRegression(max_iter=1000, random_state=42)
    fe_model.fit(X_train_basic, y_train)
    fe_pred = fe_model.predict(X_test_basic)
    fe_acc = float(accuracy_score(y_test, fe_pred))
    
    # With polynomial features
    poly_model = RandomForestClassifier(n_estimators=100, random_state=42)
    poly_model.fit(X_train_poly, y_train)
    poly_pred = poly_model.predict(X_test_poly)
    poly_acc = float(accuracy_score(y_test, poly_pred))
    
    # 5. Feature Selection
    selector = FeatureSelector(k=10, score_func='f_classif')
    X_train_selected = selector.fit_transform(
        X_train_basic, y_train, fe_pipeline.feature_names
    )
    X_test_selected = selector.transform(X_test_basic)
    
    selected_model = LogisticRegression(max_iter=1000, random_state=42)
    selected_model.fit(X_train_selected, y_train)
    selected_pred = selected_model.predict(X_test_selected)
    selected_acc = float(accuracy_score(y_test, selected_pred))
    
    # Get top selected features
    if selector.selected_features:
        top_features = selector.selected_features[:10]
    else:
        top_features = []
    
    return {
        'dataset': {
            'total_samples': int(df.shape[0]),
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'original_features': int(X.shape[1]),
            'churn_rate': float(y.mean())
        },
        'feature_engineering': {
            'numeric_features': len(numeric_features),
            'categorical_features': len(categorical_features),
            'original_shape': [int(X_train.shape[0]), int(X_train.shape[1])],
            'transformed_shape': [int(X_train_basic.shape[0]), int(X_train_basic.shape[1])],
            'feature_expansion': int(X_train_basic.shape[1] - X_train.shape[1])
        },
        'polynomial_features': {
            'original_count': int(X_train_numeric.shape[1]),
            'polynomial_count': int(X_train_poly.shape[1]),
            'expansion_ratio': float(X_train_poly.shape[1] / X_train_numeric.shape[1]),
            'degree': 2,
            'interaction_only': True
        },
        'binning': {
            'feature': 'monthly_charges',
            'n_bins': 5,
            'strategy': 'quantile',
            'bin_distribution': bin_distribution
        },
        'model_performance': {
            'baseline': {
                'accuracy': baseline_acc,
                'improvement': 0.0
            },
            'with_feature_engineering': {
                'accuracy': fe_acc,
                'improvement': float((fe_acc - baseline_acc) * 100)
            },
            'with_polynomial': {
                'accuracy': poly_acc,
                'improvement': float((poly_acc - baseline_acc) * 100)
            },
            'with_selection': {
                'accuracy': selected_acc,
                'improvement': float((selected_acc - baseline_acc) * 100)
            }
        },
        'feature_selection': {
            'original_features': int(X_train_basic.shape[1]),
            'selected_features': int(X_train_selected.shape[1]),
            'reduction_pct': float((1 - X_train_selected.shape[1]/X_train_basic.shape[1]) * 100),
            'top_features': top_features
        }
    }

def update_metrics():
    """Update metrics in background thread"""
    while metrics_data['is_running']:
        try:
            metrics = calculate_feature_engineering_metrics()
            metrics_data['feature_metrics'] = metrics
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
    <title>Day 74: Feature Engineering Dashboard</title>
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
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Day 74: Feature Engineering Dashboard</h1>
        <p class="subtitle">Real-time metrics and model performance comparison</p>
        
        {% if metrics %}
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Dataset Size</h3>
                <div class="metric-value">{{ metrics.dataset.total_samples }}</div>
                <div class="metric-label">Total Samples</div>
                <div style="margin-top: 10px;">
                    <small>Train: {{ metrics.dataset.train_samples }} | Test: {{ metrics.dataset.test_samples }}</small>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Feature Expansion</h3>
                <div class="metric-value">{{ metrics.feature_engineering.feature_expansion }}</div>
                <div class="metric-label">New Features Created</div>
                <div style="margin-top: 10px;">
                    <small>{{ metrics.feature_engineering.original_shape[1] }} → {{ metrics.feature_engineering.transformed_shape[1] }} features</small>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Polynomial Features</h3>
                <div class="metric-value">{{ "%.1f"|format(metrics.polynomial_features.expansion_ratio) }}x</div>
                <div class="metric-label">Feature Expansion Ratio</div>
                <div style="margin-top: 10px;">
                    <small>{{ metrics.polynomial_features.original_count }} → {{ metrics.polynomial_features.polynomial_count }} features</small>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Feature Selection</h3>
                <div class="metric-value">{{ "%.1f"|format(metrics.feature_selection.reduction_pct) }}%</div>
                <div class="metric-label">Dimensionality Reduction</div>
                <div style="margin-top: 10px;">
                    <small>{{ metrics.feature_selection.original_features }} → {{ metrics.feature_selection.selected_features }} features</small>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Model Performance Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Improvement</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Baseline (Raw Features)</strong></td>
                        <td>{{ "%.4f"|format(metrics.model_performance.baseline.accuracy) }}</td>
                        <td>0.00%</td>
                    </tr>
                    <tr>
                        <td><strong>With Feature Engineering</strong></td>
                        <td>{{ "%.4f"|format(metrics.model_performance.with_feature_engineering.accuracy) }}</td>
                        <td class="improvement">+{{ "%.2f"|format(metrics.model_performance.with_feature_engineering.improvement) }}%</td>
                    </tr>
                    <tr>
                        <td><strong>With Polynomial Features</strong></td>
                        <td>{{ "%.4f"|format(metrics.model_performance.with_polynomial.accuracy) }}</td>
                        <td class="improvement">+{{ "%.2f"|format(metrics.model_performance.with_polynomial.improvement) }}%</td>
                    </tr>
                    <tr>
                        <td><strong>With Feature Selection</strong></td>
                        <td>{{ "%.4f"|format(metrics.model_performance.with_selection.accuracy) }}</td>
                        <td class="improvement">+{{ "%.2f"|format(metrics.model_performance.with_selection.improvement) }}%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>🔧 Feature Engineering Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Numeric Features</td>
                        <td>{{ metrics.feature_engineering.numeric_features }}</td>
                    </tr>
                    <tr>
                        <td>Categorical Features</td>
                        <td>{{ metrics.feature_engineering.categorical_features }}</td>
                    </tr>
                    <tr>
                        <td>Original Shape</td>
                        <td>{{ metrics.feature_engineering.original_shape[0] }} × {{ metrics.feature_engineering.original_shape[1] }}</td>
                    </tr>
                    <tr>
                        <td>Transformed Shape</td>
                        <td>{{ metrics.feature_engineering.transformed_shape[0] }} × {{ metrics.feature_engineering.transformed_shape[1] }}</td>
                    </tr>
                    <tr>
                        <td>Churn Rate</td>
                        <td>{{ "%.1f"|format(metrics.dataset.churn_rate * 100) }}%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        {% if metrics.feature_selection.top_features %}
        <div class="section">
            <h2>⭐ Top Selected Features</h2>
            <ol>
                {% for feature in metrics.feature_selection.top_features[:10] %}
                <li>{{ feature }}</li>
                {% endfor %}
            </ol>
        </div>
        {% endif %}
        
        {% if metrics.binning.bin_distribution %}
        <div class="section">
            <h2>📦 Feature Binning: {{ metrics.binning.feature }}</h2>
            <table>
                <thead>
                    <tr>
                        <th>Bin</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    {% for bin_data in metrics.binning.bin_distribution %}
                    <tr>
                        <td>Bin {{ bin_data.bin }}</td>
                        <td>{{ bin_data.count }}</td>
                        <td>{{ "%.1f"|format(bin_data.percentage) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        {% else %}
        <div class="section">
            <p>Loading metrics...</p>
        </div>
        {% endif %}
        
        <div class="last-update">
            <span class="status active">● LIVE</span>
            {% if last_update %}
            Last updated: {{ last_update }}
            {% endif %}
        </div>
    </div>
</body>
</html>
    ''', metrics=metrics_data.get('feature_metrics'), last_update=metrics_data.get('last_update'))

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for metrics"""
    return jsonify(metrics_data.get('feature_metrics', {}))

if __name__ == '__main__':
    # Start background metrics update thread
    metrics_data['is_running'] = True
    update_thread = threading.Thread(target=update_metrics, daemon=True)
    update_thread.start()
    
    # Initial metrics calculation
    try:
        metrics = calculate_feature_engineering_metrics()
        metrics_data['feature_metrics'] = metrics
        metrics_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error calculating initial metrics: {e}")
    
    print("="*70)
    print("🚀 Day 74: Feature Engineering Dashboard")
    print("="*70)
    print("📊 Dashboard available at: http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/api/metrics")
    print("🔄 Metrics update every 5 seconds")
    print("="*70)
    print("\nPress Ctrl+C to stop the dashboard\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

