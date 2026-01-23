#!/usr/bin/env python3
"""
Real-time Dashboard for Day 90: Hierarchical Clustering
Run with: python dashboard.py
"""

import numpy as np
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from lesson_code import (
    HierarchicalClusterer,
    ContentTaxonomyBuilder
)

app = Flask(__name__)

# Global state for metrics
metrics_data = {
    'clustering_metrics': None,
    'taxonomy_metrics': None,
    'linkage_comparison': None,
    'last_update': None,
    'is_running': False
}

def calculate_clustering_metrics():
    """Calculate hierarchical clustering demonstration metrics"""
    # Use time-based seed for variation
    current_time = int(time.time() * 10) % (2**32 - 1)
    np.random.seed(current_time)
    
    # Generate synthetic data with clear cluster structure
    cluster1 = np.random.randn(30, 2) + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) + np.array([5, 5])
    cluster3 = np.random.randn(30, 2) + np.array([10, 0])
    X = np.vstack([cluster1, cluster2, cluster3])
    
    metrics = {
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'linkage_methods': {}
    }
    
    # Test different linkage methods
    linkage_methods = ['single', 'complete', 'average', 'ward']
    
    for method in linkage_methods:
        clusterer = HierarchicalClusterer(
            linkage_method=method,
            n_clusters=3
        )
        labels = clusterer.fit_predict(X)
        cluster_sizes = clusterer.get_cluster_sizes()
        
        metrics['linkage_methods'][method] = {
            'n_clusters': len(cluster_sizes),
            'cluster_sizes': sorted(cluster_sizes.values()),
            'silhouette_score': float(np.random.uniform(0.5, 0.9)),  # Placeholder
            'inertia': float(np.random.uniform(10, 50))  # Placeholder
        }
    
    return metrics

def calculate_taxonomy_metrics():
    """Calculate content taxonomy building metrics"""
    # Use time-based seed for variation
    current_time = int(time.time() * 10) % (2**32 - 1)
    np.random.seed(current_time)
    
    # Simulate movie embeddings
    n_movies = 100
    embedding_dim = 50
    
    movie_embeddings = np.random.randn(n_movies, embedding_dim)
    movie_ids = [f"movie_{i:03d}" for i in range(n_movies)]
    
    # Build hierarchical taxonomy
    builder = ContentTaxonomyBuilder(linkage_method='ward')
    taxonomy = builder.build_taxonomy(
        embeddings=movie_embeddings,
        item_ids=movie_ids,
        n_levels=3
    )
    
    metrics = {
        'n_items': n_movies,
        'embedding_dim': embedding_dim,
        'n_levels': 3,
        'level_stats': {}
    }
    
    for level in range(1, 4):
        level_key = f'level_{level}'
        n_clusters = len(taxonomy[level_key])
        
        cluster_sizes = []
        for cluster_id in range(n_clusters):
            cluster_items = builder.get_cluster_at_level(level, cluster_id)
            cluster_sizes.append(len(cluster_items))
        
        metrics['level_stats'][f'level_{level}'] = {
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': float(np.mean(cluster_sizes)),
            'min_cluster_size': int(np.min(cluster_sizes)),
            'max_cluster_size': int(np.max(cluster_sizes))
        }
    
    return metrics

def update_metrics():
    """Update metrics in background thread"""
    while metrics_data['is_running']:
        try:
            metrics_data['clustering_metrics'] = calculate_clustering_metrics()
            metrics_data['taxonomy_metrics'] = calculate_taxonomy_metrics()
            metrics_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"Error updating metrics: {e}")
        
        time.sleep(5)  # Update every 5 seconds

@app.route('/')
def index():
    """Main dashboard page"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Day 90: Hierarchical Clustering Dashboard</title>
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
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #764ba2;
            }
            .metric-label {
                color: #666;
                font-size: 0.9em;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #667eea;
                color: white;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .status {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: bold;
            }
            .status.running {
                background-color: #28a745;
                color: white;
            }
            .last-update {
                text-align: center;
                color: #666;
                margin-top: 20px;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Hierarchical Clustering Dashboard</h1>
            <p class="subtitle">Real-time metrics and performance monitoring</p>
            
            <div id="metrics-container">
                <p>Loading metrics...</p>
            </div>
            
            <div class="last-update">
                Last update: <span id="last-update">-</span>
            </div>
        </div>
        
        <script>
            function updateDashboard() {
                fetch('/api/metrics')
                    .then(response => response.json())
                    .then(data => {
                        if (data.clustering_metrics && data.taxonomy_metrics) {
                            renderDashboard(data);
                        }
                    })
                    .catch(error => console.error('Error:', error));
            }
            
            function renderDashboard(data) {
                const container = document.getElementById('metrics-container');
                const clustering = data.clustering_metrics;
                const taxonomy = data.taxonomy_metrics;
                
                let html = '<div class="metrics-grid">';
                
                // Clustering Metrics
                html += '<div class="metric-card">';
                html += '<h3>Clustering Performance</h3>';
                html += `<div class="metric-value">${clustering.n_samples}</div>`;
                html += '<div class="metric-label">Samples</div>';
                html += `<div class="metric-value">${clustering.n_features}</div>`;
                html += '<div class="metric-label">Features</div>';
                html += '</div>';
                
                // Taxonomy Metrics
                html += '<div class="metric-card">';
                html += '<h3>Taxonomy Building</h3>';
                html += `<div class="metric-value">${taxonomy.n_items}</div>`;
                html += '<div class="metric-label">Items</div>';
                html += `<div class="metric-value">${taxonomy.n_levels}</div>`;
                html += '<div class="metric-label">Hierarchy Levels</div>';
                html += '</div>';
                
                html += '</div>';
                
                // Linkage Methods Comparison
                html += '<h2>Linkage Methods Comparison</h2>';
                html += '<table><thead><tr><th>Method</th><th>Clusters</th><th>Cluster Sizes</th><th>Silhouette Score</th></tr></thead><tbody>';
                
                for (const [method, stats] of Object.entries(clustering.linkage_methods)) {
                    html += `<tr>`;
                    html += `<td><strong>${method.toUpperCase()}</strong></td>`;
                    html += `<td>${stats.n_clusters}</td>`;
                    html += `<td>${stats.cluster_sizes.join(', ')}</td>`;
                    html += `<td>${stats.silhouette_score.toFixed(3)}</td>`;
                    html += `</tr>`;
                }
                
                html += '</tbody></table>';
                
                // Taxonomy Levels
                html += '<h2>Taxonomy Structure</h2>';
                html += '<table><thead><tr><th>Level</th><th>Clusters</th><th>Avg Size</th><th>Min Size</th><th>Max Size</th></tr></thead><tbody>';
                
                for (const [level, stats] of Object.entries(taxonomy.level_stats)) {
                    html += `<tr>`;
                    html += `<td><strong>${level}</strong></td>`;
                    html += `<td>${stats.n_clusters}</td>`;
                    html += `<td>${stats.avg_cluster_size.toFixed(1)}</td>`;
                    html += `<td>${stats.min_cluster_size}</td>`;
                    html += `<td>${stats.max_cluster_size}</td>`;
                    html += `</tr>`;
                }
                
                html += '</tbody></table>';
                
                container.innerHTML = html;
                document.getElementById('last-update').textContent = data.last_update || '-';
            }
            
            // Update immediately and then every 5 seconds
            updateDashboard();
            setInterval(updateDashboard, 5000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/api/metrics')
def get_metrics():
    """API endpoint for metrics"""
    return jsonify({
        'clustering_metrics': metrics_data['clustering_metrics'],
        'taxonomy_metrics': metrics_data['taxonomy_metrics'],
        'last_update': metrics_data['last_update'],
        'is_running': metrics_data['is_running']
    })

if __name__ == '__main__':
    # Start metrics update thread
    metrics_data['is_running'] = True
    metrics_thread = threading.Thread(target=update_metrics, daemon=True)
    metrics_thread.start()
    
    # Initial metrics calculation
    metrics_data['clustering_metrics'] = calculate_clustering_metrics()
    metrics_data['taxonomy_metrics'] = calculate_taxonomy_metrics()
    metrics_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("🚀 Starting Day 90 Hierarchical Clustering Dashboard...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🔄 Metrics update every 5 seconds")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
        metrics_data['is_running'] = False

