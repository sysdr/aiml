"""
Real-time Dashboard for ML Dataset Analyzer
Flask web application with live metrics
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from lesson_code import MLDatasetAnalyzer
import threading
import time

app = Flask(__name__)
CORS(app)

# Global storage for current analysis
current_analysis = {
    'status': 'idle',
    'data': None,
    'timestamp': None
}

# Lock for thread-safe updates
analysis_lock = threading.Lock()

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current analysis status"""
    with analysis_lock:
        return jsonify(current_analysis)

@app.route('/api/analyze', methods=['POST'])
def analyze_dataset():
    """Analyze uploaded dataset"""
    global current_analysis
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read dataset
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_csv(file)  # For simplicity, convert Excel to CSV first
        else:
            return jsonify({'error': 'Unsupported file format. Please upload CSV.'}), 400
        
        # Get target column if provided
        target_column = request.form.get('target_column', None)
        if target_column and target_column not in df.columns:
            target_column = None
        
        # Update status
        with analysis_lock:
            current_analysis['status'] = 'analyzing'
            current_analysis['timestamp'] = datetime.now().isoformat()
        
        # Run analysis in background thread
        def run_analysis():
            global current_analysis
            try:
                analyzer = MLDatasetAnalyzer(df, target_column=target_column)
                analyzer.run_full_analysis(generate_viz=True, generate_html=False)
                
                # Prepare response data
                results = {
                    'dataset_info': {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'numeric_features': len(analyzer.numeric_features),
                        'categorical_features': len(analyzer.categorical_features),
                        'missing_values': int(df.isnull().sum().sum()),
                        'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2)
                    },
                    'feature_profiles': {},
                    'quality_issues': analyzer.analysis_results.get('quality_issues', {}),
                    'correlations': {
                        'high_corr_pairs': len(analyzer.analysis_results.get('high_correlations', [])),
                        'high_corr_list': [
                            {'feature1': f1, 'feature2': f2, 'correlation': float(corr)}
                            for f1, f2, corr in analyzer.analysis_results.get('high_correlations', [])
                        ]
                    },
                    'normality_tests': {},
                    'ml_readiness': analyzer.analysis_results.get('ml_readiness', {}),
                    'feature_names': analyzer.numeric_features
                }
                
                # Format feature profiles
                profiles = analyzer.analysis_results.get('feature_profiles', {})
                for feature, profile in profiles.items():
                    results['feature_profiles'][feature] = {
                        'mean': float(profile['mean']),
                        'median': float(profile['median']),
                        'std': float(profile['std']),
                        'min': float(profile['min']),
                        'max': float(profile['max']),
                        'skewness': float(profile['skewness']),
                        'kurtosis': float(profile['kurtosis']),
                        'missing_pct': float(profile['missing_pct']),
                        'outlier_pct': float(profile['outlier_pct']),
                        'cv': float(profile['cv'])
                    }
                
                # Format normality tests
                normality = analyzer.analysis_results.get('normality_tests', {})
                for feature, test in normality.items():
                    results['normality_tests'][feature] = {
                        'statistic': float(test['statistic']),
                        'p_value': float(test['p_value']),
                        'is_normal': test['is_normal'],
                        'interpretation': test['interpretation']
                    }
                
                # Update global state
                with analysis_lock:
                    current_analysis['status'] = 'complete'
                    current_analysis['data'] = results
                    current_analysis['timestamp'] = datetime.now().isoformat()
                    
            except Exception as e:
                with analysis_lock:
                    current_analysis['status'] = 'error'
                    current_analysis['error'] = str(e)
                    current_analysis['timestamp'] = datetime.now().isoformat()
        
        # Start analysis in background
        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Analysis started. Poll /api/status for results.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/sample/<dataset_type>')
def analyze_sample_dataset(dataset_type):
    """Analyze a sample dataset"""
    global current_analysis
    
    try:
        if dataset_type == 'clean':
            if not os.path.exists('clean_dataset.csv'):
                return jsonify({'error': 'Clean dataset not found. Run lesson_code.py first.'}), 404
            df = pd.read_csv('clean_dataset.csv')
            target_column = 'approved'
        elif dataset_type == 'messy':
            if not os.path.exists('messy_dataset.csv'):
                return jsonify({'error': 'Messy dataset not found. Run lesson_code.py first.'}), 404
            df = pd.read_csv('messy_dataset.csv')
            target_column = 'approved'
        else:
            return jsonify({'error': 'Invalid dataset type'}), 400
        
        # Update status
        with analysis_lock:
            current_analysis['status'] = 'analyzing'
            current_analysis['timestamp'] = datetime.now().isoformat()
        
        # Run analysis in background
        def run_analysis():
            global current_analysis
            try:
                analyzer = MLDatasetAnalyzer(df, target_column=target_column)
                analyzer.run_full_analysis(generate_viz=True, generate_html=False)
                
                # Prepare response data (same as analyze_dataset)
                results = {
                    'dataset_info': {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'numeric_features': len(analyzer.numeric_features),
                        'categorical_features': len(analyzer.categorical_features),
                        'missing_values': int(df.isnull().sum().sum()),
                        'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2)
                    },
                    'feature_profiles': {},
                    'quality_issues': analyzer.analysis_results.get('quality_issues', {}),
                    'correlations': {
                        'high_corr_pairs': len(analyzer.analysis_results.get('high_correlations', [])),
                        'high_corr_list': [
                            {'feature1': f1, 'feature2': f2, 'correlation': float(corr)}
                            for f1, f2, corr in analyzer.analysis_results.get('high_correlations', [])
                        ]
                    },
                    'normality_tests': {},
                    'ml_readiness': analyzer.analysis_results.get('ml_readiness', {}),
                    'feature_names': analyzer.numeric_features
                }
                
                profiles = analyzer.analysis_results.get('feature_profiles', {})
                for feature, profile in profiles.items():
                    results['feature_profiles'][feature] = {
                        'mean': float(profile['mean']),
                        'median': float(profile['median']),
                        'std': float(profile['std']),
                        'min': float(profile['min']),
                        'max': float(profile['max']),
                        'skewness': float(profile['skewness']),
                        'kurtosis': float(profile['kurtosis']),
                        'missing_pct': float(profile['missing_pct']),
                        'outlier_pct': float(profile['outlier_pct']),
                        'cv': float(profile['cv'])
                    }
                
                normality = analyzer.analysis_results.get('normality_tests', {})
                for feature, test in normality.items():
                    results['normality_tests'][feature] = {
                        'statistic': float(test['statistic']),
                        'p_value': float(test['p_value']),
                        'is_normal': test['is_normal'],
                        'interpretation': test['interpretation']
                    }
                
                with analysis_lock:
                    current_analysis['status'] = 'complete'
                    current_analysis['data'] = results
                    current_analysis['timestamp'] = datetime.now().isoformat()
                    
            except Exception as e:
                with analysis_lock:
                    current_analysis['status'] = 'error'
                    current_analysis['error'] = str(e)
                    current_analysis['timestamp'] = datetime.now().isoformat()
        
        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': f'Analysis of {dataset_type} dataset started.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve generated plot images"""
    return send_from_directory('plots', filename)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("=" * 60)
    print("🚀 Starting ML Dataset Analyzer Dashboard")
    print("=" * 60)
    print("\n📊 Dashboard URL: http://localhost:5000")
    print("📊 Dashboard URL: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

