"""
Flask Dashboard for Day 34: DataFrame Indexing, Slicing, and Filtering
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from lesson_code import ContentRecommendationEngine

import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            static_folder=os.path.join(BASE_DIR, 'static'),
            template_folder=os.path.join(BASE_DIR, 'templates'))
CORS(app)

# Initialize the recommendation engine
engine = ContentRecommendationEngine(n_content_items=1000)

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('dashboard.html')

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests to prevent 404 errors."""
    return '', 204  # No content

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors gracefully."""
    return jsonify({'error': 'Not found'}), 404

@app.route('/api/stats')
def get_stats():
    """Get overall statistics about the dataset."""
    df = engine.df
    
    stats = {
        'total_items': len(df),
        'total_views': int(df['views'].sum()),
        'avg_views': float(df['views'].mean()),
        'avg_rating': float(df['rating'].mean()),
        'avg_completion_rate': float(df['completion_rate'].mean()),
        'categories': df['category'].value_counts().to_dict(),
        'total_likes': int(df['likes'].sum()),
        'total_shares': int(df['shares'].sum()),
        'avg_watch_time': float(df['watch_time_minutes'].mean())
    }
    
    return jsonify(stats)

@app.route('/api/data')
def get_data():
    """Get paginated data from the DataFrame."""
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    category = request.args.get('category', 'all')
    min_views = int(request.args.get('min_views', 0))
    min_rating = float(request.args.get('min_rating', 0))
    
    df = engine.df.copy()
    
    # Apply filters
    if category != 'all':
        df = df[df['category'] == category]
    if min_views > 0:
        df = df[df['views'] >= min_views]
    if min_rating > 0:
        df = df[df['rating'] >= min_rating]
    
    # Pagination
    start = (page - 1) * per_page
    end = start + per_page
    paginated_df = df.iloc[start:end]
    
    # Convert to dict for JSON
    data = paginated_df.reset_index().to_dict('records')
    
    return jsonify({
        'data': data,
        'total': len(df),
        'page': page,
        'per_page': per_page,
        'total_pages': (len(df) + per_page - 1) // per_page
    })

@app.route('/api/recommendations')
def get_recommendations():
    """Get recommendations based on user preferences."""
    category = request.args.get('category', 'tech')
    min_quality = float(request.args.get('min_quality', 0.7))
    
    recommendations = engine.build_recommendation_filter(
        user_category=category,
        min_quality=min_quality
    )
    
    # Convert to dict for JSON
    data = recommendations.reset_index().to_dict('records')
    
    return jsonify({
        'recommendations': data,
        'count': len(data)
    })

@app.route('/api/quality-analysis')
def get_quality_analysis():
    """Get data quality analysis metrics."""
    results = engine.analyze_data_quality()
    return jsonify(results)

@app.route('/api/visualization')
def get_visualization():
    """Generate and return visualization as base64 image."""
    viz_type = request.args.get('type', 'overview')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DataFrame Filtering Analysis Dashboard', 
                 fontsize=16, fontweight='bold')
    
    df = engine.df
    
    # Plot 1: Views distribution
    axes[0, 0].hist(df['views'], bins=50, alpha=0.7, color='blue', label='All Content')
    high_quality = df[
        (df['completion_rate'] > 0.7) & 
        (df['rating'] > 4.0)
    ]
    axes[0, 0].hist(high_quality['views'], bins=50, alpha=0.7, color='green', label='High Quality')
    axes[0, 0].set_xlabel('Views')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Views Distribution: All vs High Quality')
    axes[0, 0].legend()
    
    # Plot 2: Category breakdown
    category_counts = df['category'].value_counts()
    axes[0, 1].bar(category_counts.index, category_counts.values, color='coral')
    axes[0, 1].set_xlabel('Category')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Content Distribution by Category')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Completion rate vs Rating
    axes[1, 0].scatter(df['completion_rate'], df['rating'], 
                      alpha=0.3, s=10, color='purple')
    axes[1, 0].axvline(x=0.7, color='r', linestyle='--', label='Quality Threshold')
    axes[1, 0].axhline(y=4.0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Completion Rate')
    axes[1, 0].set_ylabel('Rating')
    axes[1, 0].set_title('Quality Filter: Completion Rate vs Rating')
    axes[1, 0].legend()
    
    # Plot 4: Filtering funnel
    filter_stages = ['Total', 'Views>5K', 'Complete>0.7', 'Rating>4.0', 'All Filters']
    counts = [
        len(df),
        len(df[df['views'] > 5000]),
        len(df[df['completion_rate'] > 0.7]),
        len(df[df['rating'] > 4.0]),
        len(df[(df['views'] > 5000) & 
               (df['completion_rate'] > 0.7) & 
               (df['rating'] > 4.0)])
    ]
    axes[1, 1].bar(filter_stages, counts, color=['gray', 'blue', 'green', 'orange', 'red'])
    axes[1, 1].set_ylabel('Content Count')
    axes[1, 1].set_title('Filtering Funnel: Progressive Data Reduction')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()
    
    return jsonify({'image': f'data:image/png;base64,{img_base64}'})

@app.route('/api/filtering-demo')
def filtering_demo():
    """Demonstrate filtering operations with sample data."""
    df = engine.df
    
    # Single condition
    popular = df[df['views'] > 50000]
    
    # Multiple conditions (AND)
    high_quality = df[
        (df['completion_rate'] > 0.7) &
        (df['rating'] > 4.0) &
        (df['views'] > 5000)
    ]
    
    # Multiple conditions (OR)
    viral_or_engaging = df[
        (df['shares'] > 1000) |
        (df['completion_rate'] > 0.9)
    ]
    
    return jsonify({
        'popular': len(popular),
        'high_quality': len(high_quality),
        'viral_or_engaging': len(viral_or_engaging),
        'popular_percentage': len(popular) / len(df) * 100,
        'high_quality_percentage': len(high_quality) / len(df) * 100,
        'viral_or_engaging_percentage': len(viral_or_engaging) / len(df) * 100
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

