"""
Day 105: Content-Based Filtering - Flask API Server with Dashboard
REST API and web dashboard for the recommendation system
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from lesson_code import ContentBasedRecommender, create_sample_dataset
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize recommender
recommender = None
movies_df = None

def initialize_recommender():
    """Initialize the recommender system with sample data."""
    global recommender, movies_df
    
    print("🔄 Initializing recommender system...")
    movies_df = create_sample_dataset()
    
    recommender = ContentBasedRecommender(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1
    )
    
    recommender.fit(movies_df, text_column='combined_features')
    print("✅ Recommender initialized successfully")

# Initialize on startup
initialize_recommender()

# Dashboard HTML template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day 105: Content-Based Filtering Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
            text-align: center;
        }
        .header h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-card h3 {
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .metric-card .value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }
        .demo-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .demo-section h2 {
            color: #333;
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        .form-group select, .form-group input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 16px;
        }
        .form-group select:focus, .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:active {
            transform: translateY(0);
        }
        .results {
            margin-top: 30px;
        }
        .recommendation-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        .recommendation-item h4 {
            color: #333;
            margin-bottom: 5px;
        }
        .recommendation-item p {
            color: #666;
            font-size: 14px;
            margin: 5px 0;
        }
        .score {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 10px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .refresh-btn {
            background: #28a745;
            margin-left: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Content-Based Filtering Dashboard</h1>
            <p>Day 105: Production Recommendation System</p>
        </div>

        <div class="metrics-grid" id="metrics">
            <div class="metric-card">
                <h3>Total Items</h3>
                <div class="value" id="total-items">-</div>
            </div>
            <div class="metric-card">
                <h3>Vocabulary Size</h3>
                <div class="value" id="vocab-size">-</div>
            </div>
            <div class="metric-card">
                <h3>Fit Time</h3>
                <div class="value" id="fit-time">-</div>
            </div>
            <div class="metric-card">
                <h3>Recommendations Served</h3>
                <div class="value" id="recs-served">-</div>
            </div>
        </div>

        <div class="demo-section">
            <h2>🎯 Recommendation Demo</h2>
            <div class="form-group">
                <label for="movie-select">Select a Movie:</label>
                <select id="movie-select">
                    <option value="">Loading movies...</option>
                </select>
            </div>
            <div class="form-group">
                <label for="num-recs">Number of Recommendations:</label>
                <input type="number" id="num-recs" value="5" min="1" max="10">
            </div>
            <button onclick="getRecommendations()">Get Recommendations</button>
            <button class="refresh-btn" onclick="refreshMetrics()">Refresh Metrics</button>
            
            <div id="results" class="results"></div>
        </div>
    </div>

    <script>
        // Load movies on page load
        fetch('/api/movies')
            .then(r => r.json())
            .then(data => {
                const select = document.getElementById('movie-select');
                select.innerHTML = '<option value="">Select a movie...</option>';
                data.movies.forEach(movie => {
                    const option = document.createElement('option');
                    option.value = movie.item_id;
                    option.textContent = `${movie.title} (${movie.year})`;
                    select.appendChild(option);
                });
            });

        // Load metrics on page load
        refreshMetrics();

        function refreshMetrics() {
            fetch('/api/metrics')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('total-items').textContent = data.metrics.total_items;
                    document.getElementById('vocab-size').textContent = data.metrics.vocabulary_size;
                    document.getElementById('fit-time').textContent = data.metrics.fit_time.toFixed(2) + 's';
                    document.getElementById('recs-served').textContent = data.metrics.recommendations_served;
                })
                .catch(err => console.error('Error loading metrics:', err));
        }

        function getRecommendations() {
            const movieId = document.getElementById('movie-select').value;
            const numRecs = parseInt(document.getElementById('num-recs').value) || 5;
            const resultsDiv = document.getElementById('results');

            if (!movieId) {
                resultsDiv.innerHTML = '<div class="error">Please select a movie</div>';
                return;
            }

            resultsDiv.innerHTML = '<div class="loading">Loading recommendations...</div>';

            fetch(`/api/recommendations/${movieId}?n=${numRecs}`)
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        resultsDiv.innerHTML = `<div class="error">${data.error}</div>`;
                        return;
                    }

                    let html = '<h3>Top Recommendations:</h3>';
                    data.recommendations.forEach((rec, idx) => {
                        const item = rec.item_data;
                        html += `
                            <div class="recommendation-item">
                                <h4>${idx + 1}. ${item.title} (${item.year})</h4>
                                <p><strong>Genres:</strong> ${item.genres}</p>
                                <p><strong>Director:</strong> ${item.director}</p>
                                <p>
                                    <span class="score">Similarity: ${rec.similarity_score.toFixed(3)}</span>
                                    <span class="score">Final Score: ${rec.final_score.toFixed(3)}</span>
                                </p>
                            </div>
                        `;
                    });
                    resultsDiv.innerHTML = html;
                    refreshMetrics(); // Refresh metrics after getting recommendations
                })
                .catch(err => {
                    resultsDiv.innerHTML = `<div class="error">Error: ${err.message}</div>`;
                });
        }

        // Auto-refresh metrics every 10 seconds
        setInterval(refreshMetrics, 10000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Serve the dashboard page."""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics."""
    if recommender is None:
        return jsonify({'error': 'Recommender not initialized'}), 500
    
    metrics = recommender.get_metrics()
    return jsonify({
        'status': 'success',
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
def get_movies():
    """Get list of all movies."""
    if movies_df is None:
        return jsonify({'error': 'Movies not loaded'}), 500
    
    movies = movies_df.to_dict('records')
    return jsonify({
        'status': 'success',
        'movies': movies,
        'count': len(movies)
    })

@app.route('/api/recommendations/<item_id>', methods=['GET'])
def get_recommendations(item_id):
    """Get recommendations for a specific item."""
    if recommender is None:
        return jsonify({'error': 'Recommender not initialized'}), 500
    
    try:
        n_recommendations = int(request.args.get('n', 5))
        n_recommendations = max(1, min(10, n_recommendations))  # Clamp between 1 and 10
        
        recommendations = recommender.get_recommendations(
            item_id,
            n_recommendations=n_recommendations,
            apply_boost=True,
            diversity_threshold=0.85
        )
        
        return jsonify({
            'status': 'success',
            'item_id': item_id,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'recommender_initialized': recommender is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    print("=" * 70)
    print("🚀 Starting Day 105: Content-Based Filtering API Server")
    print("=" * 70)
    print(f"📊 Dashboard: http://localhost:{port}/")
    print(f"🔍 API Health: http://localhost:{port}/api/health")
    print(f"📈 Metrics API: http://localhost:{port}/api/metrics")
    print(f"🎬 Movies API: http://localhost:{port}/api/movies")
    print("=" * 70)
    app.run(host='0.0.0.0', port=port, debug=True)

