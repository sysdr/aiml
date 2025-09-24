#!/usr/bin/env python3
"""
Web Dashboard for Day 8: Introduction to Linear Algebra for AI
180-Day AI and Machine Learning Course

A web dashboard to display linear algebra metrics and demo functionality.
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import matplotlib
import time
import threading
from lesson_code import SimpleRecommender, demo_vectors, demo_similarity, demo_matrices, demo_recommendation_system
import json

app = Flask(__name__)

# Global metrics tracking
metrics = {
    'status': 'Active',
    'numpy_version': np.__version__,
    'matplotlib_status': 'Available',
    'vectors_created': 0,
    'dot_products': 0,
    'similarity_calculations': 0,
    'matrices_created': 0,
    'matrix_multiplications': 0,
    'recommendations_generated': 0,
    'users_count': 0,
    'items_count': 0,
    'avg_recommendation_score': 0.0,
    'total_operations': 0
}

# Global recommender instance
recommender = SimpleRecommender()

def initialize_recommender():
    """Initialize the recommendation system with sample data"""
    global recommender, metrics
    
    # Add users
    recommender.add_user("tech_lover", [0.9, 0.2, 0.3, 0.1])
    recommender.add_user("bookworm", [0.1, 0.9, 0.4, 0.2])
    recommender.add_user("athlete", [0.2, 0.1, 0.3, 0.9])
    recommender.add_user("balanced", [0.5, 0.5, 0.5, 0.5])
    
    # Add items
    recommender.add_item("laptop", [0.95, 0.1, 0.2, 0.0])
    recommender.add_item("sci_fi_novel", [0.3, 0.9, 0.6, 0.0])
    recommender.add_item("action_movie", [0.2, 0.1, 0.9, 0.3])
    recommender.add_item("tennis_racket", [0.1, 0.0, 0.2, 0.95])
    recommender.add_item("programming_book", [0.8, 0.8, 0.2, 0.0])
    recommender.add_item("fitness_tracker", [0.6, 0.1, 0.3, 0.8])
    
    metrics['users_count'] = len(recommender.user_profiles)
    metrics['items_count'] = len(recommender.item_features)

def update_metrics():
    """Update metrics periodically"""
    global metrics
    
    while True:
        # Calculate average recommendation score
        total_score = 0
        total_recommendations = 0
        
        for user_id in recommender.user_profiles:
            recommendations = recommender.recommend(user_id, top_n=3)
            for _, score in recommendations:
                total_score += score
                total_recommendations += 1
        
        if total_recommendations > 0:
            metrics['avg_recommendation_score'] = total_score / total_recommendations
        
        metrics['total_operations'] = (
            metrics['vectors_created'] + 
            metrics['dot_products'] + 
            metrics['similarity_calculations'] + 
            metrics['matrices_created'] + 
            metrics['matrix_multiplications'] + 
            metrics['recommendations_generated']
        )
        
        time.sleep(5)  # Update every 5 seconds

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """API endpoint to get current metrics"""
    return jsonify(metrics)

@app.route('/api/demo/vectors', methods=['POST'])
def run_vector_demo():
    """API endpoint to run vector operations demo"""
    try:
        # Create sample vectors
        customer_alice = np.array([0.9, 0.2, 0.1, 0.8])
        customer_bob = np.array([0.1, 0.9, 0.7, 0.2])
        customer_charlie = np.array([0.6, 0.6, 0.4, 0.6])
        
        # Vector operations
        combined_prefs = customer_alice + customer_bob
        amplified_alice = 2 * customer_alice
        alice_magnitude = np.linalg.norm(customer_alice)
        
        # Update metrics
        metrics['vectors_created'] += 3
        metrics['dot_products'] += 1
        
        return jsonify({
            'alice_vector': customer_alice.tolist(),
            'bob_vector': customer_bob.tolist(),
            'charlie_vector': customer_charlie.tolist(),
            'combined_vector': combined_prefs.tolist(),
            'amplified_vector': amplified_alice.tolist(),
            'alice_magnitude': float(alice_magnitude)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo/similarity', methods=['POST'])
def run_similarity_demo():
    """API endpoint to run similarity calculations demo"""
    try:
        # Create user preference vectors
        users = {
            "Alice": np.array([0.9, 0.2, 0.1, 0.8]),
            "Bob": np.array([0.1, 0.9, 0.7, 0.2]),
            "Charlie": np.array([0.6, 0.6, 0.4, 0.6]),
            "Diana": np.array([0.8, 0.3, 0.2, 0.7])
        }
        
        user_names = list(users.keys())
        similarities = {}
        
        for i, user1 in enumerate(user_names):
            for j, user2 in enumerate(user_names):
                if i < j:  # Avoid duplicates
                    similarity = np.dot(users[user1], users[user2])
                    similarities[f"{user1}-{user2}"] = float(similarity)
        
        # Find most similar pair
        most_similar = max(similarities.items(), key=lambda x: x[1])
        
        # Update metrics
        metrics['similarity_calculations'] += len(similarities)
        
        return jsonify({
            'similarities': similarities,
            'most_similar': f"{most_similar[0]} (score: {most_similar[1]:.3f})"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo/matrix', methods=['POST'])
def run_matrix_demo():
    """API endpoint to run matrix operations demo"""
    try:
        # User-item rating matrix
        ratings = np.array([
            [5, 2, 1, 5],  # Alice
            [1, 5, 4, 2],  # Bob  
            [3, 4, 4, 3],  # Charlie
            [5, 3, 2, 4]   # Diana
        ])
        
        users = ["Alice", "Bob", "Charlie", "Diana"]
        categories = ["Electronics", "Books", "Clothing", "Sports"]
        
        # Matrix operations
        avg_ratings = np.mean(ratings, axis=0)
        
        # Recommendation weights
        rec_weights = np.array([
            [1.2, 0.8, 0.5],  # Electronics -> [Premium, Standard, Budget]
            [0.3, 1.1, 0.9],  # Books -> [Premium, Standard, Budget] 
            [0.2, 0.9, 1.2],  # Clothing -> [Premium, Standard, Budget]
            [1.0, 0.7, 0.4]   # Sports -> [Premium, Standard, Budget]
        ])
        
        recommendations = np.dot(ratings, rec_weights)
        
        # Update metrics
        metrics['matrices_created'] += 2
        metrics['matrix_multiplications'] += 1
        
        return jsonify({
            'ratings': ratings.tolist(),
            'users': users,
            'avg_ratings': avg_ratings.tolist(),
            'recommendations': recommendations.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo/recommendations', methods=['POST'])
def run_recommendation_demo():
    """API endpoint to run recommendation system demo"""
    try:
        global recommender
        
        # Generate recommendations for all users
        recommendations = {}
        for user_id in recommender.user_profiles:
            user_recs = recommender.recommend(user_id, top_n=3)
            recommendations[user_id] = [
                {'item': item_id, 'score': float(score)} 
                for item_id, score in user_recs
            ]
        
        # Get explanation for one recommendation
        explanation = {
            'user': 'tech_lover',
            'item': 'laptop',
            'score': float(recommender.calculate_score('tech_lover', 'laptop'))
        }
        
        # Update metrics
        metrics['recommendations_generated'] += sum(len(recs) for recs in recommendations.values())
        
        return jsonify({
            'recommendations': recommendations,
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo/full', methods=['POST'])
def run_full_demo():
    """API endpoint to run complete demo"""
    try:
        # Run all demos and collect results
        results = {
            'vectors_created': metrics['vectors_created'],
            'similarity_calculations': metrics['similarity_calculations'],
            'matrices_created': metrics['matrices_created'],
            'recommendations_generated': metrics['recommendations_generated'],
            'total_operations': metrics['total_operations']
        }
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize the recommendation system
    initialize_recommender()
    
    # Start metrics update thread
    metrics_thread = threading.Thread(target=update_metrics, daemon=True)
    metrics_thread.start()
    
    print("🚀 Starting Linear Algebra Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔢 Linear Algebra system initialized and ready!")
    print("📈 NumPy version:", np.__version__)
    print("📊 Matplotlib status:", matplotlib.__version__)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
