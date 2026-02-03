#!/usr/bin/env python3
"""
Day 104: Collaborative Filtering - Complete File Generator
Generates all necessary files for the collaborative filtering implementation
"""

import os

def create_lesson_code():
    """Create lesson_code.py with collaborative filtering implementation"""
    content = '''"""
Day 104: Collaborative Filtering - From Scratch Implementation
Implementing collaborative filtering with matrix factorization and similarity computation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RecommendationResult:
    """Container for recommendation output"""
    user_id: int
    recommended_items: List[int]
    scores: List[float]
    method: str
    similarity_scores: Optional[List[float]] = None


class CollaborativeFiltering:
    """
    Collaborative Filtering implementation from scratch
    Supports both user-based and item-based approaches
    """
    
    def __init__(self, method: str = 'user-based', min_common_items: int = 2):
        """
        Initialize collaborative filtering engine
        
        Args:
            method: 'user-based' or 'item-based'
            min_common_items: Minimum common items for similarity computation
        """
        self.method = method
        self.min_common_items = min_common_items
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.interactions_df = None
        
    def fit(self, interactions: pd.DataFrame):
        """
        Build user-item interaction matrix
        
        Args:
            interactions: DataFrame with columns [user_id, item_id, rating]
        """
        self.interactions_df = interactions.copy()
        
        # Create user-item matrix (users as rows, items as columns)
        self.user_item_matrix = interactions.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        # Create item-user matrix (items as rows, users as columns)
        self.item_user_matrix = self.user_item_matrix.T
        
        print(f"Built matrix: {len(self.user_item_matrix)} users × "
              f"{len(self.user_item_matrix.columns)} items")
        print(f"Sparsity: {(1 - (len(interactions) / (len(self.user_item_matrix) * len(self.user_item_matrix.columns)))) * 100:.2f}%")
    
    def compute_user_similarity(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Compute similarity between a user and all other users
        
        Args:
            user_id: Target user ID
            top_k: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_vector = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
        
        # Compute cosine similarity with all users
        similarities = cosine_similarity(user_vector, self.user_item_matrix.values)[0]
        
        # Get top-k similar users (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        similar_users = [
            (int(self.user_item_matrix.index[idx]), float(similarities[idx]))
            for idx in similar_indices
            if similarities[idx] > 0
        ]
        
        return similar_users
    
    def compute_item_similarity(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Compute similarity between an item and all other items
        
        Args:
            item_id: Target item ID
            top_k: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if item_id not in self.item_user_matrix.index:
            return []
        
        item_vector = self.item_user_matrix.loc[item_id].values.reshape(1, -1)
        
        # Compute cosine similarity with all items
        similarities = cosine_similarity(item_vector, self.item_user_matrix.values)[0]
        
        # Get top-k similar items (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        similar_items = [
            (int(self.item_user_matrix.index[idx]), float(similarities[idx]))
            for idx in similar_indices
            if similarities[idx] > 0
        ]
        
        return similar_items
    
    def recommend_user_based(self, user_id: int, top_n: int = 5) -> RecommendationResult:
        """
        Generate recommendations using user-based collaborative filtering
        
        Args:
            user_id: Target user ID
            top_n: Number of recommendations to return
            
        Returns:
            RecommendationResult object
        """
        # Find similar users
        similar_users = self.compute_user_similarity(user_id, top_k=20)
        
        if not similar_users:
            return RecommendationResult(user_id, [], [], "user-based", [])
        
        # Get items the target user has already rated
        user_items = set(
            self.user_item_matrix.columns[
                self.user_item_matrix.loc[user_id] > 0
            ]
        )
        
        # Aggregate scores from similar users
        item_scores = {}
        similarity_scores = {}
        
        for similar_user_id, similarity in similar_users:
            similar_user_ratings = self.user_item_matrix.loc[similar_user_id]
            
            for item_id in similar_user_ratings.index:
                rating = similar_user_ratings[item_id]
                
                # Only consider items the target user hasn't rated
                if rating > 0 and item_id not in user_items:
                    if item_id not in item_scores:
                        item_scores[item_id] = {'weighted_sum': 0, 'similarity_sum': 0}
                    
                    item_scores[item_id]['weighted_sum'] += rating * similarity
                    item_scores[item_id]['similarity_sum'] += similarity
                    similarity_scores[item_id] = similarity
        
        # Calculate weighted average scores
        predictions = {
            item_id: scores['weighted_sum'] / scores['similarity_sum']
            for item_id, scores in item_scores.items()
            if scores['similarity_sum'] > 0
        }
        
        # Sort and get top-N
        top_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recommended_items = [int(item_id) for item_id, _ in top_items]
        scores = [float(score) for _, score in top_items]
        sim_scores = [float(similarity_scores.get(item_id, 0.0)) for item_id in recommended_items]
        
        return RecommendationResult(user_id, recommended_items, scores, "user-based", sim_scores)
    
    def recommend_item_based(self, user_id: int, top_n: int = 5) -> RecommendationResult:
        """
        Generate recommendations using item-based collaborative filtering
        
        Args:
            user_id: Target user ID
            top_n: Number of recommendations to return
            
        Returns:
            RecommendationResult object
        """
        if user_id not in self.user_item_matrix.index:
            return RecommendationResult(user_id, [], [], "item-based", [])
        
        # Get items the user has rated
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        if not rated_items:
            return RecommendationResult(user_id, [], [], "item-based", [])
        
        # For each rated item, find similar items
        item_scores = {}
        similarity_scores = {}
        
        for rated_item_id in rated_items:
            user_rating = user_ratings[rated_item_id]
            similar_items = self.compute_item_similarity(rated_item_id, top_k=10)
            
            for similar_item_id, similarity in similar_items:
                # Skip items user has already rated
                if similar_item_id not in rated_items:
                    if similar_item_id not in item_scores:
                        item_scores[similar_item_id] = {'weighted_sum': 0, 'similarity_sum': 0}
                    
                    item_scores[similar_item_id]['weighted_sum'] += user_rating * similarity
                    item_scores[similar_item_id]['similarity_sum'] += similarity
                    similarity_scores[similar_item_id] = max(
                        similarity_scores.get(similar_item_id, 0.0),
                        similarity
                    )
        
        # Calculate weighted average scores
        predictions = {
            item_id: scores['weighted_sum'] / scores['similarity_sum']
            for item_id, scores in item_scores.items()
            if scores['similarity_sum'] > 0
        }
        
        # Sort and get top-N
        top_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recommended_items = [int(item_id) for item_id, _ in top_items]
        scores = [float(score) for _, score in top_items]
        sim_scores = [float(similarity_scores.get(item_id, 0.0)) for item_id in recommended_items]
        
        return RecommendationResult(user_id, recommended_items, scores, "item-based", sim_scores)
    
    def recommend(self, user_id: int, top_n: int = 5) -> RecommendationResult:
        """
        Generate recommendations based on the configured method
        
        Args:
            user_id: Target user ID
            top_n: Number of recommendations to return
            
        Returns:
            RecommendationResult object
        """
        if self.method == 'user-based':
            return self.recommend_user_based(user_id, top_n)
        else:
            return self.recommend_item_based(user_id, top_n)


def create_sample_dataset(num_users: int = 50, num_items: int = 100, 
                         sparsity: float = 0.9) -> pd.DataFrame:
    """
    Create synthetic movie rating dataset
    
    Args:
        num_users: Number of users
        num_items: Number of items
        sparsity: Desired sparsity (0.9 = 90% sparse)
        
    Returns:
        DataFrame with columns [user_id, item_id, rating]
    """
    np.random.seed(42)
    
    # Calculate number of ratings to achieve desired sparsity
    total_possible = num_users * num_items
    num_ratings = int(total_possible * (1 - sparsity))
    
    # Generate interactions
    interactions = []
    user_item_pairs = set()
    
    while len(interactions) < num_ratings:
        user_id = np.random.randint(0, num_users)
        item_id = np.random.randint(0, num_items)
        
        pair = (user_id, item_id)
        if pair not in user_item_pairs:
            user_item_pairs.add(pair)
            # Ratings from 1-5, with some bias toward higher ratings
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating
            })
    
    return pd.DataFrame(interactions)


def demonstrate_collaborative_filtering():
    """Main demonstration of collaborative filtering"""
    
    print("=" * 70)
    print("Day 104: Collaborative Filtering - From Scratch Implementation")
    print("=" * 70)
    
    # Create dataset
    print("\\n1. Generating synthetic movie rating dataset...")
    interactions = create_sample_dataset(num_users=50, num_items=100, sparsity=0.9)
    
    print(f"   Created {len(interactions)} ratings from {interactions['user_id'].nunique()} users")
    print(f"   Item catalog: {interactions['item_id'].nunique()} movies")
    
    test_user = 5
    
    # User-based Collaborative Filtering
    print("\\n2. User-Based Collaborative Filtering")
    print("   " + "-" * 50)
    cf_user = CollaborativeFiltering(method='user-based')
    cf_user.fit(interactions)
    
    user_results = cf_user.recommend(test_user, top_n=5)
    print(f"   Recommendations for User {test_user}:")
    for item, score, sim in zip(user_results.recommended_items, 
                                user_results.scores, 
                                user_results.similarity_scores or []):
        print(f"      Item {item}: Score={score:.3f}, Similarity={sim:.3f}")
    
    # Item-based Collaborative Filtering
    print("\\n3. Item-Based Collaborative Filtering")
    print("   " + "-" * 50)
    cf_item = CollaborativeFiltering(method='item-based')
    cf_item.fit(interactions)
    
    item_results = cf_item.recommend(test_user, top_n=5)
    print(f"   Recommendations for User {test_user}:")
    for item, score, sim in zip(item_results.recommended_items,
                                item_results.scores,
                                item_results.similarity_scores or []):
        print(f"      Item {item}: Score={score:.3f}, Similarity={sim:.3f}")
    
    # Compare approaches
    print("\\n4. Method Comparison")
    print("   " + "-" * 50)
    print(f"   User-Based: {len(user_results.recommended_items)} recommendations")
    print(f"   Item-Based: {len(item_results.recommended_items)} recommendations")
    
    print("\\n5. Production Insights")
    print("   " + "-" * 50)
    print("   • User-based CF: Better for diverse user preferences")
    print("   • Item-based CF: More stable, better for large user bases")
    print("   • Matrix factorization (SVD) scales better for production")
    print("   • Real-time recommendations use pre-computed similarity matrices")
    
    print("\\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_collaborative_filtering()
'''
    return content


def create_test_lesson():
    """Create test_lesson.py"""
    content = '''"""
Test suite for Day 104: Collaborative Filtering
"""

import pytest
import numpy as np
import pandas as pd
from lesson_code import (
    CollaborativeFiltering,
    create_sample_dataset,
    RecommendationResult
)


class TestCollaborativeFiltering:
    """Test collaborative filtering implementation"""
    
    def test_initialization(self):
        """Test engine can be initialized"""
        cf = CollaborativeFiltering(method='user-based', min_common_items=2)
        assert cf.method == 'user-based'
        assert cf.min_common_items == 2
        assert cf.user_item_matrix is None
    
    def test_fit_creates_matrix(self):
        """Test fitting creates user-item matrix"""
        interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [10, 11, 10, 12, 11],
            'rating': [5, 4, 3, 5, 4]
        })
        
        cf = CollaborativeFiltering()
        cf.fit(interactions)
        
        assert cf.user_item_matrix is not None
        assert len(cf.user_item_matrix) == 3
        assert len(cf.user_item_matrix.columns) == 3
    
    def test_compute_user_similarity(self):
        """Test computing user similarity"""
        interactions = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3],
            'item_id': [10, 11, 12, 10, 11, 13, 12, 13],
            'rating': [5, 4, 3, 5, 4, 5, 3, 5]
        })
        
        cf = CollaborativeFiltering(method='user-based')
        cf.fit(interactions)
        
        similar = cf.compute_user_similarity(1, top_k=2)
        assert len(similar) <= 2
        assert all(isinstance(user_id, int) for user_id, _ in similar)
        assert all(0 <= sim <= 1 for _, sim in similar)
    
    def test_recommend_user_based(self):
        """Test user-based recommendations"""
        interactions, _ = create_sample_dataset(), None
        if isinstance(interactions, tuple):
            interactions = interactions[0]
        
        cf = CollaborativeFiltering(method='user-based')
        cf.fit(interactions)
        
        result = cf.recommend(user_id=5, top_n=5)
        
        assert isinstance(result, RecommendationResult)
        assert result.user_id == 5
        assert len(result.recommended_items) <= 5
        assert result.method == "user-based"
    
    def test_recommend_item_based(self):
        """Test item-based recommendations"""
        interactions = create_sample_dataset()
        if isinstance(interactions, tuple):
            interactions = interactions[0]
        
        cf = CollaborativeFiltering(method='item-based')
        cf.fit(interactions)
        
        result = cf.recommend(user_id=5, top_n=5)
        
        assert isinstance(result, RecommendationResult)
        assert result.method == "item-based"
        assert len(result.recommended_items) <= 5
    
    def test_no_duplicate_recommendations(self):
        """Test no duplicate recommendations"""
        interactions = create_sample_dataset()
        if isinstance(interactions, tuple):
            interactions = interactions[0]
        
        cf = CollaborativeFiltering()
        cf.fit(interactions)
        
        result = cf.recommend(user_id=5, top_n=10)
        
        assert len(result.recommended_items) == len(set(result.recommended_items))


class TestDataGeneration:
    """Test sample dataset generation"""
    
    def test_create_sample_dataset(self):
        """Test dataset creation"""
        interactions = create_sample_dataset()
        if isinstance(interactions, tuple):
            interactions = interactions[0]
        
        assert isinstance(interactions, pd.DataFrame)
        assert 'user_id' in interactions.columns
        assert 'item_id' in interactions.columns
        assert 'rating' in interactions.columns
    
    def test_ratings_in_range(self):
        """Test ratings are in valid range"""
        interactions = create_sample_dataset()
        if isinstance(interactions, tuple):
            interactions = interactions[0]
        
        assert interactions['rating'].min() >= 1
        assert interactions['rating'].max() <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    return content


def create_dashboard():
    """Create dashboard.py for collaborative filtering"""
    content = '''"""
Dashboard for Day 104: Collaborative Filtering
"""

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import json
import time
import sys
import os
from datetime import datetime
import numpy as np

# Add parent directory to path to import lesson_code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lesson_code import (
    CollaborativeFiltering,
    create_sample_dataset
)

app = Flask(__name__)
CORS(app)

# Global state
interactions_df = None
cf_user_engine = None
cf_item_engine = None

metrics_data = {
    'timestamp': datetime.now().isoformat(),
    'user_based_metrics': {
        'users_processed': 0,
        'recommendations_generated': 0,
        'avg_similarity_score': 0.0,
        'matrix_size': '0x0'
    },
    'item_based_metrics': {
        'items_processed': 0,
        'recommendations_generated': 0,
        'avg_similarity_score': 0.0,
        'matrix_size': '0x0'
    },
    'demo_metrics': {
        'demos_run': 0,
        'last_demo_time': None,
        'success_rate': 100.0
    }
}

def initialize_engines():
    """Initialize collaborative filtering engines with sample data"""
    global interactions_df, cf_user_engine, cf_item_engine
    
    if interactions_df is None:
        interactions_df = create_sample_dataset()
        
        cf_user_engine = CollaborativeFiltering(method='user-based')
        cf_user_engine.fit(interactions_df)
        
        cf_item_engine = CollaborativeFiltering(method='item-based')
        cf_item_engine.fit(interactions_df)
        
        # Update metrics
        metrics_data['user_based_metrics']['users_processed'] = len(cf_user_engine.user_item_matrix)
        metrics_data['user_based_metrics']['matrix_size'] = f"{len(cf_user_engine.user_item_matrix)}x{len(cf_user_engine.user_item_matrix.columns)}"
        metrics_data['item_based_metrics']['items_processed'] = len(cf_item_engine.item_user_matrix)
        metrics_data['item_based_metrics']['matrix_size'] = f"{len(cf_item_engine.item_user_matrix)}x{len(cf_item_engine.item_user_matrix.columns)}"

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Day 104: Collaborative Filtering Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
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
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Collaborative Filtering Dashboard</h1>
        
        <div class="control-panel">
            <div class="control-title">🎮 Demo Controls</div>
            <div class="demo-buttons">
                <button class="demo-btn" onclick="runUserBased()">
                    👥 Run User-Based CF
                </button>
                <button class="demo-btn" onclick="runItemBased()">
                    📝 Run Item-Based CF
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
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    const grid = document.getElementById('metrics-grid');
                    grid.innerHTML = '';
                    
                    // User-Based Metrics
                    const ubCard = createMetricCard(
                        'User-Based CF',
                        data.user_based_metrics.users_processed,
                        `Matrix: ${data.user_based_metrics.matrix_size}`,
                        '👥'
                    );
                    grid.appendChild(ubCard);
                    
                    // Item-Based Metrics
                    const ibCard = createMetricCard(
                        'Item-Based CF',
                        data.item_based_metrics.items_processed,
                        `Matrix: ${data.item_based_metrics.matrix_size}`,
                        '📝'
                    );
                    grid.appendChild(ibCard);
                    
                    // User-Based Recommendations
                    const ubRecsCard = createMetricCard(
                        'User-Based Recommendations',
                        data.user_based_metrics.recommendations_generated,
                        `Avg Score: ${data.user_based_metrics.avg_similarity_score.toFixed(3)}`,
                        '🎯'
                    );
                    grid.appendChild(ubRecsCard);
                    
                    // Item-Based Recommendations
                    const ibRecsCard = createMetricCard(
                        'Item-Based Recommendations',
                        data.item_based_metrics.recommendations_generated,
                        `Avg Score: ${data.item_based_metrics.avg_similarity_score.toFixed(3)}`,
                        '🎯'
                    );
                    grid.appendChild(ibRecsCard);
                    
                    // Demo Metrics
                    const demoCard = createMetricCard(
                        'Demos Run',
                        data.demo_metrics.demos_run,
                        `Success Rate: ${data.demo_metrics.success_rate.toFixed(1)}%`,
                        '🚀'
                    );
                    grid.appendChild(demoCard);
                    
                    document.getElementById('timestamp').textContent = 
                        `Last updated: ${new Date(data.timestamp).toLocaleString()}`;
                })
                .catch(error => {
                    console.error('Error fetching metrics:', error);
                });
        }
        
        function createMetricCard(title, value, label, emoji) {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <div class="metric-title">${emoji} ${title}</div>
                <div class="metric-value">${value}</div>
                <div class="metric-label">${label}</div>
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
        
        function runUserBased() {
            showStatus('Running user-based collaborative filtering demo...');
            fetch('/api/run-user-based', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`✅ User-based CF complete! Generated ${data.results.recommendations} recommendations`);
                    updateDashboard();
                } else {
                    showStatus('❌ Demo failed: ' + data.message, true);
                }
            })
            .catch(error => {
                showStatus('❌ Error running demo: ' + error.message, true);
            });
        }
        
        function runItemBased() {
            showStatus('Running item-based collaborative filtering demo...');
            fetch('/api/run-item-based', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    showStatus(`✅ Item-based CF complete! Generated ${data.results.recommendations} recommendations`);
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
    initialize_engines()
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    initialize_engines()
    metrics_data['timestamp'] = datetime.now().isoformat()
    return jsonify(metrics_data)

@app.route('/api/update-demo', methods=['POST'])
def update_demo_metrics():
    """Update demo metrics"""
    metrics_data['demo_metrics']['demos_run'] += 1
    metrics_data['demo_metrics']['last_demo_time'] = datetime.now().isoformat()
    return jsonify({'status': 'success'})

@app.route('/api/run-user-based', methods=['POST'])
def run_user_based():
    """Run user-based collaborative filtering demo"""
    try:
        initialize_engines()
        result = cf_user_engine.recommend(user_id=5, top_n=5)
        
        metrics_data['user_based_metrics']['recommendations_generated'] += len(result.recommended_items)
        if result.scores:
            metrics_data['user_based_metrics']['avg_similarity_score'] = np.mean(result.scores)
        
        return jsonify({
            'status': 'success',
            'results': {
                'recommendations': len(result.recommended_items),
                'items': result.recommended_items,
                'scores': result.scores
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/run-item-based', methods=['POST'])
def run_item_based():
    """Run item-based collaborative filtering demo"""
    try:
        initialize_engines()
        result = cf_item_engine.recommend(user_id=5, top_n=5)
        
        metrics_data['item_based_metrics']['recommendations_generated'] += len(result.recommended_items)
        if result.scores:
            metrics_data['item_based_metrics']['avg_similarity_score'] = np.mean(result.scores)
        
        return jsonify({
            'status': 'success',
            'results': {
                'recommendations': len(result.recommended_items),
                'items': result.recommended_items,
                'scores': result.scores
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Collaborative Filtering Dashboard on http://localhost:5000")
    print("📊 Access dashboard at: http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/api/metrics")
    app.run(host='0.0.0.0', port=5000, debug=True)
'''
    return content


def create_startup_script():
    """Create startup.sh script"""
    content = '''#!/bin/bash

# Startup script for Day 104: Collaborative Filtering Dashboard

echo "🚀 Starting Day 104: Collaborative Filtering Dashboard..."
echo "===================================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || exit 1

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if dashboard is already running
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Dashboard already running on port 5000"
    echo "   Stopping existing instance..."
    pkill -9 -f "python.*dashboard.py" || true
    sleep 2
    # Double check
    if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "❌ Failed to stop existing dashboard"
        exit 1
    fi
fi

# Also check for any dashboard processes
DASHBOARD_COUNT=$(ps aux | grep -E "python.*dashboard.py" | grep -v grep | wc -l)
if [ "$DASHBOARD_COUNT" -gt 0 ]; then
    echo "⚠️  Found $DASHBOARD_COUNT existing dashboard process(es), stopping..."
    pkill -9 -f "python.*dashboard.py" || true
    sleep 2
fi

# Change to sources directory
cd sources || exit 1

# Start dashboard
echo "📊 Starting dashboard on http://localhost:5000"
nohup python dashboard.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!

echo "✅ Dashboard started (PID: $DASHBOARD_PID)"
echo ""
echo "📱 Access dashboard at: http://localhost:5000"
echo "📡 API endpoint: http://localhost:5000/api/metrics"
echo ""
echo "To stop the dashboard, run:"
echo "   kill $DASHBOARD_PID"
echo "   or"
echo "   pkill -f 'python.*dashboard.py'"
echo ""

# Wait a moment for startup
sleep 3

# Check if dashboard started successfully
if ps -p $DASHBOARD_PID > /dev/null; then
    echo "✅ Dashboard is running successfully!"
else
    echo "❌ Dashboard failed to start"
    exit 1
fi
'''
    return content


def main():
    """Generate all files"""
    print("Generating Day 104: Collaborative Filtering files...")
    
    # Create sources directory
    os.makedirs('sources', exist_ok=True)
    
    # Write lesson_code.py
    with open('lesson_code.py', 'w') as f:
        f.write(create_lesson_code())
    print("✓ lesson_code.py")
    
    # Write test_lesson.py
    with open('test_lesson.py', 'w') as f:
        f.write(create_test_lesson())
    print("✓ test_lesson.py")
    
    # Write dashboard.py in sources/
    with open('sources/dashboard.py', 'w') as f:
        f.write(create_dashboard())
    print("✓ sources/dashboard.py")
    
    # Write startup.sh in sources/
    with open('sources/startup.sh', 'w') as f:
        f.write(create_startup_script())
    os.chmod('sources/startup.sh', 0o755)
    print("✓ sources/startup.sh")
    
    print("\\nAll files generated successfully!")


if __name__ == "__main__":
    main()
