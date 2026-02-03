"""
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
    print("\n1. Generating synthetic movie rating dataset...")
    interactions = create_sample_dataset(num_users=50, num_items=100, sparsity=0.9)
    
    print(f"   Created {len(interactions)} ratings from {interactions['user_id'].nunique()} users")
    print(f"   Item catalog: {interactions['item_id'].nunique()} movies")
    
    test_user = 5
    
    # User-based Collaborative Filtering
    print("\n2. User-Based Collaborative Filtering")
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
    print("\n3. Item-Based Collaborative Filtering")
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
    print("\n4. Method Comparison")
    print("   " + "-" * 50)
    print(f"   User-Based: {len(user_results.recommended_items)} recommendations")
    print(f"   Item-Based: {len(item_results.recommended_items)} recommendations")
    
    print("\n5. Production Insights")
    print("   " + "-" * 50)
    print("   • User-based CF: Better for diverse user preferences")
    print("   • Item-based CF: More stable, better for large user bases")
    print("   • Matrix factorization (SVD) scales better for production")
    print("   • Real-time recommendations use pre-computed similarity matrices")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_collaborative_filtering()
