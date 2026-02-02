"""
Day 103: Recommender Systems Theory
Implementing the three core recommendation approaches with real-world patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
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


class CollaborativeFilteringEngine:
    """
    User-based collaborative filtering implementation
    Mirrors Netflix's early recommendation approach
    """
    
    def __init__(self, min_common_items: int = 2):
        self.min_common_items = min_common_items
        self.user_item_matrix = None
        self.item_users = {}  # Inverted index for efficiency
        
    def fit(self, interactions: pd.DataFrame):
        """
        Build user-item interaction matrix
        
        Args:
            interactions: DataFrame with columns [user_id, item_id, rating]
        """
        # Create pivot table: users as rows, items as columns
        self.user_item_matrix = interactions.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        # Build inverted index: item -> set of users who rated it
        for item_id in self.user_item_matrix.columns:
            self.item_users[item_id] = set(
                self.user_item_matrix[self.user_item_matrix[item_id] > 0].index
            )
        
        print(f"Built matrix: {len(self.user_item_matrix)} users × "
              f"{len(self.user_item_matrix.columns)} items")
        
    def find_similar_users(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find k most similar users using cosine similarity
        Production systems use approximate nearest neighbors for scale
        """
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_vector = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
        
        # Compute similarity with all users (in production, use ANN)
        similarities = cosine_similarity(user_vector, self.user_item_matrix.values)[0]
        
        # Get top-k similar users (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        similar_users = [
            (self.user_item_matrix.index[idx], similarities[idx])
            for idx in similar_indices
            if similarities[idx] > 0  # Only positive correlations
        ]
        
        return similar_users
    
    def recommend(self, user_id: int, top_n: int = 5) -> RecommendationResult:
        """
        Generate recommendations using collaborative filtering
        Algorithm: Aggregate ratings from similar users, weighted by similarity
        """
        # Find similar users
        similar_users = self.find_similar_users(user_id, top_k=20)
        
        if not similar_users:
            return RecommendationResult(user_id, [], [], "collaborative_filtering")
        
        # Get items the target user has already rated
        user_items = set(
            self.user_item_matrix.columns[
                self.user_item_matrix.loc[user_id] > 0
            ]
        )
        
        # Aggregate scores from similar users
        item_scores = {}
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
        
        # Calculate weighted average scores
        predictions = {
            item_id: scores['weighted_sum'] / scores['similarity_sum']
            for item_id, scores in item_scores.items()
            if scores['similarity_sum'] > 0
        }
        
        # Sort and get top-N
        top_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recommended_items = [item_id for item_id, _ in top_items]
        scores = [score for _, score in top_items]
        
        return RecommendationResult(user_id, recommended_items, scores, "collaborative_filtering")


class ContentBasedEngine:
    """
    Content-based filtering using item features
    Similar to Pandora's Music Genome Project approach
    """
    
    def __init__(self):
        self.item_features = None
        self.user_profiles = {}
        
    def fit(self, item_features: pd.DataFrame, interactions: pd.DataFrame):
        """
        Build item feature matrix and user preference profiles
        
        Args:
            item_features: DataFrame with item_id and feature columns
            interactions: DataFrame with user_id, item_id, rating
        """
        self.item_features = item_features.set_index('item_id')
        
        # Build user profiles by aggregating features of items they liked
        for user_id in interactions['user_id'].unique():
            user_interactions = interactions[interactions['user_id'] == user_id]
            
            # Weight features by ratings
            liked_items = user_interactions[user_interactions['rating'] >= 4]['item_id']
            
            if len(liked_items) > 0:
                # Average feature vectors of liked items
                user_profile = self.item_features.loc[liked_items].mean(axis=0)
                self.user_profiles[user_id] = user_profile.values
        
        print(f"Built profiles for {len(self.user_profiles)} users using "
              f"{len(self.item_features.columns)} features")
    
    def recommend(self, user_id: int, top_n: int = 5, 
                   exclude_items: Set[int] = None) -> RecommendationResult:
        """
        Recommend items similar to user's preference profile
        """
        if user_id not in self.user_profiles:
            return RecommendationResult(user_id, [], [], "content_based")
        
        user_profile = self.user_profiles[user_id].reshape(1, -1)
        
        # Compute similarity between user profile and all items
        item_similarities = cosine_similarity(user_profile, self.item_features.values)[0]
        
        # Create item-score pairs
        item_scores = list(zip(self.item_features.index, item_similarities))
        
        # Filter out items user has already seen
        if exclude_items:
            item_scores = [(item, score) for item, score in item_scores 
                          if item not in exclude_items]
        
        # Sort and get top-N
        top_items = sorted(item_scores, key=lambda x: x[1], reverse=True)[:top_n]
        
        recommended_items = [int(item_id) for item_id, _ in top_items]
        scores = [float(score) for _, score in top_items]
        
        return RecommendationResult(user_id, recommended_items, scores, "content_based")


class HybridRecommender:
    """
    Combines collaborative and content-based approaches
    Mirrors production systems like Amazon and Netflix
    """
    
    def __init__(self, collaborative_weight: float = 0.6, content_weight: float = 0.4):
        self.cf_engine = CollaborativeFilteringEngine()
        self.cb_engine = ContentBasedEngine()
        self.cf_weight = collaborative_weight
        self.cb_weight = content_weight
        
    def fit(self, interactions: pd.DataFrame, item_features: pd.DataFrame):
        """Fit both engines"""
        print("Training collaborative filtering engine...")
        self.cf_engine.fit(interactions)
        
        print("Training content-based engine...")
        self.cb_engine.fit(item_features, interactions)
        
    def recommend(self, user_id: int, top_n: int = 5) -> RecommendationResult:
        """
        Hybrid recommendation using weighted combination
        
        Production systems use more sophisticated ensembles (gradient boosting, neural nets)
        This demonstrates the core concept
        """
        # Get recommendations from both engines
        cf_results = self.cf_engine.recommend(user_id, top_n=20)
        
        # Get items user has interacted with for exclusion
        if user_id in self.cf_engine.user_item_matrix.index:
            exclude_items = set(
                self.cf_engine.user_item_matrix.columns[
                    self.cf_engine.user_item_matrix.loc[user_id] > 0
                ]
            )
        else:
            exclude_items = set()
            
        cb_results = self.cb_engine.recommend(user_id, top_n=20, exclude_items=exclude_items)
        
        # Normalize scores to [0, 1] range
        def normalize_scores(scores):
            if not scores or max(scores) == min(scores):
                return scores
            min_s, max_s = min(scores), max(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]
        
        # Create combined score dictionary
        combined_scores = {}
        
        # Add collaborative filtering scores
        cf_norm = normalize_scores(cf_results.scores)
        for item, score in zip(cf_results.recommended_items, cf_norm):
            combined_scores[item] = self.cf_weight * score
        
        # Add content-based scores
        cb_norm = normalize_scores(cb_results.scores)
        for item, score in zip(cb_results.recommended_items, cb_norm):
            if item in combined_scores:
                combined_scores[item] += self.cb_weight * score
            else:
                combined_scores[item] = self.cb_weight * score
        
        # Sort by combined score
        top_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        recommended_items = [item for item, _ in top_items]
        scores = [score for _, score in top_items]
        
        return RecommendationResult(user_id, recommended_items, scores, "hybrid")


def create_sample_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic movie rating dataset
    Simulates user preferences and item features
    """
    np.random.seed(42)
    
    # Create users with different taste profiles
    num_users = 50
    num_items = 100
    
    # Generate interactions (sparse matrix - users rate ~10% of items)
    interactions = []
    for user_id in range(num_users):
        # Each user rates 8-12 items
        num_ratings = np.random.randint(8, 13)
        rated_items = np.random.choice(num_items, num_ratings, replace=False)
        
        for item_id in rated_items:
            # Ratings from 1-5
            rating = np.random.randint(1, 6)
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    # Generate item features (5 features per item)
    # In production: genre vectors, audio features, text embeddings, etc.
    item_features = []
    for item_id in range(num_items):
        features = {
            'item_id': item_id,
            'feature_1': np.random.rand(),  # e.g., action score
            'feature_2': np.random.rand(),  # e.g., comedy score
            'feature_3': np.random.rand(),  # e.g., drama score
            'feature_4': np.random.rand(),  # e.g., release year (normalized)
            'feature_5': np.random.rand(),  # e.g., popularity score
        }
        item_features.append(features)
    
    item_features_df = pd.DataFrame(item_features)
    
    return interactions_df, item_features_df


def demonstrate_recommender_systems():
    """Main demonstration of all three recommender approaches"""
    
    print("=" * 70)
    print("Day 103: Recommender Systems Theory - Implementation Demo")
    print("=" * 70)
    
    # Create dataset
    print("\n1. Generating synthetic movie rating dataset...")
    interactions, item_features = create_sample_dataset()
    
    print(f"   Created {len(interactions)} ratings from {interactions['user_id'].nunique()} users")
    print(f"   Item catalog: {interactions['item_id'].nunique()} movies")
    print(f"   Sparsity: {len(interactions) / (interactions['user_id'].nunique() * interactions['item_id'].nunique()) * 100:.2f}%")
    
    # Test user
    test_user = 5
    
    # Collaborative Filtering
    print("\n2. Collaborative Filtering Demonstration")
    print("   " + "-" * 50)
    cf_engine = CollaborativeFilteringEngine()
    cf_engine.fit(interactions)
    
    cf_results = cf_engine.recommend(test_user, top_n=5)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(cf_results.recommended_items, cf_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    # Content-Based Filtering
    print("\n3. Content-Based Filtering Demonstration")
    print("   " + "-" * 50)
    cb_engine = ContentBasedEngine()
    cb_engine.fit(item_features, interactions)
    
    user_rated_items = set(interactions[interactions['user_id'] == test_user]['item_id'])
    cb_results = cb_engine.recommend(test_user, top_n=5, exclude_items=user_rated_items)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(cb_results.recommended_items, cb_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    # Hybrid Approach
    print("\n4. Hybrid Recommender Demonstration")
    print("   " + "-" * 50)
    hybrid = HybridRecommender(collaborative_weight=0.6, content_weight=0.4)
    hybrid.fit(interactions, item_features)
    
    hybrid_results = hybrid.recommend(test_user, top_n=5)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(hybrid_results.recommended_items, hybrid_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    # Compare approaches
    print("\n5. Method Comparison")
    print("   " + "-" * 50)
    print(f"   Collaborative: {len(cf_results.recommended_items)} recommendations")
    print(f"   Content-Based: {len(cb_results.recommended_items)} recommendations")
    print(f"   Hybrid:        {len(hybrid_results.recommended_items)} recommendations")
    
    print("\n6. Production Insights")
    print("   " + "-" * 50)
    print("   • Netflix uses hybrid approach with 200+ models in ensemble")
    print("   • Candidate generation (CF) runs in <20ms using ANN search")
    print("   • Ranking stage (content features) processes 500+ items in <50ms")
    print("   • A/B testing determines optimal collaborative/content weights")
    print("   • Cold-start items rely more on content features")
    
    print("\n" + "=" * 70)
    print("Demo complete! Tomorrow: Implement collaborative filtering from scratch")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_recommender_systems()
