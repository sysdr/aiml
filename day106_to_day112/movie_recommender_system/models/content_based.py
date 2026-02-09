"""
Content-Based Filtering using item features.

Analyzes movie attributes (genres, release year) to find
similar items and generate recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional
import pickle


class ContentBasedFilter:
    """
    Content-based recommendation using item feature similarity.
    
    Spotify uses deep audio analysis (CNNs on raw waveforms) to
    extract features like tempo, energy, valence. For new tracks,
    audio similarity enables instant recommendations before any
    user interaction.
    """
    
    def __init__(self):
        self.item_features = None
        self.similarity_matrix = None
        self.tfidf_vectorizer = None
        
    def fit(
        self,
        movies_df: pd.DataFrame,
        genre_features: np.ndarray
    ) -> 'ContentBasedFilter':
        """
        Build item feature matrix and compute similarities.
        
        Args:
            movies_df: Movie metadata DataFrame
            genre_features: Binary genre matrix (n_movies × n_genres)
        """
        
        # Extract release year as feature
        movies_df['year'] = movies_df['release_date'].str.extract(r'(\d{4})')
        movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
        movies_df['year'] = movies_df['year'].fillna(movies_df['year'].median())
        
        # Normalize year to [0, 1] range
        year_normalized = (
            (movies_df['year'] - movies_df['year'].min()) /
            (movies_df['year'].max() - movies_df['year'].min())
        ).values.reshape(-1, 1)
        
        # Combine genre features with year
        # Shape: (n_movies, n_genres + 1)
        self.item_features = np.hstack([genre_features, year_normalized])
        
        # Compute item-item similarity matrix
        # Cosine similarity: measures angle between feature vectors
        # Range: [0, 1] where 1 = identical features
        self.similarity_matrix = cosine_similarity(self.item_features)
        
        return self
    
    def predict(
        self,
        user_id: int,
        item_id: int,
        user_profile: np.ndarray,
        user_ratings: np.ndarray
    ) -> float:
        """
        Predict rating based on content similarity to user's profile.
        
        Strategy: Find items similar to those the user rated highly,
        weight by rating and similarity.
        
        Args:
            user_id: User index (unused, for API consistency)
            item_id: Target item index
            user_profile: User's rated item indices
            user_ratings: User's ratings for those items
            
        Returns:
            Predicted rating (1-5 scale)
        """
        if len(user_profile) == 0:
            return 3.0  # Cold-start: return neutral rating
        
        # Get similarities between target item and user's rated items
        similarities = self.similarity_matrix[item_id, user_profile]
        
        # Weighted average: higher weight for more similar items
        # Formula: Σ(similarity × rating) / Σ(similarity)
        weighted_sum = np.dot(similarities, user_ratings)
        similarity_sum = np.sum(np.abs(similarities))
        
        if similarity_sum == 0:
            return np.mean(user_ratings)
        
        prediction = weighted_sum / similarity_sum
        
        # Clip to valid range
        return np.clip(prediction, 1.0, 5.0)
    
    def recommend(
        self,
        user_rated_items: np.ndarray,
        user_ratings: np.ndarray,
        n_recommendations: int = 10,
        exclude_rated: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate recommendations based on content similarity.
        
        Args:
            user_rated_items: Indices of items user has rated
            user_ratings: Ratings for those items
            n_recommendations: Number of items to recommend
            exclude_rated: Item indices to exclude
            
        Returns:
            item_ids: Recommended item indices
            scores: Content-based scores
        """
        n_items = self.similarity_matrix.shape[0]
        scores = np.zeros(n_items)
        
        # For each candidate item, compute weighted similarity
        # to user's positively-rated items (rating >= 4)
        positive_items = user_rated_items[user_ratings >= 4]
        
        if len(positive_items) > 0:
            # Average similarity to all positive items
            similarities = self.similarity_matrix[:, positive_items]
            scores = similarities.mean(axis=1)
        
        # Exclude already rated items
        if exclude_rated is not None:
            scores[exclude_rated] = -np.inf
        
        # Get top-N
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        top_scores = scores[top_indices]
        
        return top_indices, top_scores
    
    def get_similar_items(
        self,
        item_id: int,
        n_similar: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find most similar items based on content features.
        
        Used for "Users who liked X also liked Y" style recommendations.
        """
        similarities = self.similarity_matrix[item_id, :]
        
        # Exclude the item itself
        similarities[item_id] = -np.inf
        
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'item_features': self.item_features,
                'similarity_matrix': self.similarity_matrix
            }, f)
    
    def load(self, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.item_features = data['item_features']
            self.similarity_matrix = data['similarity_matrix']

