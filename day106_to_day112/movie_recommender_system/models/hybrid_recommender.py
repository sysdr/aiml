"""
Hybrid Recommender System combining collaborative and content-based filtering.

Implements adaptive blending based on data availability and user experience.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import pickle
from models.collaborative_filtering import CollaborativeFilter
from models.content_based import ContentBasedFilter


class HybridRecommender:
    """
    Production-grade hybrid recommendation system.
    
    Netflix's hybrid approach:
    - New users: Content-heavy (80% content, 20% collaborative)
    - Established users: Collaborative-heavy (80% collaborative, 20% content)
    - Dynamic blending adapts to data availability
    """
    
    def __init__(
        self,
        collaborative_model: CollaborativeFilter,
        content_model: ContentBasedFilter,
        min_alpha: float = 0.2,
        max_alpha: float = 0.8
    ):
        """
        Args:
            collaborative_model: Trained collaborative filter
            content_model: Trained content-based filter
            min_alpha: Minimum collaborative weight (for new users)
            max_alpha: Maximum collaborative weight (for established users)
        """
        self.collab_model = collaborative_model
        self.content_model = content_model
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
    def _compute_blend_weight(
        self,
        user_rating_count: int,
        threshold: int = 20
    ) -> float:
        """
        Compute adaptive blending weight (alpha) based on user experience.
        
        Args:
            user_rating_count: Number of ratings user has provided
            threshold: Rating count at which to reach max_alpha
            
        Returns:
            alpha: Collaborative weight [min_alpha, max_alpha]
            
        Cold-start users (few ratings) rely more on content.
        Established users (many ratings) rely more on collaborative signals.
        """
        alpha = self.min_alpha + (
            (self.max_alpha - self.min_alpha) *
            min(user_rating_count / threshold, 1.0)
        )
        return alpha
    
    def predict(
        self,
        user_id: int,
        item_id: int,
        user_rated_items: Optional[np.ndarray] = None,
        user_ratings: Optional[np.ndarray] = None,
        user_rating_count: Optional[int] = None
    ) -> float:
        """
        Predict rating using hybrid approach.
        
        Formula: score = α × collab_score + (1-α) × content_score
        where α adapts based on user rating history.
        
        Args:
            user_id: User index
            item_id: Item index
            user_rated_items: Items user has rated (for content-based)
            user_ratings: Ratings for those items
            user_rating_count: Total ratings by user (for alpha computation)
            
        Returns:
            Predicted rating (1-5 scale)
        """
        
        # Get collaborative prediction
        collab_score = self.collab_model.predict(user_id, item_id)
        
        # Get content-based prediction
        if user_rated_items is not None and user_ratings is not None:
            content_score = self.content_model.predict(
                user_id, item_id, user_rated_items, user_ratings
            )
        else:
            content_score = 3.0  # Neutral baseline
        
        # Compute adaptive blend weight
        if user_rating_count is None:
            user_rating_count = len(user_rated_items) if user_rated_items is not None else 0
        
        alpha = self._compute_blend_weight(user_rating_count)
        
        # Hybrid prediction
        hybrid_score = alpha * collab_score + (1 - alpha) * content_score
        
        return np.clip(hybrid_score, 1.0, 5.0)
    
    def recommend(
        self,
        user_id: int,
        user_rated_items: np.ndarray,
        user_ratings: np.ndarray,
        n_recommendations: int = 10,
        diversity_weight: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate top-N recommendations using hybrid approach.
        
        Args:
            user_id: User index
            user_rated_items: Items user has rated
            user_ratings: Ratings for those items
            n_recommendations: Number of recommendations to return
            diversity_weight: Weight for diversity injection (0-1)
            
        Returns:
            item_ids: Recommended item indices
            scores: Combined scores
            metadata: Additional info (blend weight, coverage, etc.)
        """
        
        # Compute blend weight
        alpha = self._compute_blend_weight(len(user_rated_items))
        
        # Get collaborative recommendations
        collab_items, collab_scores = self.collab_model.recommend(
            user_id,
            n_recommendations=n_recommendations * 3,  # Get more candidates
            exclude_rated=user_rated_items
        )
        
        # Get content-based recommendations
        content_items, content_scores = self.content_model.recommend(
            user_rated_items,
            user_ratings,
            n_recommendations=n_recommendations * 3,
            exclude_rated=user_rated_items
        )
        
        # Combine candidates
        all_candidates = np.unique(np.concatenate([collab_items, content_items]))
        
        # Score each candidate using hybrid approach
        hybrid_scores = []
        for item_id in all_candidates:
            # Get individual scores
            c_score = (collab_scores[np.where(collab_items == item_id)[0][0]]
                      if item_id in collab_items else 3.0)
            co_score = (content_scores[np.where(content_items == item_id)[0][0]]
                       if item_id in content_items else 3.0)
            
            # Hybrid score
            h_score = alpha * c_score + (1 - alpha) * co_score
            hybrid_scores.append(h_score)
        
        hybrid_scores = np.array(hybrid_scores)
        
        # Apply diversity injection (prevent filter bubbles)
        if diversity_weight > 0:
            # Add small random noise to promote diversity
            noise = np.random.randn(len(hybrid_scores)) * diversity_weight
            hybrid_scores += noise
        
        # Get top-N
        top_indices = np.argsort(hybrid_scores)[::-1][:n_recommendations]
        final_items = all_candidates[top_indices]
        final_scores = hybrid_scores[top_indices]
        
        # Metadata for analysis
        metadata = {
            'alpha': alpha,
            'user_experience': 'new' if alpha < 0.4 else 'established',
            'n_candidates': len(all_candidates),
            'collab_weight': alpha,
            'content_weight': 1 - alpha
        }
        
        return final_items, final_scores, metadata
    
    def save(self, filepath: str):
        """Save hybrid model configuration."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'min_alpha': self.min_alpha,
                'max_alpha': self.max_alpha
            }, f)
    
    def load(self, filepath: str):
        """Load hybrid model configuration."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.min_alpha = data['min_alpha']
            self.max_alpha = data['max_alpha']

