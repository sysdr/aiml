"""
Collaborative Filtering using Matrix Factorization (SVD).

Implements user-based and item-based collaborative filtering
using Singular Value Decomposition to learn latent factors.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from typing import Tuple, Optional
import pickle


class CollaborativeFilter:
    """
    Matrix Factorization using SVD for collaborative filtering.
    
    Netflix uses ALS (Alternating Least Squares) for distributed
    computation across Spark clusters. SVD provides similar results
    for smaller datasets with cleaner mathematical properties.
    """
    
    def __init__(self, n_factors: int = 50):
        """
        Args:
            n_factors: Number of latent factors (embedding dimensions).
                      Netflix uses 100-200 factors for production.
                      More factors = more nuanced preferences but
                      higher computation cost.
        """
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.predictions_matrix = None
        
    def fit(self, user_item_matrix: csr_matrix) -> 'CollaborativeFilter':
        """
        Decompose user-item matrix using SVD.
        
        Factorization: R ≈ U Σ V^T
        where:
            R: m×n ratings matrix (users × items)
            U: m×k user factors (user embeddings)
            Σ: k×k singular values (importance weights)
            V^T: k×n item factors (item embeddings)
        
        Prediction: r̂_ui = μ + U[u] @ Σ @ V[i]^T
        """
        
        # Compute global mean for baseline
        self.global_mean = user_item_matrix.data.mean()
        
        # Center the ratings (subtract mean)
        # This helps SVD focus on preference patterns
        # rather than absolute rating levels
        # Convert to float to avoid dtype issues
        user_item_centered = user_item_matrix.astype(np.float64).copy()
        user_item_centered.data -= self.global_mean
        
        # Perform SVD
        # svds returns: U (users), sigma (values), Vt (items)
        U, sigma, Vt = svds(user_item_centered, k=self.n_factors)
        
        # Store factors
        self.user_factors = U
        self.item_factors = Vt.T
        
        # Reconstruct predictions matrix
        # Shape: (n_users, n_items)
        sigma_diag = np.diag(sigma)
        self.predictions_matrix = (
            np.dot(np.dot(U, sigma_diag), Vt) + self.global_mean
        )
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for user-item pair.
        
        Args:
            user_id: User index (0-based)
            item_id: Item index (0-based)
            
        Returns:
            Predicted rating (typically 1-5 scale)
        """
        if self.predictions_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Bounds checking
        if user_id >= self.predictions_matrix.shape[0]:
            return self.global_mean  # Cold-start: return average
        if item_id >= self.predictions_matrix.shape[1]:
            return self.global_mean
        
        prediction = self.predictions_matrix[user_id, item_id]
        
        # Clip to valid rating range [1, 5]
        return np.clip(prediction, 1.0, 5.0)
    
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_rated: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User index
            n_recommendations: Number of items to recommend
            exclude_rated: Item indices already rated (to exclude)
            
        Returns:
            item_ids: Recommended item indices
            scores: Predicted ratings for recommended items
        """
        if self.predictions_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get all predictions for this user
        user_predictions = self.predictions_matrix[user_id, :]
        
        # Exclude already rated items
        if exclude_rated is not None:
            user_predictions[exclude_rated] = -np.inf
        
        # Get top-N items
        top_indices = np.argsort(user_predictions)[::-1][:n_recommendations]
        top_scores = user_predictions[top_indices]
        
        return top_indices, top_scores
    
    def compute_similarity_matrix(self, based_on: str = 'item') -> np.ndarray:
        """
        Compute item-item or user-user similarity matrix.
        
        Args:
            based_on: 'item' for item-based CF, 'user' for user-based CF
            
        Returns:
            Similarity matrix using cosine similarity
        """
        if based_on == 'item':
            # Item similarity: cosine between item factor vectors
            factors = self.item_factors
        else:
            # User similarity: cosine between user factor vectors
            factors = self.user_factors
        
        # Normalize factors
        norms = np.linalg.norm(factors, axis=1, keepdims=True)
        normalized = factors / (norms + 1e-9)
        
        # Compute cosine similarity
        similarity = normalized @ normalized.T
        
        return similarity
    
    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_factors': self.n_factors,
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'global_mean': self.global_mean,
                'predictions_matrix': self.predictions_matrix
            }, f)
    
    def load(self, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.n_factors = data['n_factors']
            self.user_factors = data['user_factors']
            self.item_factors = data['item_factors']
            self.global_mean = data['global_mean']
            self.predictions_matrix = data['predictions_matrix']

