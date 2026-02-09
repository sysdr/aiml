"""
Recommendation system evaluation framework.

Implements offline metrics used in production systems:
RMSE, Precision@K, Recall@K, Coverage, Diversity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter


class RecommenderEvaluator:
    """
    Comprehensive evaluation framework for recommendation systems.
    
    Netflix measures both offline metrics (RMSE, precision) and
    online metrics (CTR, watch time, retention) via A/B testing.
    This class implements offline evaluation strategies.
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def rmse(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """
        Root Mean Square Error - measures prediction accuracy.
        
        Lower is better. Netflix Prize goal: reduce RMSE by 10%
        from baseline (RMSE=0.9525 → 0.8572).
        
        RMSE = sqrt(mean((predicted - actual)²))
        """
        return np.sqrt(np.mean((predictions - actuals) ** 2))
    
    def mae(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """
        Mean Absolute Error - more interpretable than RMSE.
        
        MAE = mean(|predicted - actual|)
        """
        return np.mean(np.abs(predictions - actuals))
    
    def precision_at_k(
        self,
        recommended_items: np.ndarray,
        relevant_items: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Precision@K: fraction of recommended items that are relevant.
        
        Precision@K = |recommended ∩ relevant| / K
        
        Measures recommendation accuracy at rank K.
        High precision means most recommendations are good.
        """
        top_k = recommended_items[:k]
        n_relevant = len(set(top_k) & set(relevant_items))
        return n_relevant / k
    
    def recall_at_k(
        self,
        recommended_items: np.ndarray,
        relevant_items: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Recall@K: fraction of relevant items that were recommended.
        
        Recall@K = |recommended ∩ relevant| / |relevant|
        
        Measures catalog coverage for each user.
        High recall means we found most of what user likes.
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        n_relevant = len(set(top_k) & set(relevant_items))
        return n_relevant / len(relevant_items)
    
    def f1_score_at_k(
        self,
        recommended_items: np.ndarray,
        relevant_items: np.ndarray,
        k: int = 10
    ) -> float:
        """
        F1@K: harmonic mean of precision and recall.
        
        F1 = 2 × (precision × recall) / (precision + recall)
        
        Balances both metrics.
        """
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(
        self,
        recommended_items: np.ndarray,
        relevant_items: np.ndarray,
        relevance_scores: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Normalized Discounted Cumulative Gain@K.
        
        NDCG considers both relevance AND ranking position.
        Higher-ranked relevant items contribute more to the score.
        
        DCG = Σ(relevance[i] / log2(i + 1))
        NDCG = DCG / ideal_DCG
        
        Used by YouTube to evaluate ranking quality.
        """
        top_k = recommended_items[:k]
        
        # Compute DCG for recommendations
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in relevant_items:
                idx = np.where(relevant_items == item)[0][0]
                relevance = relevance_scores[idx]
                dcg += relevance / np.log2(i + 2)  # i+2 to avoid log(1)=0
        
        # Compute ideal DCG (best possible ordering)
        sorted_relevance = np.sort(relevance_scores)[::-1][:k]
        idcg = np.sum(sorted_relevance / np.log2(np.arange(2, k + 2)))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def catalog_coverage(
        self,
        all_recommendations: List[np.ndarray],
        n_items: int
    ) -> float:
        """
        Catalog coverage: fraction of items ever recommended.
        
        Coverage = |unique recommended items| / |total items|
        
        Low coverage indicates filter bubble - system only
        recommends popular items. Netflix aims for high coverage
        to surface long-tail content.
        """
        unique_items = set()
        for recs in all_recommendations:
            unique_items.update(recs)
        
        return len(unique_items) / n_items
    
    def diversity_score(
        self,
        recommended_items: np.ndarray,
        item_similarity_matrix: np.ndarray
    ) -> float:
        """
        Average pairwise dissimilarity in recommendations.
        
        Diversity = mean(1 - similarity(i, j)) for all pairs i,j
        
        Higher diversity prevents recommendations from being
        too similar to each other. YouTube injects diversity
        to avoid showing only slightly different videos.
        """
        if len(recommended_items) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(recommended_items)):
            for j in range(i + 1, len(recommended_items)):
                item_i = recommended_items[i]
                item_j = recommended_items[j]
                sim = item_similarity_matrix[item_i, item_j]
                similarities.append(sim)
        
        return 1 - np.mean(similarities)

