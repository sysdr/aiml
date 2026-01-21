"""
Test suite for Day 88: Cluster Evaluation
Validates all three evaluation methods work correctly
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from lesson_code import ClusterEvaluator, generate_sample_data


class TestClusterEvaluator:
    @pytest.fixture
    def sample_data(self):
        X, y = make_blobs(n_samples=300, n_features=4, centers=3, 
                         cluster_std=1.0, random_state=42)
        return X, y
    
    @pytest.fixture
    def evaluator(self):
        return ClusterEvaluator(k_range=(2, 6), random_state=42)
    
    def test_initialization(self, evaluator):
        assert evaluator.k_range == range(2, 7)
        assert evaluator.random_state == 42
        assert 'elbow' in evaluator.results
        assert 'silhouette' in evaluator.results
        assert 'gap' in evaluator.results
    
    def test_elbow_method_decreasing_wcss(self, evaluator, sample_data):
        X, _ = sample_data
        evaluator._compute_elbow_method(X)
        wcss_values = evaluator.results['elbow']['wcss']
        for i in range(len(wcss_values) - 1):
            assert wcss_values[i] > wcss_values[i + 1], "WCSS should decrease monotonically"
    
    def test_elbow_finds_optimal_k(self, evaluator, sample_data):
        X, _ = sample_data
        evaluator._compute_elbow_method(X)
        optimal_k = evaluator.results['elbow']['optimal_k']
        assert optimal_k in evaluator.k_range, "Optimal k should be in evaluated range"
    
    def test_silhouette_scores_range(self, evaluator, sample_data):
        X, _ = sample_data
        evaluator._compute_silhouette_scores(X)
        avg_scores = evaluator.results['silhouette']['avg_scores']
        for score in avg_scores:
            assert -1 <= score <= 1, f"Silhouette score {score} outside valid range [-1, 1]"
    
    def test_silhouette_best_near_true_k(self, sample_data):
        X, y = sample_data
        true_k = len(np.unique(y))
        evaluator = ClusterEvaluator(k_range=(2, 6), random_state=42)
        evaluator._compute_silhouette_scores(X)
        optimal_k = evaluator.results['silhouette']['optimal_k']
        assert abs(optimal_k - true_k) <= 1, f"Optimal k={optimal_k} should be close to true k={true_k}"
    
    def test_gap_statistic_positive(self, evaluator, sample_data):
        X, _ = sample_data
        evaluator._compute_gap_statistic(X, n_refs=10)
        gaps = evaluator.results['gap']['gaps']
        assert all(gap > 0 for gap in gaps), "Gap statistics should be positive for structured data"
    
    def test_gap_returns_valid_k(self, evaluator, sample_data):
        X, _ = sample_data
        evaluator._compute_gap_statistic(X, n_refs=10)
        optimal_k = evaluator.results['gap']['optimal_k']
        assert optimal_k in evaluator.k_range, "Optimal k from Gap statistic should be in evaluated range"
    
    def test_full_evaluation_pipeline(self, evaluator, sample_data):
        X, _ = sample_data
        evaluator.fit(X)
        assert len(evaluator.results['elbow']['wcss']) > 0
        assert len(evaluator.results['silhouette']['avg_scores']) > 0
        assert len(evaluator.results['gap']['gaps']) > 0
    
    def test_get_recommendations(self, evaluator, sample_data):
        X, _ = sample_data
        evaluator.fit(X)
        recommendations = evaluator.get_recommendations()
        assert 'elbow_method' in recommendations
        assert 'silhouette_analysis' in recommendations
        assert 'gap_statistic' in recommendations
        assert 'consensus' in recommendations
        assert 'agreement' in recommendations
        for method in ['elbow_method', 'silhouette_analysis', 'gap_statistic', 'consensus']:
            k = recommendations[method]
            assert k in evaluator.k_range, f"{method} returned k={k} outside valid range"
    
    def test_consensus_logic(self, evaluator, sample_data):
        X, _ = sample_data
        evaluator.fit(X)
        recommendations = evaluator.get_recommendations()
        k_values = [
            recommendations['elbow_method'],
            recommendations['silhouette_analysis'],
            recommendations['gap_statistic']
        ]
        most_common = max(set(k_values), key=k_values.count)
        assert recommendations['consensus'] == most_common


class TestDataGeneration:
    def test_generate_sample_data_shape(self):
        X, y = generate_sample_data(n_samples=100, n_features=3, n_clusters=4)
        assert X.shape == (100, 3), "Data shape should match parameters"
        assert len(y) == 100, "Labels should match sample count"
        assert len(np.unique(y)) == 4, "Should have correct number of clusters"
    
    def test_generate_sample_data_is_dataframe(self):
        X, y = generate_sample_data()
        assert isinstance(X, pd.DataFrame), "Should return DataFrame"
        assert all(isinstance(col, str) for col in X.columns), "Columns should have string names"


class TestEdgeCases:
    def test_small_k_range(self):
        evaluator = ClusterEvaluator(k_range=(2, 3))
        X, _ = generate_sample_data(n_samples=100, n_features=2)
        evaluator.fit(X)
        recommendations = evaluator.get_recommendations()
        assert recommendations['consensus'] in [2, 3]
    
    def test_single_cluster_not_in_range(self):
        evaluator = ClusterEvaluator(k_range=(2, 5))
        assert 1 not in evaluator.k_range, "k=1 should not be in range"
    
    def test_high_dimensional_data(self):
        X = np.random.randn(200, 20)
        evaluator = ClusterEvaluator(k_range=(2, 4))
        evaluator.fit(X)
        recommendations = evaluator.get_recommendations()
        assert recommendations['consensus'] in [2, 3, 4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
