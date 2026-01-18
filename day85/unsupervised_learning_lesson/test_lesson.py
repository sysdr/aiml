"""
Test Suite for Day 85: Introduction to Unsupervised Learning
Validates all components of the customer segmentation pipeline
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import os
import sys

# Import the lesson code
from lesson_code import CustomerSegmentationPipeline


class TestCustomerSegmentation:
    """Test suite for customer segmentation pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a fresh pipeline instance for each test."""
        return CustomerSegmentationPipeline(n_components=2, random_state=42)
    
    @pytest.fixture
    def pipeline_with_data(self, pipeline):
        """Pipeline with generated data."""
        pipeline.generate_sample_data(n_samples=500)
        return pipeline
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes with correct parameters."""
        assert pipeline.n_components == 2
        assert pipeline.random_state == 42
        assert pipeline.scaler is not None
        assert pipeline.data is None
    
    def test_data_generation(self, pipeline):
        """Test synthetic data generation creates valid dataset."""
        data = pipeline.generate_sample_data(n_samples=1000)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000
        assert 'customer_id' in data.columns
        assert 'recency_days' in data.columns
        assert 'purchase_frequency' in data.columns
        assert 'total_monetary' in data.columns
        assert 'engagement_score' in data.columns
        
        # Check data ranges
        assert data['recency_days'].min() >= 1
        assert data['recency_days'].max() <= 365
        assert data['purchase_frequency'].min() >= 1
        assert data['total_monetary'].min() > 0
    
    def test_no_missing_values(self, pipeline_with_data):
        """Ensure generated data has no missing values."""
        assert pipeline_with_data.data.isnull().sum().sum() == 0
    
    def test_feature_preprocessing(self, pipeline_with_data):
        """Test feature scaling produces standardized data."""
        scaled = pipeline_with_data.preprocess_features()
        
        assert scaled is not None
        assert scaled.shape[0] == 500
        assert scaled.shape[1] == 6  # 6 features
        
        # Check standardization (mean ≈ 0, std ≈ 1)
        assert np.abs(scaled.mean(axis=0)).max() < 1e-10
        assert np.abs(scaled.std(axis=0) - 1.0).max() < 1e-10
    
    def test_dimensionality_reduction(self, pipeline_with_data):
        """Test PCA reduces dimensions correctly."""
        pipeline_with_data.preprocess_features()
        pca_features = pipeline_with_data.dimensionality_reduction(save_plots=False)
        
        assert pca_features is not None
        assert pca_features.shape == (500, 2)
        assert pipeline_with_data.pca is not None
        
        # Check explained variance
        explained_var = pipeline_with_data.pca.explained_variance_ratio_
        assert len(explained_var) == 2
        assert 0 < explained_var.sum() <= 1.0
    
    def test_clustering_execution(self, pipeline_with_data):
        """Test K-Means clustering produces valid results."""
        pipeline_with_data.preprocess_features()
        pipeline_with_data.dimensionality_reduction(save_plots=False)
        
        labels = pipeline_with_data.perform_clustering(n_clusters=3, save_plots=False)
        
        assert labels is not None
        assert len(labels) == 500
        assert len(np.unique(labels)) == 3
        assert labels.min() >= 0
        assert labels.max() <= 2
    
    def test_optimal_k_discovery(self, pipeline_with_data):
        """Test optimal K detection identifies reasonable cluster count."""
        pipeline_with_data.preprocess_features()
        pipeline_with_data.dimensionality_reduction(save_plots=False)
        
        optimal_k = pipeline_with_data.find_optimal_clusters(max_k=6, save_plots=False)
        
        assert optimal_k is not None
        assert 2 <= optimal_k <= 6
        assert isinstance(optimal_k, (int, np.integer))
    
    def test_silhouette_score_quality(self, pipeline_with_data):
        """Test clustering quality using silhouette score."""
        pipeline_with_data.preprocess_features()
        pipeline_with_data.dimensionality_reduction(save_plots=False)
        pipeline_with_data.perform_clustering(n_clusters=3, save_plots=False)
        
        score = silhouette_score(pipeline_with_data.pca_features, 
                                pipeline_with_data.cluster_labels)
        
        # Silhouette score should be positive for reasonable clustering
        assert score > 0.2
    
    def test_segment_analysis(self, pipeline_with_data):
        """Test segment profiling produces interpretable results."""
        pipeline_with_data.preprocess_features()
        pipeline_with_data.dimensionality_reduction(save_plots=False)
        pipeline_with_data.perform_clustering(n_clusters=3, save_plots=False)
        
        profiles = pipeline_with_data.analyze_segments(save_plots=False)
        
        assert profiles is not None
        assert isinstance(profiles, pd.DataFrame)
        assert len(profiles) == 3  # 3 clusters
    
    def test_cluster_label_assignment(self, pipeline_with_data):
        """Test cluster labels are properly assigned to original data."""
        pipeline_with_data.preprocess_features()
        pipeline_with_data.dimensionality_reduction(save_plots=False)
        pipeline_with_data.perform_clustering(n_clusters=3, save_plots=False)
        
        assert 'cluster' in pipeline_with_data.data.columns
        assert pipeline_with_data.data['cluster'].notna().all()
    
    def test_export_functionality(self, pipeline_with_data, tmp_path):
        """Test results export creates valid CSV file."""
        pipeline_with_data.preprocess_features()
        pipeline_with_data.dimensionality_reduction(save_plots=False)
        pipeline_with_data.perform_clustering(n_clusters=3, save_plots=False)
        
        output_file = tmp_path / "test_segments.csv"
        pipeline_with_data.export_results(filename=str(output_file))
        
        assert output_file.exists()
        
        # Verify exported data
        exported = pd.read_csv(output_file)
        assert len(exported) == 500
        assert 'customer_id' in exported.columns
        assert 'cluster' in exported.columns
    
    def test_reproducibility(self):
        """Test pipeline produces consistent results with same random seed."""
        pipeline1 = CustomerSegmentationPipeline(random_state=42)
        pipeline1.generate_sample_data(n_samples=300)
        pipeline1.preprocess_features()
        pipeline1.dimensionality_reduction(save_plots=False)
        pipeline1.perform_clustering(n_clusters=3, save_plots=False)
        
        pipeline2 = CustomerSegmentationPipeline(random_state=42)
        pipeline2.generate_sample_data(n_samples=300)
        pipeline2.preprocess_features()
        pipeline2.dimensionality_reduction(save_plots=False)
        pipeline2.perform_clustering(n_clusters=3, save_plots=False)
        
        # Cluster assignments should be identical
        assert np.array_equal(pipeline1.cluster_labels, pipeline2.cluster_labels)
    
    def test_different_cluster_counts(self, pipeline_with_data):
        """Test pipeline handles various cluster counts correctly."""
        pipeline_with_data.preprocess_features()
        pipeline_with_data.dimensionality_reduction(save_plots=False)
        
        for k in [2, 4, 5]:
            labels = pipeline_with_data.perform_clustering(n_clusters=k, save_plots=False)
            assert len(np.unique(labels)) == k
    
    def test_visualization_directory_creation(self, pipeline_with_data):
        """Test visualization directory is created properly."""
        pipeline_with_data.exploratory_analysis(save_plots=True)
        
        assert os.path.exists('visualizations')
        assert os.path.isdir('visualizations')


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_dataset(self):
        """Test pipeline handles small datasets."""
        pipeline = CustomerSegmentationPipeline()
        pipeline.generate_sample_data(n_samples=50)
        pipeline.preprocess_features()
        pipeline.dimensionality_reduction(save_plots=False)
        
        # Should still work with small data
        labels = pipeline.perform_clustering(n_clusters=2, save_plots=False)
        assert len(labels) == 50
    
    def test_large_dataset(self):
        """Test pipeline scales to larger datasets."""
        pipeline = CustomerSegmentationPipeline()
        pipeline.generate_sample_data(n_samples=5000)
        pipeline.preprocess_features()
        
        # Should handle 5000 samples efficiently
        assert pipeline.scaled_features.shape[0] == 5000


def test_imports():
    """Test all required packages are importable."""
    try:
        import numpy
        import pandas
        import sklearn
        import matplotlib
        import seaborn
        import scipy
    except ImportError as e:
        pytest.fail(f"Required package import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

