"""
Day 42: Data Splitting - Test Suite
Verifies all splitting strategies work correctly
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from lesson_code import DataSplitter, ProductionMLPipeline


class TestDataSplitter:
    """Test suite for DataSplitter class"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def imbalanced_data(self):
        """Generate imbalanced data for testing"""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=2,
            weights=[0.9, 0.1],
            random_state=42
        )
        return X, y
    
    def test_basic_split_sizes(self, sample_data):
        """Test that basic split creates correct sizes"""
        X, y = sample_data
        splitter = DataSplitter()
        
        X_train, X_val, X_test, y_train, y_val, y_test = \
            splitter.basic_split(X, y, test_size=0.2, val_size=0.1)
        
        # Check sizes
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)
        
        # Check proportions (roughly 70-15-15)
        total = len(X)
        assert 0.65 < len(X_train) / total < 0.75
        assert 0.10 < len(X_val) / total < 0.20
        assert 0.15 < len(X_test) / total < 0.25
    
    def test_no_data_overlap(self, sample_data):
        """Verify train/val/test sets don't overlap"""
        X, y = sample_data
        splitter = DataSplitter()
        
        X_train, X_val, X_test, _, _, _ = splitter.basic_split(X, y)
        
        # Check no overlap (using first column as identifier)
        train_ids = set(X_train[:, 0])
        val_ids = set(X_val[:, 0])
        test_ids = set(X_test[:, 0])
        
        assert len(train_ids & val_ids) == 0, "Train and validation sets overlap"
        assert len(train_ids & test_ids) == 0, "Train and test sets overlap"
        assert len(val_ids & test_ids) == 0, "Validation and test sets overlap"
    
    def test_stratified_split_preserves_distribution(self, imbalanced_data):
        """Test that stratified split preserves class distribution"""
        X, y = imbalanced_data
        splitter = DataSplitter()
        
        # Get original distribution
        original_dist = np.bincount(y) / len(y)
        
        X_train, X_val, X_test, y_train, y_val, y_test = \
            splitter.stratified_split(X, y)
        
        # Check each split preserves distribution (within 5%)
        train_dist = np.bincount(y_train) / len(y_train)
        val_dist = np.bincount(y_val) / len(y_val)
        test_dist = np.bincount(y_test) / len(y_test)
        
        for i in range(len(original_dist)):
            assert abs(train_dist[i] - original_dist[i]) < 0.05
            assert abs(val_dist[i] - original_dist[i]) < 0.05
            assert abs(test_dist[i] - original_dist[i]) < 0.05
    
    def test_time_series_split_ordering(self):
        """Test that time series split maintains temporal ordering"""
        X = np.arange(100).reshape(-1, 1)
        y = np.random.randint(0, 2, 100)
        
        splitter = DataSplitter()
        splits = splitter.time_series_split(X, y, n_splits=5)
        
        # Verify each split: train indices < test indices
        for train_idx, test_idx in splits:
            assert train_idx[-1] < test_idx[0], \
                "Train set contains future data relative to test set"
    
    def test_k_fold_coverage(self, sample_data):
        """Test that k-fold covers all data exactly once"""
        X, y = sample_data
        splitter = DataSplitter()
        
        splits = splitter.k_fold_cross_validation(X, y, n_splits=5)
        
        # Each sample should appear in exactly one test fold
        all_test_indices = []
        for train_idx, test_idx in splits:
            all_test_indices.extend(test_idx)
        
        assert len(all_test_indices) == len(X)
        assert len(set(all_test_indices)) == len(X)
    
    def test_reproducibility(self, sample_data):
        """Test that splits are reproducible with same random_state"""
        X, y = sample_data
        
        splitter1 = DataSplitter(random_state=42)
        splitter2 = DataSplitter(random_state=42)
        
        X_train1, _, _, _, _, _ = splitter1.basic_split(X, y)
        X_train2, _, _, _, _, _ = splitter2.basic_split(X, y)
        
        np.testing.assert_array_equal(X_train1, X_train2)


class TestProductionMLPipeline:
    """Test suite for ProductionMLPipeline"""
    
    def test_data_leakage_detection(self):
        """Verify that data leakage demonstration runs"""
        pipeline = ProductionMLPipeline()
        acc_wrong, acc_correct = pipeline.demonstrate_data_leakage()
        
        # Both should be reasonable accuracies
        assert 0.5 < acc_wrong < 1.0
        assert 0.5 < acc_correct < 1.0
    
    def test_full_pipeline(self):
        """Test complete training pipeline"""
        pipeline = ProductionMLPipeline()
        model, test_score = pipeline.train_with_proper_splits()
        
        # Model should achieve reasonable performance
        assert test_score > 0.7, "Model performance too low"
        assert model is not None


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_small_dataset(self):
        """Test splitting very small datasets"""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        splitter = DataSplitter()
        X_train, X_val, X_test, _, _, _ = splitter.basic_split(X, y)
        
        # Should still create all three sets
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
    
    def test_binary_classification(self):
        """Test with binary classification data"""
        X, y = make_classification(
            n_samples=500,
            n_classes=2,
            random_state=42
        )
        
        splitter = DataSplitter()
        _, _, _, y_train, y_val, y_test = splitter.stratified_split(X, y)
        
        # Both classes should appear in all splits
        assert len(np.unique(y_train)) == 2
        assert len(np.unique(y_val)) == 2
        assert len(np.unique(y_test)) == 2


def test_integration():
    """Integration test - run complete workflow"""
    # Generate data
    X, y = make_classification(n_samples=1000, random_state=42)
    
    # Split data
    splitter = DataSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = \
        splitter.stratified_split(X, y)
    
    # Train simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import accuracy_score
    val_score = accuracy_score(y_val, model.predict(X_val))
    test_score = accuracy_score(y_test, model.predict(X_test))
    
    # Reasonable performance
    assert val_score > 0.6
    assert test_score > 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
