"""
Test suite for Customer Segmentation System
Tests cover data processing, model training, prediction, and edge cases
"""

import pytest
import numpy as np
import pandas as pd
from lesson_code import CustomerSegmentationEngine, generate_sample_data


class TestDataGeneration:
    """Test synthetic data generation."""
    
    def test_generate_sample_data_shape(self):
        """Test data generation produces correct shape."""
        df = generate_sample_data(n_customers=100)
        assert len(df) == 100
        assert set(df.columns) == {'total_spend', 'purchase_frequency', 'days_since_last_purchase'}
    
    def test_generate_sample_data_non_negative(self):
        """Test all generated values are non-negative."""
        df = generate_sample_data(n_customers=100)
        assert (df >= 0).all().all()
    
    def test_generate_sample_data_no_nulls(self):
        """Test no null values in generated data."""
        df = generate_sample_data(n_customers=100)
        assert not df.isnull().any().any()


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def test_engineer_features_creates_new_columns(self):
        """Test feature engineering creates expected columns."""
        df = generate_sample_data(n_customers=50)
        engine = CustomerSegmentationEngine()
        df_engineered = engine.engineer_features(df)
        
        expected_features = {'value_score', 'engagement_level', 'ltv_proxy'}
        assert expected_features.issubset(set(df_engineered.columns))
    
    def test_engineer_features_handles_zero_division(self):
        """Test feature engineering handles zero values properly."""
        df = pd.DataFrame({
            'total_spend': [100, 0, 200],
            'purchase_frequency': [0, 5, 10],
            'days_since_last_purchase': [10, 0, 20]
        })
        
        engine = CustomerSegmentationEngine()
        df_engineered = engine.engineer_features(df)
        
        assert not df_engineered.isnull().any().any()
        assert not np.isinf(df_engineered).any().any()
    
    def test_engineer_features_preserves_original_columns(self):
        """Test original columns are preserved after engineering."""
        df = generate_sample_data(n_customers=50)
        original_cols = set(df.columns)
        
        engine = CustomerSegmentationEngine()
        df_engineered = engine.engineer_features(df)
        
        assert original_cols.issubset(set(df_engineered.columns))


class TestSegmentationEngine:
    """Test core segmentation engine functionality."""
    
    def test_engine_initialization(self):
        """Test engine initializes with correct parameters."""
        engine = CustomerSegmentationEngine(min_clusters=3, max_clusters=7)
        assert engine.min_clusters == 3
        assert engine.max_clusters == 7
        assert engine.model is None
        assert engine.optimal_k is None
    
    def test_fit_finds_optimal_clusters(self):
        """Test fit method finds optimal number of clusters."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine(min_clusters=2, max_clusters=6)
        engine.fit(df)
        
        assert engine.optimal_k is not None
        assert 2 <= engine.optimal_k <= 6
        assert engine.model is not None
    
    def test_fit_with_fixed_clusters(self):
        """Test fit method with predefined cluster count."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df, n_clusters=4)
        
        assert engine.optimal_k == 4
        assert engine.model.n_clusters == 4
    
    def test_fit_creates_segment_profiles(self):
        """Test segment profiles are created after fitting."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        assert engine.segment_profiles is not None
        assert len(engine.segment_profiles) == engine.optimal_k
    
    def test_fit_handles_missing_values(self):
        """Test fitting handles missing values properly."""
        df = generate_sample_data(n_customers=100)
        # Introduce missing values
        df.loc[0:10, 'total_spend'] = np.nan
        
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        assert engine.model is not None
        assert engine.optimal_k is not None


class TestPrediction:
    """Test prediction functionality."""
    
    def test_predict_returns_correct_shapes(self):
        """Test prediction returns arrays of correct shape."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        new_data = generate_sample_data(n_customers=10)
        labels, confidence = engine.predict(new_data)
        
        assert len(labels) == 10
        assert len(confidence) == 10
    
    def test_predict_labels_in_valid_range(self):
        """Test predicted labels are within valid cluster range."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        new_data = generate_sample_data(n_customers=10)
        labels, _ = engine.predict(new_data)
        
        assert all(0 <= label < engine.optimal_k for label in labels)
    
    def test_predict_confidence_in_valid_range(self):
        """Test confidence scores are between 0 and 1."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        new_data = generate_sample_data(n_customers=10)
        _, confidence = engine.predict(new_data)
        
        assert all(0 <= conf <= 1 for conf in confidence)
    
    def test_predict_without_fit_raises_error(self):
        """Test prediction without training raises appropriate error."""
        engine = CustomerSegmentationEngine()
        new_data = generate_sample_data(n_customers=10)
        
        with pytest.raises(ValueError, match="Model not trained"):
            engine.predict(new_data)
    
    def test_predict_handles_missing_features(self):
        """Test prediction handles missing features gracefully."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        # Create data with missing engineered features
        new_data = pd.DataFrame({
            'total_spend': [1000, 2000],
            'purchase_frequency': [10, 20],
            'days_since_last_purchase': [30, 15]
        })
        
        labels, confidence = engine.predict(new_data)
        assert len(labels) == 2


class TestSegmentCharacteristics:
    """Test segment characteristics retrieval."""
    
    def test_get_segment_characteristics_returns_dict(self):
        """Test segment characteristics returns proper dictionary."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        characteristics = engine.get_segment_characteristics(0)
        
        assert isinstance(characteristics, dict)
        assert 'segment_id' in characteristics
        assert 'size' in characteristics
        assert 'features' in characteristics
    
    def test_get_segment_characteristics_all_segments(self):
        """Test characteristics can be retrieved for all segments."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        for segment_id in range(engine.optimal_k):
            characteristics = engine.get_segment_characteristics(segment_id)
            assert characteristics['segment_id'] == segment_id
            assert characteristics['size'] > 0
    
    def test_get_segment_characteristics_invalid_segment(self):
        """Test invalid segment ID raises appropriate error."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        with pytest.raises(ValueError, match="Segment .* not found"):
            engine.get_segment_characteristics(999)


class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_save_and_load_model(self, tmp_path):
        """Test model can be saved and loaded correctly."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        engine.save_model(str(model_path))
        
        # Load model
        loaded_engine = CustomerSegmentationEngine.load_model(str(model_path))
        
        assert loaded_engine.optimal_k == engine.optimal_k
        assert loaded_engine.feature_names == engine.feature_names
    
    def test_loaded_model_makes_predictions(self, tmp_path):
        """Test loaded model can make predictions."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        # Save and load
        model_path = tmp_path / "test_model.pkl"
        engine.save_model(str(model_path))
        loaded_engine = CustomerSegmentationEngine.load_model(str(model_path))
        
        # Test predictions
        new_data = generate_sample_data(n_customers=10)
        labels, confidence = loaded_engine.predict(new_data)
        
        assert len(labels) == 10
        assert len(confidence) == 10


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_fit_with_very_few_samples(self):
        """Test fitting with minimal samples."""
        df = generate_sample_data(n_customers=10)
        engine = CustomerSegmentationEngine(min_clusters=2, max_clusters=3)
        engine.fit(df)
        
        assert engine.model is not None
    
    def test_fit_with_identical_samples(self):
        """Test fitting when all samples are identical."""
        df = pd.DataFrame({
            'total_spend': [1000] * 50,
            'purchase_frequency': [10] * 50,
            'days_since_last_purchase': [30] * 50
        })
        
        engine = CustomerSegmentationEngine(min_clusters=2, max_clusters=4)
        engine.fit(df)
        
        # Should still create model even with identical data
        assert engine.model is not None
    
    def test_predict_single_sample(self):
        """Test prediction on single sample."""
        df = generate_sample_data(n_customers=200)
        engine = CustomerSegmentationEngine()
        engine.fit(df)
        
        single_sample = generate_sample_data(n_customers=1)
        labels, confidence = engine.predict(single_sample)
        
        assert len(labels) == 1
        assert len(confidence) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
