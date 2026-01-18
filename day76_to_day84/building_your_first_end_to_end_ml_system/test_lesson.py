"""
Comprehensive tests for End-to-End ML Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from lesson_code import (
    DataValidator, FeatureTransformer, MLPipeline, create_sample_data
)
import os
from pathlib import Path


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return create_sample_data()


@pytest.fixture
def config():
    """Standard configuration for testing"""
    return {
        'required_columns': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived'],
        'feature_columns': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
        'target_column': 'Survived',
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 3,
        'model_params': {
            'n_estimators': 10,
            'max_depth': 5,
            'random_state': 42
        }
    }


class TestDataValidator:
    """Test data validation component"""
    
    def test_validator_accepts_valid_data(self, sample_data):
        """Valid data passes validation"""
        validator = DataValidator(['Pclass', 'Sex', 'Age', 'Fare'])
        validated = validator.validate(sample_data)
        assert len(validated) == len(sample_data)
    
    def test_validator_rejects_missing_columns(self, sample_data):
        """Missing required columns raises error"""
        validator = DataValidator(['Pclass', 'NonExistentColumn'])
        with pytest.raises(ValueError, match="Missing required columns"):
            validator.validate(sample_data)
    
    def test_validator_checks_numeric_types(self, sample_data):
        """Non-numeric age raises TypeError"""
        validator = DataValidator(['Age'])
        df_invalid = sample_data.copy()
        df_invalid['Age'] = 'invalid'
        with pytest.raises(TypeError, match="Age must be numeric"):
            validator.validate(df_invalid)
    
    def test_validator_checks_value_ranges(self, sample_data):
        """Out of range values raise ValueError"""
        validator = DataValidator(['Age'])
        df_invalid = sample_data.copy()
        df_invalid.loc[0, 'Age'] = 150  # Above max
        with pytest.raises(ValueError, match="above maximum"):
            validator.validate(df_invalid)
    
    def test_data_quality_report(self, sample_data):
        """Quality report contains expected metrics"""
        validator = DataValidator(['Pclass'])
        report = validator.get_data_quality_report(sample_data)
        
        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'missing_values' in report
        assert report['total_rows'] == len(sample_data)


class TestFeatureTransformer:
    """Test feature transformation component"""
    
    def test_transformer_fits_and_transforms(self, sample_data):
        """Transformer learns from data and applies transformations"""
        transformer = FeatureTransformer()
        df_features = sample_data[['Pclass', 'Age', 'Fare', 'Sex']]
        
        transformed = transformer.fit_transform(df_features)
        assert len(transformed) == len(df_features)
        assert not transformer.is_fitted == False
    
    def test_transformer_handles_missing_values(self, sample_data):
        """Missing values are imputed correctly"""
        transformer = FeatureTransformer()
        df_features = sample_data[['Age', 'Fare']].copy()
        
        # Introduce missing values
        df_features.loc[0:10, 'Age'] = np.nan
        
        transformer.fit(df_features)
        transformed = transformer.transform(df_features)
        
        assert not transformed.isnull().any().any()
    
    def test_transformer_scales_numerical_features(self, sample_data):
        """Numerical features are standardized"""
        transformer = FeatureTransformer()
        df_features = sample_data[['Age', 'Fare']]
        
        transformed = transformer.fit_transform(df_features)
        
        # Check if approximately normalized (mean ~ 0, std ~ 1)
        assert abs(transformed['Age'].mean()) < 0.1
        assert abs(transformed['Fare'].mean()) < 0.1
    
    def test_transformer_encodes_categorical_features(self, sample_data):
        """Categorical features are encoded as integers"""
        transformer = FeatureTransformer()
        df_features = sample_data[['Sex', 'Embarked']]
        
        transformed = transformer.fit_transform(df_features)
        
        assert pd.api.types.is_numeric_dtype(transformed['Sex'])
        assert pd.api.types.is_numeric_dtype(transformed['Embarked'])
    
    def test_transformer_requires_fit_before_transform(self, sample_data):
        """Transform without fit raises error"""
        transformer = FeatureTransformer()
        df_features = sample_data[['Age', 'Fare']]
        
        with pytest.raises(ValueError, match="must be fitted"):
            transformer.transform(df_features)


class TestMLPipeline:
    """Test complete ML pipeline"""
    
    def test_pipeline_loads_data(self, sample_data, config, tmp_path):
        """Pipeline loads and validates data"""
        pipeline = MLPipeline(config)
        
        # Save sample data
        data_file = tmp_path / "train.csv"
        sample_data.to_csv(data_file, index=False)
        
        df, quality_report = pipeline.load_data(str(data_file))
        assert len(df) == len(sample_data)
        assert 'total_rows' in quality_report
    
    def test_pipeline_trains_model(self, sample_data, config):
        """Pipeline trains model successfully"""
        pipeline = MLPipeline(config)
        metrics = pipeline.train(sample_data)
        
        assert pipeline.is_trained
        assert 'test_accuracy' in metrics
        assert 0 <= metrics['test_accuracy'] <= 1
    
    def test_pipeline_makes_predictions(self, sample_data, config):
        """Trained pipeline makes predictions"""
        pipeline = MLPipeline(config)
        pipeline.train(sample_data)
        
        predictions, probabilities = pipeline.predict(sample_data.head(10))
        
        assert len(predictions) == 10
        assert len(probabilities) == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_pipeline_predicts_single_sample(self, sample_data, config):
        """Pipeline handles single prediction"""
        pipeline = MLPipeline(config)
        pipeline.train(sample_data)
        
        test_passenger = {
            'Pclass': 3,
            'Sex': 'male',
            'Age': 25.0,
            'SibSp': 1,
            'Parch': 0,
            'Fare': 8.05,
            'Embarked': 'S'
        }
        
        result = pipeline.predict_single(test_passenger)
        
        assert 'survived' in result
        assert 'confidence' in result
        assert result['survived'] in [0, 1]
        assert 0 <= result['confidence'] <= 1
    
    def test_pipeline_requires_training_before_prediction(self, config):
        """Prediction without training raises error"""
        pipeline = MLPipeline(config)
        df = create_sample_data()
        
        with pytest.raises(ValueError, match="must be trained"):
            pipeline.predict(df)
    
    def test_pipeline_saves_and_loads_model(self, sample_data, config, tmp_path):
        """Pipeline saves and loads correctly"""
        # Train and save
        pipeline = MLPipeline(config)
        pipeline.train(sample_data)
        
        model_file = tmp_path / "model.pkl"
        pipeline.save_model(str(model_file))
        
        assert model_file.exists()
        
        # Load and verify
        loaded_pipeline = MLPipeline.load_model(str(model_file))
        assert loaded_pipeline.is_trained
        assert loaded_pipeline.metrics == pipeline.metrics
    
    def test_pipeline_cross_validation_returns_multiple_scores(self, sample_data, config):
        """Cross-validation produces expected number of scores"""
        pipeline = MLPipeline(config)
        metrics = pipeline.train(sample_data)
        
        cv_scores = metrics['cv_scores']
        assert len(cv_scores) == config['cv_folds']
        assert all(0 <= score <= 1 for score in cv_scores)
    
    def test_pipeline_calculates_all_metrics(self, sample_data, config):
        """Pipeline calculates comprehensive metrics"""
        pipeline = MLPipeline(config)
        metrics = pipeline.train(sample_data)
        
        expected_metrics = [
            'test_accuracy', 'test_precision', 'test_recall', 'test_f1',
            'cv_mean', 'cv_std'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_end_to_end_pipeline_execution(self, sample_data, config, tmp_path):
        """Complete pipeline from data to prediction"""
        # Save data
        data_file = tmp_path / "train.csv"
        sample_data.to_csv(data_file, index=False)
        
        # Initialize and train
        pipeline = MLPipeline(config)
        df, _ = pipeline.load_data(str(data_file))
        metrics = pipeline.train(df)
        
        # Save model
        model_file = tmp_path / "model.pkl"
        pipeline.save_model(str(model_file))
        
        # Load and predict
        loaded_pipeline = MLPipeline.load_model(str(model_file))
        test_data = {
            'Pclass': 1,
            'Sex': 'female',
            'Age': 30.0,
            'SibSp': 0,
            'Parch': 0,
            'Fare': 50.0,
            'Embarked': 'C'
        }
        
        result = loaded_pipeline.predict_single(test_data)
        
        assert 'survived' in result
        assert 'confidence' in result
        assert result['confidence'] > 0.5  # Should have reasonable confidence
    
    def test_pipeline_handles_edge_cases(self, config):
        """Pipeline handles edge cases gracefully"""
        pipeline = MLPipeline(config)
        
        # Create edge case data
        edge_data = pd.DataFrame({
            'PassengerId': [1, 2, 3],
            'Survived': [0, 1, 0],
            'Pclass': [1, 3, 2],
            'Sex': ['male', 'female', 'male'],
            'Age': [np.nan, 80.0, 0.42],  # Missing, old, infant
            'SibSp': [0, 8, 0],  # Extreme family size
            'Parch': [0, 5, 0],
            'Fare': [0.0, 512.0, 7.25],  # Min, max, normal
            'Embarked': ['S', np.nan, 'Q']  # Missing
        })
        
        metrics = pipeline.train(edge_data)
        
        # Should complete without errors
        assert pipeline.is_trained
        assert 'test_accuracy' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
