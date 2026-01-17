"""
Day 75: Model Persistence - Test Suite

Comprehensive tests for production model persistence patterns.
"""

import pytest
import joblib
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import tempfile
import shutil
import time

from lesson_code import (
    ModelPersistence,
    ModelVersionManager,
    ModelServer,
    train_fraud_detection_models
)


@pytest.fixture
def temp_models_dir():
    """Create temporary directory for test models"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_model():
    """Create simple trained model for testing"""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = LogisticRegression()
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def persistence_manager(temp_models_dir):
    """Create ModelPersistence instance"""
    return ModelPersistence(models_dir=temp_models_dir)


class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_save_model_creates_file(self, persistence_manager, sample_model):
        """Test that saving creates pkl file"""
        model, X, y = sample_model
        
        metadata = {
            'version': 'v1.0.0',
            'accuracy': 0.95,
            'n_features': X.shape[1]
        }
        
        path = persistence_manager.save_model(
            model=model,
            model_name='test_model',
            metadata=metadata
        )
        
        assert Path(path).exists()
        assert path.endswith('.pkl')
    
    def test_save_creates_metadata_file(self, persistence_manager, sample_model):
        """Test that separate metadata JSON is created"""
        model, X, y = sample_model
        
        metadata = {'version': 'v1.0.0'}
        
        persistence_manager.save_model(
            model=model,
            model_name='test_model',
            metadata=metadata
        )
        
        metadata_path = Path(persistence_manager.models_dir) / 'test_model_metadata.json'
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            loaded_metadata = json.load(f)
        
        assert 'version' in loaded_metadata
        assert loaded_metadata['version'] == 'v1.0.0'
    
    def test_load_model_returns_correct_types(self, persistence_manager, sample_model):
        """Test that loading returns model and metadata"""
        model, X, y = sample_model
        
        metadata = {'version': 'v1.0.0', 'n_features': X.shape[1]}
        
        persistence_manager.save_model(
            model=model,
            model_name='test_model',
            metadata=metadata
        )
        
        loaded_model, loaded_metadata = persistence_manager.load_model('test_model')
        
        assert hasattr(loaded_model, 'predict')
        assert isinstance(loaded_metadata, dict)
        assert loaded_metadata['version'] == 'v1.0.0'
    
    def test_loaded_model_predictions_match(self, persistence_manager, sample_model):
        """Test that loaded model produces same predictions"""
        model, X, y = sample_model
        
        # Original predictions
        original_pred = model.predict(X[:5])
        
        # Save and load
        metadata = {'version': 'v1.0.0', 'n_features': X.shape[1]}
        persistence_manager.save_model(
            model=model,
            model_name='test_model',
            metadata=metadata
        )
        
        loaded_model, _ = persistence_manager.load_model('test_model')
        loaded_pred = loaded_model.predict(X[:5])
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_compression_reduces_file_size(self, persistence_manager, temp_models_dir):
        """Test that compression reduces file size"""
        # Create larger model for compression test
        X, y = make_classification(n_samples=1000, n_features=50, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save without compression
        path_no_compress = persistence_manager.save_model(
            model=model,
            model_name='no_compress',
            metadata={'version': 'v1'},
            compress=0
        )
        
        # Save with compression
        path_compress = persistence_manager.save_model(
            model=model,
            model_name='with_compress',
            metadata={'version': 'v1'},
            compress=9
        )
        
        size_no_compress = Path(path_no_compress).stat().st_size
        size_compress = Path(path_compress).stat().st_size
        
        # Compressed should be smaller
        assert size_compress < size_no_compress
    
    def test_list_models(self, persistence_manager, sample_model):
        """Test listing saved models"""
        model, X, y = sample_model
        
        # Save multiple models
        for i in range(3):
            persistence_manager.save_model(
                model=model,
                model_name=f'model_{i}',
                metadata={'version': f'v1.{i}.0'}
            )
        
        models = persistence_manager.list_models()
        assert len(models) == 3
        assert 'model_0' in models
        assert 'model_1' in models
        assert 'model_2' in models
    
    def test_get_model_info_without_loading(self, persistence_manager, sample_model):
        """Test retrieving metadata without loading full model"""
        model, X, y = sample_model
        
        metadata = {
            'version': 'v1.0.0',
            'accuracy': 0.95,
            'author': 'test'
        }
        
        persistence_manager.save_model(
            model=model,
            model_name='test_model',
            metadata=metadata
        )
        
        info = persistence_manager.get_model_info('test_model')
        
        assert info['version'] == 'v1.0.0'
        assert info['accuracy'] == 0.95
        assert info['author'] == 'test'
    
    def test_validation_catches_feature_mismatch(self, persistence_manager, sample_model):
        """Test that validation detects feature count mismatch"""
        model, X, y = sample_model
        
        # Save with wrong feature count
        metadata = {
            'version': 'v1.0.0',
            'n_features': 999  # Wrong!
        }
        
        persistence_manager.save_model(
            model=model,
            model_name='bad_model',
            metadata=metadata
        )
        
        # Should raise assertion error on validation
        with pytest.raises(AssertionError):
            persistence_manager.load_model('bad_model', validate=True)


class TestModelVersionManager:
    """Test version management"""
    
    def test_register_version(self, persistence_manager):
        """Test registering model version"""
        version_manager = ModelVersionManager(persistence_manager)
        
        version_manager.register_version(
            model_name='fraud_detector',
            version='v1.0.0',
            metrics={'accuracy': 0.95, 'f1_score': 0.92},
            notes='Initial version'
        )
        
        assert len(version_manager.version_history) == 1
        assert version_manager.version_history[0]['version'] == 'v1.0.0'
    
    def test_compare_versions(self, persistence_manager):
        """Test version comparison"""
        version_manager = ModelVersionManager(persistence_manager)
        
        # Register two versions
        version_manager.register_version(
            model_name='model',
            version='v1.0.0',
            metrics={'accuracy': 0.90, 'f1_score': 0.88}
        )
        
        version_manager.register_version(
            model_name='model',
            version='v2.0.0',
            metrics={'accuracy': 0.95, 'f1_score': 0.93}
        )
        
        comparison = version_manager.compare_versions('v1.0.0', 'v2.0.0', metric='accuracy')
        
        assert comparison['v1_score'] == 0.90
        assert comparison['v2_score'] == 0.95
        assert abs(comparison['improvement'] - 0.05) < 1e-10
    
    def test_get_best_version(self, persistence_manager):
        """Test finding best version by metric"""
        version_manager = ModelVersionManager(persistence_manager)
        
        # Register versions with different scores
        versions = [
            ('v1.0.0', {'f1_score': 0.88}),
            ('v2.0.0', {'f1_score': 0.92}),
            ('v3.0.0', {'f1_score': 0.90}),
        ]
        
        for version, metrics in versions:
            version_manager.register_version(
                model_name='model',
                version=version,
                metrics=metrics
            )
        
        best = version_manager.get_best_version(metric='f1_score')
        assert best == 'v2.0.0'


class TestModelServer:
    """Test hot-swapping model server"""
    
    def test_server_loads_model_on_init(self, persistence_manager, sample_model, temp_models_dir):
        """Test that server loads model on initialization"""
        model, X, y = sample_model
        
        # Save model
        metadata = {'version': 'v1.0.0', 'n_features': X.shape[1]}
        persistence_manager.save_model(
            model=model,
            model_name='server_model',
            metadata=metadata
        )
        
        model_path = Path(temp_models_dir) / 'server_model.pkl'
        server = ModelServer(model_path)
        
        assert server.model is not None
        assert server.metadata['version'] == 'v1.0.0'
    
    def test_server_predict(self, persistence_manager, sample_model, temp_models_dir):
        """Test server predictions"""
        model, X, y = sample_model
        
        # Save model
        metadata = {'version': 'v1.0.0', 'n_features': X.shape[1]}
        persistence_manager.save_model(
            model=model,
            model_name='server_model',
            metadata=metadata
        )
        
        model_path = Path(temp_models_dir) / 'server_model.pkl'
        server = ModelServer(model_path)
        
        predictions = server.predict(X[:3])
        assert len(predictions) == 3
        assert server.request_count == 1
    
    def test_server_detects_model_updates(self, persistence_manager, sample_model, temp_models_dir):
        """Test that server detects and reloads updated models"""
        model, X, y = sample_model
        
        # Save initial model
        metadata_v1 = {'version': 'v1.0.0', 'n_features': X.shape[1]}
        persistence_manager.save_model(
            model=model,
            model_name='server_model',
            metadata=metadata_v1
        )
        
        model_path = Path(temp_models_dir) / 'server_model.pkl'
        server = ModelServer(model_path)
        
        assert server.metadata['version'] == 'v1.0.0'
        
        # Wait a moment to ensure timestamp differs
        time.sleep(0.1)
        
        # Update model
        metadata_v2 = {'version': 'v2.0.0', 'n_features': X.shape[1]}
        persistence_manager.save_model(
            model=model,
            model_name='server_model',
            metadata=metadata_v2
        )
        
        # Force reload check
        updated = server.reload_if_updated()
        
        assert updated == True
        assert server.metadata['version'] == 'v2.0.0'
    
    def test_server_status(self, persistence_manager, sample_model, temp_models_dir):
        """Test server status reporting"""
        model, X, y = sample_model
        
        metadata = {'version': 'v1.0.0', 'n_features': X.shape[1]}
        persistence_manager.save_model(
            model=model,
            model_name='server_model',
            metadata=metadata
        )
        
        model_path = Path(temp_models_dir) / 'server_model.pkl'
        server = ModelServer(model_path)
        
        # Make some predictions
        server.predict(X[:5])
        
        status = server.get_status()
        
        assert status['model_loaded'] == True
        assert status['version'] == 'v1.0.0'
        assert status['requests_served'] == 1


class TestTrainingPipeline:
    """Test complete training pipeline"""
    
    def test_train_fraud_detection_models(self):
        """Test training multiple models"""
        results = train_fraud_detection_models(n_samples=1000)
        
        assert 'models' in results
        assert 'test_data' in results
        assert len(results['models']) == 2  # LR and RF
        
        # Check models have required metadata
        for model_data in results['models'].values():
            assert 'model' in model_data
            assert 'metadata' in model_data
            assert 'accuracy' in model_data['metadata']
            assert 'f1_score' in model_data['metadata']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
