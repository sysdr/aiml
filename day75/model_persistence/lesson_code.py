"""
Day 75: Model Persistence - Production-Ready Model Saving and Loading

This module demonstrates enterprise-grade model persistence patterns:
- Serialization with joblib (compression, validation)
- Version management with metadata tracking
- Hot-swapping for zero-downtime updates
- Model comparison and rollback capabilities
"""

import joblib
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')


class ModelPersistence:
    """
    Production model persistence with compression and validation.
    
    Similar to how Stripe manages payment fraud models:
    - Compressed serialization (70% size reduction)
    - Integrity validation on load
    - Metadata bundling for debugging
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Dict[str, Any],
        compress: int = 3
    ) -> str:
        """
        Save model with metadata and compression.
        
        Args:
            model: Trained scikit-learn model
            model_name: Unique model identifier (e.g., 'fraud_v1.0.0')
            metadata: Performance metrics, hyperparameters, features
            compress: Compression level 0-9 (3=balanced, 9=maximum)
        
        Returns:
            Path to saved model file
        """
        # Create model bundle
        bundle = {
            'model': model,
            'metadata': {
                **metadata,
                'saved_at': datetime.now().isoformat(),
                'scikit_learn_version': joblib.__version__,
                'model_type': type(model).__name__
            }
        }
        
        # Save with compression
        filepath = self.models_dir / f"{model_name}.pkl"
        joblib.dump(bundle, filepath, compress=compress)
        
        # Save metadata separately for quick access
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(bundle['metadata'], f, indent=2)
        
        # Get file size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✅ Model saved: {filepath} ({size_mb:.2f} MB)")
        
        return str(filepath)
    
    def load_model(
        self,
        model_name: str,
        validate: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model with validation.
        
        Args:
            model_name: Model identifier to load
            validate: Run sanity checks on loaded model
        
        Returns:
            Tuple of (model, metadata)
        """
        filepath = self.models_dir / f"{model_name}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")
        
        # Load bundle
        bundle = joblib.load(filepath)
        model = bundle['model']
        metadata = bundle['metadata']
        
        print(f"📦 Loaded model: {model_name}")
        print(f"   Type: {metadata['model_type']}")
        print(f"   Saved: {metadata['saved_at']}")
        
        if validate:
            self._validate_model(model, metadata)
        
        return model, metadata
    
    def _validate_model(self, model: Any, metadata: Dict[str, Any]):
        """Validate model integrity after loading"""
        # Check model has expected attributes
        if not hasattr(model, 'predict'):
            raise ValueError("Model missing predict method")
        
        # Check feature count matches metadata
        if 'n_features' in metadata:
            if hasattr(model, 'n_features_in_'):
                assert model.n_features_in_ == metadata['n_features'], \
                    f"Feature mismatch: {model.n_features_in_} vs {metadata['n_features']}"
        
        print("   ✓ Validation passed")
    
    def list_models(self) -> List[str]:
        """List all saved models"""
        models = [f.stem for f in self.models_dir.glob("*.pkl")]
        return models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata without loading full model"""
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        # Fallback: load full bundle
        _, metadata = self.load_model(model_name, validate=False)
        return metadata


class ModelVersionManager:
    """
    Version control for ML models - Netflix style.
    
    Tracks model lineage, compares versions, enables rollback.
    """
    
    def __init__(self, persistence: ModelPersistence):
        self.persistence = persistence
        self.version_history: List[Dict[str, Any]] = []
    
    def register_version(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        notes: str = ""
    ):
        """Register new model version"""
        version_info = {
            'model_name': model_name,
            'version': version,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'notes': notes
        }
        
        self.version_history.append(version_info)
        
        # Save version history
        history_path = self.persistence.models_dir / "version_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.version_history, f, indent=2)
    
    def compare_versions(
        self,
        version1: str,
        version2: str,
        metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        v1_info = next((v for v in self.version_history if v['version'] == version1), None)
        v2_info = next((v for v in self.version_history if v['version'] == version2), None)
        
        if not v1_info or not v2_info:
            raise ValueError("Version not found in history")
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'metric': metric,
            'v1_score': v1_info['metrics'].get(metric, 0),
            'v2_score': v2_info['metrics'].get(metric, 0),
            'improvement': v2_info['metrics'].get(metric, 0) - v1_info['metrics'].get(metric, 0)
        }
        
        return comparison
    
    def get_best_version(self, metric: str = 'f1_score') -> Optional[str]:
        """Get best performing version by metric"""
        if not self.version_history:
            return None
        
        best = max(
            self.version_history,
            key=lambda v: v['metrics'].get(metric, 0)
        )
        
        return best['version']


class ModelServer:
    """
    Production model serving with hot-swapping - Uber style.
    
    Automatically detects model updates and reloads without downtime.
    """
    
    def __init__(self, model_path: Path, check_interval: int = 30):
        self.model_path = Path(model_path)
        self.check_interval = check_interval
        self.model = None
        self.metadata = None
        self.last_modified = None
        self.request_count = 0
        
        # Initial load
        self.reload_if_updated()
    
    def reload_if_updated(self) -> bool:
        """Check if model file updated and reload if necessary"""
        if not self.model_path.exists():
            print(f"⚠️  Model file not found: {self.model_path}")
            return False
        
        current_modified = os.path.getmtime(self.model_path)
        
        # First load or file changed
        if self.last_modified is None or current_modified > self.last_modified:
            print(f"🔄 Loading model from {self.model_path.name}")
            
            bundle = joblib.load(self.model_path)
            self.model = bundle['model']
            self.metadata = bundle['metadata']
            self.last_modified = current_modified
            
            print(f"   ✅ Loaded: {self.metadata['model_type']}")
            print(f"   Version: {self.metadata.get('version', 'unknown')}")
            
            return True
        
        return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with automatic reload check"""
        # Periodically check for updates
        if self.request_count % 100 == 0:
            self.reload_if_updated()
        
        self.request_count += 1
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        return self.model.predict(X)
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        return {
            'model_loaded': self.model is not None,
            'model_type': self.metadata.get('model_type', 'unknown') if self.metadata else None,
            'version': self.metadata.get('version', 'unknown') if self.metadata else None,
            'requests_served': self.request_count,
            'last_updated': datetime.fromtimestamp(self.last_modified).isoformat() if self.last_modified else None
        }


def train_fraud_detection_models(n_samples: int = 10000) -> Dict[str, Any]:
    """
    Train multiple fraud detection models for demonstration.
    
    Simulates real-world scenario where you train several models
    and need to persist the best performer.
    """
    print("🎯 Training fraud detection models...\n")
    
    # Generate synthetic fraud detection dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],  # Imbalanced: 90% legitimate, 10% fraud
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    
    models['logistic_regression_v1'] = {
        'model': lr,
        'scaler': scaler,
        'metadata': {
            'version': 'v1.0.0',
            'model_name': 'logistic_regression',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'n_features': X_train.shape[1],
            'feature_names': feature_names,
            'training_samples': X_train.shape[0],
            'hyperparameters': {
                'max_iter': 1000,
                'random_state': 42
            }
        }
    }
    
    print(f"  Accuracy: {models['logistic_regression_v1']['metadata']['accuracy']:.4f}")
    print(f"  F1 Score: {models['logistic_regression_v1']['metadata']['f1_score']:.4f}\n")
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=50,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    models['random_forest_v1'] = {
        'model': rf,
        'scaler': None,  # RF doesn't need scaling
        'metadata': {
            'version': 'v1.0.0',
            'model_name': 'random_forest',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'n_features': X_train.shape[1],
            'feature_names': feature_names,
            'training_samples': X_train.shape[0],
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 50,
                'random_state': 42
            }
        }
    }
    
    print(f"  Accuracy: {models['random_forest_v1']['metadata']['accuracy']:.4f}")
    print(f"  F1 Score: {models['random_forest_v1']['metadata']['f1_score']:.4f}\n")
    
    return {
        'models': models,
        'test_data': (X_test, y_test),
        'test_data_scaled': (X_test_scaled, y_test)
    }


def demonstrate_persistence():
    """Main demonstration of model persistence patterns"""
    print("=" * 60)
    print("Day 75: Model Persistence Demo")
    print("=" * 60 + "\n")
    
    # Initialize persistence manager
    persistence = ModelPersistence(models_dir="models")
    
    # Train models
    results = train_fraud_detection_models(n_samples=10000)
    models = results['models']
    
    # Save all models
    print("💾 Saving models...\n")
    saved_paths = {}
    
    for model_key, model_data in models.items():
        path = persistence.save_model(
            model=model_data['model'],
            model_name=model_key,
            metadata=model_data['metadata'],
            compress=3  # Balanced compression
        )
        saved_paths[model_key] = path
    
    print()
    
    # List all models
    print("📋 Available models:")
    for model_name in persistence.list_models():
        print(f"   - {model_name}")
    print()
    
    # Load and validate model
    print("📦 Loading Random Forest model...\n")
    loaded_model, metadata = persistence.load_model('random_forest_v1')
    
    print(f"\nModel metadata:")
    print(f"   Version: {metadata['version']}")
    print(f"   Accuracy: {metadata['accuracy']:.4f}")
    print(f"   F1 Score: {metadata['f1_score']:.4f}")
    print(f"   Features: {metadata['n_features']}")
    print()
    
    # Test loaded model
    X_test, y_test = results['test_data']
    predictions = loaded_model.predict(X_test[:5])
    print(f"Test predictions: {predictions}")
    print()
    
    # Version management demo
    print("📊 Version Management Demo\n")
    version_manager = ModelVersionManager(persistence)
    
    # Register versions
    for model_key, model_data in models.items():
        version_manager.register_version(
            model_name=model_data['metadata']['model_name'],
            version=model_data['metadata']['version'],
            metrics={
                'accuracy': model_data['metadata']['accuracy'],
                'f1_score': model_data['metadata']['f1_score']
            },
            notes="Initial training"
        )
    
    # Compare versions
    comparison = version_manager.compare_versions('v1.0.0', 'v1.0.0', metric='f1_score')
    print(f"Version comparison: {comparison}")
    print()
    
    # Get best version
    best_version = version_manager.get_best_version(metric='f1_score')
    print(f"Best version by F1 score: {best_version}")
    print()
    
    # Model server demo (hot-swapping)
    print("🔄 Model Server Demo (Hot-Swapping)\n")
    model_path = Path("models/random_forest_v1.pkl")
    server = ModelServer(model_path)
    
    # Make predictions
    print("Making predictions...")
    for i in range(5):
        pred = server.predict(X_test[i:i+1])
        print(f"   Request {i+1}: Prediction = {pred[0]}")
    
    print()
    status = server.get_status()
    print(f"Server status: {json.dumps(status, indent=2)}")
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_persistence()
