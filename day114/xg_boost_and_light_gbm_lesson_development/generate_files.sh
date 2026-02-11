#!/bin/bash

# Day 114: XGBoost and LightGBM - Complete Implementation Package Generator
# This script creates all necessary files for the lesson

set -e

echo "Generating Day 114: XGBoost and LightGBM lesson files..."

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1.post1
xgboost==2.0.3
lightgbm==4.3.0
matplotlib==3.8.3
seaborn==0.13.2
pytest==8.0.2
imbalanced-learn==0.12.0
EOF

echo "✓ Created requirements.txt"

# Create setup.sh (nested setup script)
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 114: XGBoost and LightGBM environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download fraud detection dataset
echo "Downloading credit card fraud dataset..."
if [ ! -f "creditcard.csv" ]; then
    # Create synthetic fraud dataset for learning
    python3 << 'PYTHON_SCRIPT'
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate synthetic fraud detection dataset
# Similar structure to credit card fraud dataset
n_samples = 100000
n_features = 28

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=20,
    n_redundant=8,
    n_classes=2,
    weights=[0.998, 0.002],  # 0.2% fraud rate
    flip_y=0.01,
    random_state=42
)

# Create feature names similar to anonymized PCA features
feature_names = [f'V{i}' for i in range(1, n_features + 1)]

# Add time and amount features
time = np.random.randint(0, 172800, n_samples)  # 48 hours in seconds
amount = np.random.lognormal(3.5, 1.5, n_samples)

# Combine into dataframe
df = pd.DataFrame(X, columns=feature_names)
df.insert(0, 'Time', time)
df['Amount'] = amount
df['Class'] = y

# Save to CSV
df.to_csv('creditcard.csv', index=False)
print(f"Created synthetic fraud dataset: {len(df)} transactions, {y.sum()} fraudulent")
PYTHON_SCRIPT
fi

echo "✓ Environment setup complete!"
echo "Run: source venv/bin/activate"
EOF

chmod +x setup.sh
echo "✓ Created setup.sh"

# Create lesson_code.py
cat > lesson_code.py << 'EOF'
"""
Day 114: XGBoost and LightGBM - Production Fraud Detection
Production-grade implementation of advanced gradient boosting frameworks
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, 
    precision_recall_curve, confusion_matrix, f1_score
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionSystem:
    """Production fraud detection using XGBoost and LightGBM"""
    
    def __init__(self, data_path='creditcard.csv'):
        """Initialize fraud detection system"""
        self.data_path = data_path
        self.xgb_model = None
        self.lgb_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_and_prepare_data(self, test_size=0.2, random_state=42):
        """Load and prepare fraud detection dataset"""
        print("Loading fraud detection dataset...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Fraud rate: {df['Class'].mean()*100:.3f}%")
        print(f"Fraudulent transactions: {df['Class'].sum()}")
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        self.feature_names = X.columns.tolist()
        
        # Stratified split to preserve fraud ratio
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Training fraud rate: {self.y_train.mean()*100:.3f}%")
        
        return self
    
    def train_xgboost(self, params=None):
        """Train XGBoost model with production configuration"""
        print("\n" + "="*60)
        print("Training XGBoost Model")
        print("="*60)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1])
        
        # Default production parameters
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'scale_pos_weight': scale_pos_weight,
            'tree_method': 'hist',  # Fast histogram-based
            'eval_metric': 'auc',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
        
        if params:
            default_params.update(params)
        
        # Train model
        self.xgb_model = xgb.XGBClassifier(**default_params)
        
        start_time = time.time()
        self.xgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        training_time = time.time() - start_time
        
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        # Evaluate
        y_pred = self.xgb_model.predict(self.X_test)
        y_pred_proba = self.xgb_model.predict_proba(self.X_test)[:, 1]
        
        print(f"\nPerformance Metrics:")
        print(f"ROC-AUC Score: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred):.4f}")
        
        return self
    
    def train_lightgbm(self, params=None):
        """Train LightGBM model with production configuration"""
        print("\n" + "="*60)
        print("Training LightGBM Model")
        print("="*60)
        
        # Default production parameters
        default_params = {
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary',
            'is_unbalance': True,  # Handle class imbalance
            'boosting_type': 'gbdt',
            'metric': 'auc',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
        }
        
        if params:
            default_params.update(params)
        
        # Train model
        self.lgb_model = lgb.LGBMClassifier(**default_params)
        
        start_time = time.time()
        self.lgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
        )
        training_time = time.time() - start_time
        
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        # Evaluate
        y_pred = self.lgb_model.predict(self.X_test)
        y_pred_proba = self.lgb_model.predict_proba(self.X_test)[:, 1]
        
        print(f"\nPerformance Metrics:")
        print(f"ROC-AUC Score: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred):.4f}")
        
        return self
    
    def benchmark_inference(self, n_iterations=100, batch_size=1000):
        """Benchmark inference performance"""
        print("\n" + "="*60)
        print("Inference Performance Benchmark")
        print("="*60)
        
        # Prepare test batch
        test_batch = self.X_test.iloc[:batch_size]
        
        # Benchmark XGBoost
        xgb_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.xgb_model.predict_proba(test_batch)
            xgb_times.append(time.perf_counter() - start)
        
        xgb_mean = np.mean(xgb_times) * 1000
        xgb_std = np.std(xgb_times) * 1000
        
        # Benchmark LightGBM
        lgb_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.lgb_model.predict_proba(test_batch)
            lgb_times.append(time.perf_counter() - start)
        
        lgb_mean = np.mean(lgb_times) * 1000
        lgb_std = np.std(lgb_times) * 1000
        
        print(f"\nBatch size: {batch_size} predictions")
        print(f"Iterations: {n_iterations}")
        print(f"\nXGBoost: {xgb_mean:.2f}ms ± {xgb_std:.2f}ms")
        print(f"  Per prediction: {xgb_mean/batch_size:.4f}ms")
        print(f"  Throughput: {batch_size/(xgb_mean/1000):.0f} predictions/sec")
        
        print(f"\nLightGBM: {lgb_mean:.2f}ms ± {lgb_std:.2f}ms")
        print(f"  Per prediction: {lgb_mean/batch_size:.4f}ms")
        print(f"  Throughput: {batch_size/(lgb_mean/1000):.0f} predictions/sec")
        
        speedup = xgb_mean / lgb_mean
        print(f"\nLightGBM speedup: {speedup:.2f}x")
        
        return {
            'xgb_mean': xgb_mean,
            'lgb_mean': lgb_mean,
            'speedup': speedup
        }
    
    def compare_feature_importance(self, top_n=15):
        """Compare feature importance between models"""
        print("\n" + "="*60)
        print("Feature Importance Analysis")
        print("="*60)
        
        # Get feature importances
        xgb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        lgb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.lgb_model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        print(f"\nTop {top_n} Features - XGBoost:")
        print(xgb_importance.to_string(index=False))
        
        print(f"\nTop {top_n} Features - LightGBM:")
        print(lgb_importance.to_string(index=False))
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # XGBoost importance
        axes[0].barh(range(top_n), xgb_importance['importance'].values)
        axes[0].set_yticks(range(top_n))
        axes[0].set_yticklabels(xgb_importance['feature'].values)
        axes[0].set_xlabel('Importance Score')
        axes[0].set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
        
        # LightGBM importance
        axes[1].barh(range(top_n), lgb_importance['importance'].values)
        axes[1].set_yticks(range(top_n))
        axes[1].set_yticklabels(lgb_importance['feature'].values)
        axes[1].set_xlabel('Importance Score')
        axes[1].set_title('LightGBM Feature Importance', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved feature_importance_comparison.png")
        
        return xgb_importance, lgb_importance
    
    def plot_roc_curves(self):
        """Plot ROC curves for both models"""
        # Get predictions
        xgb_proba = self.xgb_model.predict_proba(self.X_test)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate ROC curves
        xgb_fpr, xgb_tpr, _ = roc_curve(self.y_test, xgb_proba)
        lgb_fpr, lgb_tpr, _ = roc_curve(self.y_test, lgb_proba)
        
        xgb_auc = roc_auc_score(self.y_test, xgb_proba)
        lgb_auc = roc_auc_score(self.y_test, lgb_proba)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.4f})', linewidth=2)
        plt.plot(lgb_fpr, lgb_tpr, label=f'LightGBM (AUC = {lgb_auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Fraud Detection', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Saved roc_curves.png")
    
    def generate_production_report(self):
        """Generate comprehensive production analysis report"""
        print("\n" + "="*60)
        print("PRODUCTION DEPLOYMENT ANALYSIS")
        print("="*60)
        
        # Predictions
        xgb_pred = self.xgb_model.predict(self.X_test)
        lgb_pred = self.lgb_model.predict(self.X_test)
        
        xgb_proba = self.xgb_model.predict_proba(self.X_test)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(self.X_test)[:, 1]
        
        # Confusion matrices
        xgb_cm = confusion_matrix(self.y_test, xgb_pred)
        lgb_cm = confusion_matrix(self.y_test, lgb_pred)
        
        print("\nXGBoost Confusion Matrix:")
        print(f"True Negatives: {xgb_cm[0,0]}, False Positives: {xgb_cm[0,1]}")
        print(f"False Negatives: {xgb_cm[1,0]}, True Positives: {xgb_cm[1,1]}")
        
        print("\nLightGBM Confusion Matrix:")
        print(f"True Negatives: {lgb_cm[0,0]}, False Positives: {lgb_cm[0,1]}")
        print(f"False Negatives: {lgb_cm[1,0]}, True Positives: {lgb_cm[1,1]}")
        
        # Calculate costs (example: FP costs $10, FN costs $100)
        fp_cost = 10
        fn_cost = 100
        
        xgb_total_cost = xgb_cm[0,1] * fp_cost + xgb_cm[1,0] * fn_cost
        lgb_total_cost = lgb_cm[0,1] * fp_cost + lgb_cm[1,0] * fn_cost
        
        print(f"\nEstimated Costs (FP=${fp_cost}, FN=${fn_cost}):")
        print(f"XGBoost total cost: ${xgb_total_cost}")
        print(f"LightGBM total cost: ${lgb_total_cost}")
        
        # Model agreement analysis
        agreement = (xgb_pred == lgb_pred).sum() / len(xgb_pred)
        print(f"\nModel Agreement: {agreement*100:.2f}%")
        
        # Cases where models disagree
        disagreement_mask = xgb_pred != lgb_pred
        print(f"Disagreement cases: {disagreement_mask.sum()} ({disagreement_mask.sum()/len(xgb_pred)*100:.2f}%)")
        
        return {
            'xgb_cm': xgb_cm,
            'lgb_cm': lgb_cm,
            'agreement': agreement,
            'xgb_cost': xgb_total_cost,
            'lgb_cost': lgb_total_cost
        }


def main():
    """Main execution function"""
    print("="*60)
    print("Day 114: XGBoost and LightGBM")
    print("Production Fraud Detection System")
    print("="*60)
    
    # Initialize system
    system = FraudDetectionSystem()
    
    # Load data
    system.load_and_prepare_data()
    
    # Train both models
    system.train_xgboost()
    system.train_lightgbm()
    
    # Benchmark inference
    system.benchmark_inference()
    
    # Analyze features
    system.compare_feature_importance()
    
    # Plot ROC curves
    system.plot_roc_curves()
    
    # Generate production report
    system.generate_production_report()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("Generated outputs:")
    print("  - feature_importance_comparison.png")
    print("  - roc_curves.png")
    print("="*60)


if __name__ == "__main__":
    main()
EOF

echo "✓ Created lesson_code.py"

# Create test_lesson.py
cat > test_lesson.py << 'EOF'
"""
Day 114: XGBoost and LightGBM - Comprehensive Test Suite
Tests for production fraud detection implementation
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import os


@pytest.fixture
def sample_data():
    """Generate sample fraud detection data"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def xgb_model(sample_data):
    """Train XGBoost model"""
    X_train, X_test, y_train, y_test = sample_data
    
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=50,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def lgb_model(sample_data):
    """Train LightGBM model"""
    X_train, X_test, y_train, y_test = sample_data
    
    model = lgb.LGBMClassifier(
        num_leaves=15,
        learning_rate=0.1,
        n_estimators=50,
        is_unbalance=True,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    return model


class TestXGBoost:
    """Test suite for XGBoost implementation"""
    
    def test_model_creation(self):
        """Test XGBoost model can be created"""
        model = xgb.XGBClassifier()
        assert model is not None
    
    def test_model_training(self, sample_data):
        """Test XGBoost model can be trained"""
        X_train, X_test, y_train, y_test = sample_data
        
        model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X_train.shape[1]
    
    def test_predictions(self, xgb_model, sample_data):
        """Test XGBoost predictions"""
        _, X_test, _, y_test = sample_data
        
        predictions = xgb_model.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
    
    def test_probability_predictions(self, xgb_model, sample_data):
        """Test XGBoost probability predictions"""
        _, X_test, _, _ = sample_data
        
        probas = xgb_model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert (probas >= 0).all() and (probas <= 1).all()
    
    def test_feature_importance(self, xgb_model, sample_data):
        """Test feature importance extraction"""
        X_train, _, _, _ = sample_data
        
        importances = xgb_model.feature_importances_
        
        assert len(importances) == X_train.shape[1]
        assert (importances >= 0).all()
        assert importances.sum() > 0
    
    def test_scale_pos_weight(self, sample_data):
        """Test class imbalance handling"""
        X_train, X_test, y_train, y_test = sample_data
        
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Model should predict some positive cases despite imbalance
        predictions = model.predict(X_test)
        assert predictions.sum() > 0
    
    def test_early_stopping(self, sample_data):
        """Test early stopping functionality"""
        X_train, X_test, y_train, y_test = sample_data
        
        model = xgb.XGBClassifier(
            n_estimators=1000,
            early_stopping_rounds=10,
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Should stop before 1000 iterations
        assert model.best_iteration < 1000
    
    def test_tree_method_hist(self, sample_data):
        """Test histogram-based tree method"""
        X_train, _, y_train, _ = sample_data
        
        model = xgb.XGBClassifier(
            tree_method='hist',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')


class TestLightGBM:
    """Test suite for LightGBM implementation"""
    
    def test_model_creation(self):
        """Test LightGBM model can be created"""
        model = lgb.LGBMClassifier(verbose=-1)
        assert model is not None
    
    def test_model_training(self, sample_data):
        """Test LightGBM model can be trained"""
        X_train, X_test, y_train, y_test = sample_data
        
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X_train.shape[1]
    
    def test_predictions(self, lgb_model, sample_data):
        """Test LightGBM predictions"""
        _, X_test, _, y_test = sample_data
        
        predictions = lgb_model.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
    
    def test_probability_predictions(self, lgb_model, sample_data):
        """Test LightGBM probability predictions"""
        _, X_test, _, _ = sample_data
        
        probas = lgb_model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert (probas >= 0).all() and (probas <= 1).all()
    
    def test_feature_importance(self, lgb_model, sample_data):
        """Test feature importance extraction"""
        X_train, _, _, _ = sample_data
        
        importances = lgb_model.feature_importances_
        
        assert len(importances) == X_train.shape[1]
        assert (importances >= 0).all()
        assert importances.sum() > 0
    
    def test_is_unbalance(self, sample_data):
        """Test class imbalance handling"""
        X_train, X_test, y_train, y_test = sample_data
        
        model = lgb.LGBMClassifier(
            is_unbalance=True,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Model should predict some positive cases despite imbalance
        predictions = model.predict(X_test)
        assert predictions.sum() > 0
    
    def test_leaf_wise_growth(self, sample_data):
        """Test leaf-wise tree growth"""
        X_train, _, y_train, _ = sample_data
        
        model = lgb.LGBMClassifier(
            num_leaves=31,
            boosting_type='gbdt',
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')
    
    def test_categorical_features(self, sample_data):
        """Test categorical feature handling"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Add a categorical feature
        X_train_cat = np.column_stack([
            X_train,
            np.random.randint(0, 5, len(X_train))
        ])
        X_test_cat = np.column_stack([
            X_test,
            np.random.randint(0, 5, len(X_test))
        ])
        
        model = lgb.LGBMClassifier(
            categorical_feature=[X_train.shape[1]],
            random_state=42,
            verbose=-1
        )
        model.fit(X_train_cat, y_train)
        predictions = model.predict(X_test_cat)
        
        assert len(predictions) == len(y_test)


class TestModelComparison:
    """Test suite for comparing XGBoost and LightGBM"""
    
    def test_similar_performance(self, xgb_model, lgb_model, sample_data):
        """Test both models achieve similar performance"""
        _, X_test, _, y_test = sample_data
        
        xgb_score = xgb_model.score(X_test, y_test)
        lgb_score = lgb_model.score(X_test, y_test)
        
        # Both should achieve reasonable accuracy
        assert xgb_score > 0.7
        assert lgb_score > 0.7
        
        # Scores should be similar (within 10%)
        assert abs(xgb_score - lgb_score) < 0.1
    
    def test_inference_speed(self, xgb_model, lgb_model, sample_data):
        """Test inference speed comparison"""
        _, X_test, _, _ = sample_data
        
        import time
        
        # Warmup
        xgb_model.predict(X_test[:10])
        lgb_model.predict(X_test[:10])
        
        # Benchmark XGBoost
        start = time.perf_counter()
        for _ in range(10):
            xgb_model.predict(X_test)
        xgb_time = time.perf_counter() - start
        
        # Benchmark LightGBM
        start = time.perf_counter()
        for _ in range(10):
            lgb_model.predict(X_test)
        lgb_time = time.perf_counter() - start
        
        # Both should complete quickly
        assert xgb_time < 1.0
        assert lgb_time < 1.0
    
    def test_feature_importance_correlation(self, xgb_model, lgb_model):
        """Test feature importance correlation between models"""
        xgb_importance = xgb_model.feature_importances_
        lgb_importance = lgb_model.feature_importances_
        
        # Normalize importances
        xgb_norm = xgb_importance / xgb_importance.sum()
        lgb_norm = lgb_importance / lgb_importance.sum()
        
        # Calculate correlation
        correlation = np.corrcoef(xgb_norm, lgb_norm)[0, 1]
        
        # Should have positive correlation
        assert correlation > 0.5


class TestProductionReadiness:
    """Test suite for production deployment readiness"""
    
    def test_batch_prediction(self, xgb_model, sample_data):
        """Test batch prediction capability"""
        _, X_test, _, _ = sample_data
        
        # Test different batch sizes
        for batch_size in [1, 10, 100]:
            predictions = xgb_model.predict(X_test[:batch_size])
            assert len(predictions) == batch_size
    
    def test_missing_value_handling(self, sample_data):
        """Test handling of missing values"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Introduce missing values
        X_train_missing = X_train.copy()
        X_test_missing = X_test.copy()
        
        mask_train = np.random.random(X_train.shape) < 0.1
        mask_test = np.random.random(X_test.shape) < 0.1
        
        X_train_missing[mask_train] = np.nan
        X_test_missing[mask_test] = np.nan
        
        # XGBoost should handle missing values natively
        model = xgb.XGBClassifier(random_state=42)
        model.fit(X_train_missing, y_train)
        predictions = model.predict(X_test_missing)
        
        assert len(predictions) == len(y_test)
        assert not np.isnan(predictions).any()
    
    def test_model_serialization(self, xgb_model, lgb_model, tmp_path):
        """Test model saving and loading"""
        import joblib
        
        # Save models
        xgb_path = tmp_path / "xgb_model.pkl"
        lgb_path = tmp_path / "lgb_model.pkl"
        
        joblib.dump(xgb_model, xgb_path)
        joblib.dump(lgb_model, lgb_path)
        
        # Load models
        xgb_loaded = joblib.load(xgb_path)
        lgb_loaded = joblib.load(lgb_path)
        
        assert xgb_loaded is not None
        assert lgb_loaded is not None
    
    def test_consistent_predictions(self, xgb_model, sample_data):
        """Test prediction consistency"""
        _, X_test, _, _ = sample_data
        
        # Make predictions multiple times
        pred1 = xgb_model.predict(X_test)
        pred2 = xgb_model.predict(X_test)
        pred3 = xgb_model.predict(X_test)
        
        # Predictions should be identical
        assert np.array_equal(pred1, pred2)
        assert np.array_equal(pred2, pred3)
    
    def test_memory_efficiency(self, sample_data):
        """Test memory-efficient training"""
        X_train, _, y_train, _ = sample_data
        
        # Train with subsample to reduce memory
        model = lgb.LGBMClassifier(
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'feature_importances_')


def test_data_generation():
    """Test synthetic fraud data generation"""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42
    )
    
    assert X.shape == (1000, 20)
    assert len(y) == 1000
    assert y.sum() > 0  # Has some positive cases
    assert y.sum() < len(y) * 0.1  # Imbalanced


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
EOF

echo "✓ Created test_lesson.py"

# Create README.md
cat > README.md << 'EOF'
# Day 114: XGBoost and LightGBM - Production Fraud Detection

Production-grade implementation of advanced gradient boosting frameworks for high-scale fraud detection systems.

## Overview

This lesson demonstrates:
- XGBoost and LightGBM implementation for fraud detection
- Performance benchmarking and optimization
- Feature importance analysis
- Production deployment patterns

## Quick Start

### 1. Setup Environment

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run Main Implementation

```bash
python lesson_code.py
```

Expected output:
- Training metrics for both models
- Inference performance benchmarks
- Feature importance analysis
- ROC curves and visualizations

### 3. Run Tests

```bash
pytest test_lesson.py -v
```

Expected: 25+ tests passing

## Project Structure

```
day_114_xgboost_lightgbm/
├── lesson_code.py          # Main implementation
├── test_lesson.py          # Comprehensive test suite
├── setup.sh                # Environment setup
├── requirements.txt        # Dependencies
├── creditcard.csv          # Fraud detection dataset (generated)
└── README.md               # This file
```

## Key Concepts

### XGBoost Optimizations
- Sparsity-aware split finding
- Weighted quantile sketching
- Cache-aware parallelization
- Histogram-based tree building

### LightGBM Innovations
- Gradient-based One-Side Sampling (GOSS)
- Leaf-wise tree growth
- Categorical feature support
- Memory-efficient training

### Production Patterns
- Class imbalance handling
- Early stopping for efficiency
- Feature importance extraction
- Inference performance benchmarking

## Expected Results

### Training Performance
- XGBoost: ~5-15 seconds
- LightGBM: ~3-10 seconds (typically 2-3x faster)

### Model Accuracy
- ROC-AUC: >0.95 on fraud detection
- F1 Score: >0.85 with proper threshold tuning

### Inference Speed
- XGBoost: ~100K predictions/second
- LightGBM: ~150K predictions/second

## Real-World Applications

### Fraud Detection (PayPal, Stripe)
- Real-time transaction scoring
- Adaptive fraud pattern learning
- Explainable predictions for compliance

### Dynamic Pricing (Uber, Airbnb)
- Sub-second price predictions
- Multi-feature optimization
- High-frequency model updates

### Ad Click Prediction (Microsoft Bing)
- Billions of predictions daily
- Feature importance for campaign optimization
- Memory-efficient training on massive datasets

## Common Issues

### Out of Memory
- Reduce `num_leaves` (LightGBM) or `max_depth` (XGBoost)
- Enable `subsample` and `colsample_bytree`
- Use `tree_method='hist'` for XGBoost

### Slow Training
- Reduce `n_estimators` initially
- Enable early stopping
- Use GPU training if available (`tree_method='gpu_hist'`)

### Poor Performance on Imbalanced Data
- Tune `scale_pos_weight` (XGBoost) or `is_unbalance` (LightGBM)
- Adjust prediction threshold
- Use stratified sampling

## Next Steps

Tomorrow (Day 115): **Bias-Variance Tradeoff**
- Diagnosing model errors
- Underfitting vs overfitting
- Regularization strategies
- Learning curves analysis

## Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [LightGBM Paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
EOF

echo "✓ Created README.md"

echo ""
echo "=========================================="
echo "✓ All files generated successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. chmod +x setup.sh"
echo "2. ./setup.sh"
echo "3. source venv/bin/activate"
echo "4. python lesson_code.py"
echo "5. pytest test_lesson.py -v"
echo ""

