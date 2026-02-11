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
