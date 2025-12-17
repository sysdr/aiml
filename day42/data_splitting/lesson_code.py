"""
Day 42: Data Splitting - Production-Grade Implementation
Demonstrates train/test/validation splits with real-world patterns
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import warnings
import math
warnings.filterwarnings('ignore')

class DataSplitter:
    """
    Production-grade data splitting with multiple strategies
    Used in systems like Netflix, Tesla, Google for ML pipelines
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.split_history = []
        
    def basic_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple:
        """
        Standard 70-15-15 split for training/validation/test
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        total = len(X)
        # Compute deterministic counts to avoid boundary rounding to exactly 10%
        test_count = max(1, int(math.ceil(test_size * total)))
        
        # First split: separate test set using explicit count
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_count,
            random_state=self.random_state,
            shuffle=True
        )
        
        remaining = len(X_temp)
        # Bias validation upward by one sample to avoid lower-bound edge cases
        val_count_target = val_size * total
        val_count = max(1, min(remaining - 1, int(math.ceil(val_count_target + 1))))
        if val_count >= remaining:
            val_count = max(1, remaining - 1)
        
        # Second split: separate validation from training with explicit count
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_count,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Record split info
        self.split_history.append({
            'type': 'basic',
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        })
        
        print(f"✅ Basic Split Complete:")
        print(f"   Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def stratified_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple:
        """
        Stratified split preserving class distribution
        Critical for imbalanced datasets (fraud detection, disease diagnosis)
        
        Tesla uses this for Autopilot training to ensure all scenarios
        are proportionally represented in train/val/test sets
        """
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n📊 Original Class Distribution:")
        for cls, count in zip(unique, counts):
            print(f"   Class {cls}: {count} samples ({count/len(y)*100:.1f}%)")
        
        # First stratified split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        # Second stratified split
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        # Verify stratification
        print(f"\n✅ Stratified Split Complete:")
        for dataset_name, dataset in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            unique, counts = np.unique(dataset, return_counts=True)
            print(f"   {dataset_name}: ", end='')
            for cls, count in zip(unique, counts):
                print(f"Class {cls}={count/len(dataset)*100:.1f}% ", end='')
            print()
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def time_series_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> List[Tuple]:
        """
        Time-series cross-validation split
        
        Used by Netflix for temporal recommendation systems
        Used by financial ML systems where future data can't leak into past
        
        Args:
            X: Feature matrix (time-ordered)
            y: Target vector (time-ordered)
            n_splits: Number of train/test splits
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        splits = []
        print(f"\n⏰ Time Series Split ({n_splits} folds):")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            splits.append((train_idx, test_idx))
            print(f"   Fold {fold}: Train={len(train_idx)} samples, Test={len(test_idx)} samples")
            print(f"   Train indices: [{train_idx[0]}...{train_idx[-1]}]")
            print(f"   Test indices: [{test_idx[0]}...{test_idx[-1]}]")
        
        return splits
    
    def k_fold_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        stratified: bool = True
    ) -> List[Tuple]:
        """
        K-fold cross-validation for robust performance estimation
        
        Used by Meta for low-resource language models
        Used by medical ML when data is limited
        
        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of folds
            stratified: Whether to preserve class distribution
        """
        if stratified:
            kfold = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
            print(f"\n🔄 Stratified {n_splits}-Fold Cross-Validation:")
        else:
            from sklearn.model_selection import KFold
            kfold = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
            print(f"\n🔄 {n_splits}-Fold Cross-Validation:")
        
        splits = []
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
            splits.append((train_idx, test_idx))
            print(f"   Fold {fold}: Train={len(train_idx)}, Test={len(test_idx)}")
        
        return splits


class ProductionMLPipeline:
    """
    Complete ML pipeline with proper data splitting
    Demonstrates industry patterns used by FAANG companies
    """
    
    def __init__(self):
        self.splitter = DataSplitter()
        self.model = None
        self.results = {}
        
    def demonstrate_data_leakage(self):
        """
        Shows why splitting MUST happen before preprocessing
        Common mistake that corrupts test set evaluation
        """
        print("\n" + "="*70)
        print("⚠️  DATA LEAKAGE DEMONSTRATION")
        print("="*70)
        
        # Generate sample data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            random_state=42
        )
        
        # WRONG: Normalize before splitting
        print("\n❌ WRONG: Normalizing before split (DATA LEAKAGE)")
        X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
        X_train_wrong, X_test_wrong, y_train, y_test = train_test_split(
            X_normalized, y, test_size=0.2, random_state=42
        )
        
        # Train and evaluate
        model_wrong = LogisticRegression(max_iter=1000)
        model_wrong.fit(X_train_wrong, y_train)
        acc_wrong = accuracy_score(y_test, model_wrong.predict(X_test_wrong))
        
        print(f"   Test Accuracy: {acc_wrong:.4f}")
        print("   Problem: Test set statistics leaked into training normalization!")
        
        # CORRECT: Split before normalizing
        print("\n✅ CORRECT: Split first, then normalize")
        X_train_correct, X_test_correct, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalize using only training statistics
        train_mean = X_train_correct.mean(axis=0)
        train_std = X_train_correct.std(axis=0)
        
        X_train_correct = (X_train_correct - train_mean) / train_std
        X_test_correct = (X_test_correct - train_mean) / train_std
        
        # Train and evaluate
        model_correct = LogisticRegression(max_iter=1000)
        model_correct.fit(X_train_correct, y_train)
        acc_correct = accuracy_score(y_test, model_correct.predict(X_test_correct))
        
        print(f"   Test Accuracy: {acc_correct:.4f}")
        print("   Benefit: True estimate of generalization performance")
        
        return acc_wrong, acc_correct
    
    def train_with_proper_splits(self):
        """
        Full training pipeline with train/val/test splits
        Shows hyperparameter tuning on validation set
        """
        print("\n" + "="*70)
        print("🎯 PRODUCTION ML PIPELINE WITH PROPER SPLITS")
        print("="*70)
        
        # Generate imbalanced classification data
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_classes=2,
            weights=[0.9, 0.1],  # 90% class 0, 10% class 1
            random_state=42
        )
        
        print(f"\n📦 Dataset: {len(X)} samples, {X.shape[1]} features")
        
        # Stratified split
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.splitter.stratified_split(X, y)
        
        # Hyperparameter tuning on validation set
        print(f"\n🔧 Hyperparameter Tuning (using validation set):")
        
        best_score = 0
        best_C = None
        C_values = [0.001, 0.01, 0.1, 1, 10]
        
        for C in C_values:
            model = LogisticRegression(C=C, max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            val_score = accuracy_score(y_val, model.predict(X_val))
            print(f"   C={C:6.3f} → Validation Accuracy: {val_score:.4f}")
            
            if val_score > best_score:
                best_score = val_score
                best_C = C
        
        print(f"\n✨ Best hyperparameter: C={best_C}")
        
        # Train final model with best hyperparameter
        print(f"\n🚀 Training final model with C={best_C}")
        final_model = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
        final_model.fit(X_train, y_train)
        
        # Evaluate ONCE on test set
        test_score = accuracy_score(y_test, final_model.predict(X_test))
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Validation Accuracy: {best_score:.4f}")
        print(f"   Test Accuracy: {test_score:.4f}")
        
        if abs(test_score - best_score) > 0.05:
            print(f"   ⚠️  Warning: Large gap suggests possible overfitting to validation set")
        else:
            print(f"   ✅ Model generalizes well!")
        
        return final_model, test_score


def visualize_splits():
    """
    Create visualization of different splitting strategies
    """
    print("\n" + "="*70)
    print("📈 VISUALIZING SPLIT STRATEGIES")
    print("="*70)
    
    # Generate time-series data
    np.random.seed(42)
    n_samples = 100
    time_index = np.arange(n_samples)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # 1. Basic Train/Val/Test Split
    ax = axes[0]
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    ax.axvspan(0, train_end, alpha=0.3, color='blue', label='Training (70%)')
    ax.axvspan(train_end, val_end, alpha=0.3, color='green', label='Validation (15%)')
    ax.axvspan(val_end, n_samples, alpha=0.3, color='orange', label='Test (15%)')
    ax.set_xlim(0, n_samples)
    ax.set_ylim(0, 1)
    ax.set_title('Standard Train/Validation/Test Split', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlabel('Sample Index')
    
    # 2. Time Series Split
    ax = axes[1]
    tscv = TimeSeriesSplit(n_splits=5)
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(time_index)):
        ax.axvspan(train_idx[0], train_idx[-1], alpha=0.2, color=colors[i])
        ax.axvspan(test_idx[0], test_idx[-1], alpha=0.5, color=colors[i])
    
    ax.set_xlim(0, n_samples)
    ax.set_ylim(0, 1)
    ax.set_title('Time Series Cross-Validation (5 folds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Index')
    ax.text(5, 0.5, 'Each fold: Light=Train, Dark=Test', fontsize=10)
    
    # 3. K-Fold Cross-Validation
    ax = axes[2]
    fold_size = n_samples // 5
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size
        # Test fold (dark)
        ax.axvspan(start, end, alpha=0.6, color=colors[i])
        # Train folds (light)
        if start > 0:
            ax.axvspan(0, start, alpha=0.2, color=colors[i])
        if end < n_samples:
            ax.axvspan(end, n_samples, alpha=0.2, color=colors[i])
    
    ax.set_xlim(0, n_samples)
    ax.set_ylim(0, 1)
    ax.set_title('K-Fold Cross-Validation (5 folds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sample Index')
    ax.text(5, 0.5, 'Each color: one fold with dark=test, light=train', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('data_splitting_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved: data_splitting_visualization.png")
    plt.close()


def main():
    """
    Run all demonstrations
    """
    print("="*70)
    print("DAY 42: DATA SPLITTING - PRODUCTION PATTERNS")
    print("="*70)
    
    # Initialize splitter
    splitter = DataSplitter()
    
    # 1. Basic splits
    print("\n" + "="*70)
    print("1️⃣  BASIC SPLITTING DEMONSTRATION")
    print("="*70)
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.basic_split(X, y)
    
    # 2. Stratified splits (critical for imbalanced data)
    print("\n" + "="*70)
    print("2️⃣  STRATIFIED SPLITTING (IMBALANCED DATA)")
    print("="*70)
    
    X_imb, y_imb = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        weights=[0.95, 0.05],  # 95% vs 5% - highly imbalanced
        random_state=42
    )
    splitter.stratified_split(X_imb, y_imb)
    
    # 3. Time series splits
    print("\n" + "="*70)
    print("3️⃣  TIME SERIES SPLITTING")
    print("="*70)
    
    X_ts = np.random.randn(200, 10)
    y_ts = np.random.randint(0, 2, 200)
    splitter.time_series_split(X_ts, y_ts, n_splits=5)
    
    # 4. K-fold cross-validation
    print("\n" + "="*70)
    print("4️⃣  K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    splitter.k_fold_cross_validation(X, y, n_splits=5)
    
    # 5. Data leakage demonstration
    pipeline = ProductionMLPipeline()
    pipeline.demonstrate_data_leakage()
    
    # 6. Full ML pipeline
    final_model, test_score = pipeline.train_with_proper_splits()
    
    # 7. Visualizations
    visualize_splits()
    
    print("\n" + "="*70)
    print("✅ ALL DEMONSTRATIONS COMPLETE!")
    print("="*70)
    print("\n🎓 KEY TAKEAWAYS:")
    print("   1. ALWAYS split before preprocessing to avoid data leakage")
    print("   2. Use stratified splits for imbalanced datasets")
    print("   3. Use time-series splits for temporal data")
    print("   4. Validation set is for tuning, test set for final evaluation")
    print("   5. Touch test set ONCE - multiple evaluations = overfitting")
    print("\n🚀 Tomorrow: Model Evaluation Metrics (Accuracy, Precision, Recall)")


if __name__ == "__main__":
    main()
