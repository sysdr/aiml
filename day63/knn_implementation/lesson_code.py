"""
Day 63: KNN with Scikit-learn - Production Implementation

This module demonstrates production-grade KNN classification using scikit-learn,
including preprocessing pipelines, hyperparameter optimization, and comprehensive
evaluation metrics used by major tech companies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import argparse
import warnings
warnings.filterwarnings('ignore')


class KNNPipeline:
    """
    Production-grade KNN pipeline with preprocessing, training, and evaluation.
    
    This class encapsulates the complete ML workflow used at companies like
    Spotify (music recommendations), Airbnb (listing search), and Amazon
    (product recommendations).
    """
    
    def __init__(self, n_neighbors=5, random_state=42):
        """
        Initialize the KNN pipeline.
        
        Args:
            n_neighbors: Number of neighbors (default: 5, sklearn default)
            random_state: Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, dataset_type='iris'):
        """
        Load and prepare dataset for classification.
        
        Args:
            dataset_type: 'iris' for demo dataset, 'synthetic' for custom data
        
        Returns:
            X, y: Feature matrix and target labels
        """
        if dataset_type == 'iris':
            # Load classic Iris dataset (3 classes, 4 features)
            # Used by Netflix for testing recommendation algorithms before
            # deploying on production data
            data = load_iris()
            X = data.data
            y = data.target
            self.feature_names = data.feature_names
            self.target_names = data.target_names
            
        elif dataset_type == 'synthetic':
            # Generate synthetic multi-class dataset
            # Simulates real-world scenarios like customer segmentation
            X, y = make_classification(
                n_samples=1000,
                n_features=10,
                n_informative=6,
                n_redundant=2,
                n_classes=3,
                n_clusters_per_class=2,
                random_state=self.random_state
            )
            self.feature_names = [f'feature_{i}' for i in range(10)]
            self.target_names = ['Class_0', 'Class_1', 'Class_2']
        
        print(f"✓ Loaded {dataset_type} dataset")
        print(f"  Shape: {X.shape} ({X.shape[0]} samples, {X.shape[1]} features)")
        print(f"  Classes: {len(np.unique(y))} ({self.target_names})")
        print(f"  Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def split_and_scale(self, X, y, test_size=0.2):
        """
        Split data and apply feature scaling.
        
        Critical: Scaler is fit ONLY on training data to prevent data leakage.
        This is a common bug that inflates accuracy estimates in production.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing (default: 0.2)
        """
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features using training data statistics only
        # Instagram scales engagement metrics (likes, comments) this way
        # before KNN-based content recommendations
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n✓ Data split and scaled")
        print(f"  Training: {self.X_train.shape[0]} samples")
        print(f"  Testing: {self.X_test.shape[0]} samples")
        print(f"  Feature scale - Mean: {self.scaler.mean_[:3].round(2)}...")
        print(f"  Feature scale - Std: {self.scaler.scale_[:3].round(2)}...")
    
    def train_baseline(self):
        """
        Train baseline KNN model with default hyperparameters.
        
        This establishes a performance baseline before optimization.
        Pinterest's visual search team starts with k=5 before tuning.
        """
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate on training and test sets
        train_acc = self.model.score(self.X_train_scaled, self.y_train)
        test_acc = self.model.score(self.X_test_scaled, self.y_test)
        
        print(f"\n✓ Baseline model trained (k={self.n_neighbors})")
        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")
        print(f"  Overfitting gap: {(train_acc - test_acc):.4f}")
        
        return train_acc, test_acc
    
    def optimize_hyperparameters(self, cv_folds=5):
        """
        Perform grid search to find optimal hyperparameters.
        
        Tests multiple combinations of:
        - n_neighbors: Number of neighbors to consider
        - weights: How to weight neighbor votes
        - metric: Distance calculation method
        
        Google's spam detection runs similar searches across thousands
        of configurations to maximize accuracy.
        
        Args:
            cv_folds: Number of cross-validation folds (default: 5)
        """
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        print(f"\n⚙ Running grid search...")
        print(f"  Testing {len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['metric'])} configurations")
        print(f"  With {cv_folds}-fold cross-validation")
        
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"\n✓ Optimization complete")
        print(f"  Best parameters: {self.best_params}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        print(f"  Test accuracy: {self.model.score(self.X_test_scaled, self.y_test):.4f}")
        
        return self.best_params, grid_search.best_score_
    
    def evaluate_model(self, plot=True):
        """
        Comprehensive model evaluation with multiple metrics.
        
        Generates:
        - Confusion matrix (which classes are confused)
        - Classification report (precision, recall, F1 per class)
        - Cross-validation scores (model stability)
        
        Tesla's autopilot monitors these metrics for pedestrian/vehicle
        classification - high recall for pedestrians is critical.
        """
        # Predictions
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        # Handle case where target_names might not be set (e.g., when using custom data)
        target_names = getattr(self, 'target_names', None)
        print(classification_report(
            self.y_test, y_pred,
            target_names=target_names,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nCONFUSION MATRIX")
        print(cm)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, self.X_train_scaled, self.y_train,
            cv=5, scoring='accuracy'
        )
        print(f"\nCROSS-VALIDATION SCORES")
        print(f"  Mean: {cv_scores.mean():.4f}")
        print(f"  Std: {cv_scores.std():.4f}")
        print(f"  Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
        
        if plot:
            self._plot_confusion_matrix(cm)
            self._plot_decision_boundary()
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        # Handle case where target_names might not be set
        target_names = getattr(self, 'target_names', None)
        if target_names is None:
            n_classes = cm.shape[0]
            target_names = [f'Class_{i}' for i in range(n_classes)]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names
        )
        plt.title('Confusion Matrix - KNN Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        print(f"\n✓ Confusion matrix saved to confusion_matrix.png")
        plt.close()
    
    def _plot_decision_boundary(self):
        """
        Plot decision boundary for first two features.
        
        Helps visualize how KNN creates decision regions by voting.
        Spotify uses similar visualizations for music feature space.
        """
        if self.X_train_scaled.shape[1] < 2:
            return
        
        # Use only first 2 features for visualization
        X_train_2d = self.X_train_scaled[:, :2]
        X_test_2d = self.X_test_scaled[:, :2]
        
        # Train model on 2D data
        model_2d = KNeighborsClassifier(
            n_neighbors=self.model.n_neighbors,
            weights=self.model.weights,
            metric=self.model.metric
        )
        model_2d.fit(X_train_2d, self.y_train)
        
        # Create mesh grid
        h = 0.02
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Predict on mesh
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
                   c=self.y_train, cmap='viridis',
                   edgecolors='k', s=50, alpha=0.7, label='Training')
        plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                   c=self.y_test, cmap='viridis',
                   edgecolors='red', s=100, marker='^',
                   alpha=0.9, label='Test')
        
        plt.xlabel(f'{self.feature_names[0]} (scaled)')
        plt.ylabel(f'{self.feature_names[1]} (scaled)')
        plt.title(f'KNN Decision Boundary (k={self.model.n_neighbors})')
        plt.legend()
        plt.tight_layout()
        plt.savefig('decision_boundary.png', dpi=150)
        print(f"✓ Decision boundary saved to decision_boundary.png")
        plt.close()
    
    def predict_new_samples(self, X_new):
        """
        Make predictions on new data.
        
        This is the production inference path - what happens when
        Airbnb receives a new listing search query.
        
        Args:
            X_new: New samples to classify (unscaled)
        
        Returns:
            predictions: Class labels
            probabilities: Confidence scores per class
        """
        X_new_scaled = self.scaler.transform(X_new)
        predictions = self.model.predict(X_new_scaled)
        probabilities = self.model.predict_proba(X_new_scaled)
        
        return predictions, probabilities


def run_complete_pipeline():
    """Run the complete KNN pipeline demonstration."""
    print("\n" + "="*60)
    print("Day 63: KNN with Scikit-learn - Complete Pipeline")
    print("="*60)
    
    # Initialize pipeline
    pipeline = KNNPipeline(n_neighbors=5)
    
    # Load and prepare data
    print("\n[1/6] Loading dataset...")
    X, y = pipeline.load_data(dataset_type='iris')
    
    # Split and scale
    print("\n[2/6] Splitting and scaling data...")
    pipeline.split_and_scale(X, y)
    
    # Train baseline
    print("\n[3/6] Training baseline model...")
    pipeline.train_baseline()
    
    # Optimize hyperparameters
    print("\n[4/6] Optimizing hyperparameters...")
    pipeline.optimize_hyperparameters()
    
    # Evaluate
    print("\n[5/6] Evaluating model...")
    metrics = pipeline.evaluate_model(plot=True)
    
    # Demo predictions
    print("\n[6/6] Testing predictions on new samples...")
    # Create sample from test set
    sample_idx = np.random.choice(len(pipeline.X_test), 3, replace=False)
    X_new = pipeline.X_test[sample_idx]
    y_true = pipeline.y_test[sample_idx]
    
    predictions, probabilities = pipeline.predict_new_samples(X_new)
    
    print("\nSample Predictions:")
    for i in range(len(predictions)):
        pred_class = pipeline.target_names[predictions[i]]
        true_class = pipeline.target_names[y_true[i]]
        confidence = probabilities[i][predictions[i]]
        print(f"  Sample {i+1}: Predicted={pred_class}, True={true_class}, Confidence={confidence:.2%}")
    
    print("\n" + "="*60)
    print("Pipeline complete! Check the generated visualizations.")
    print("="*60)


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description='Day 63: KNN with Scikit-learn')
    parser.add_argument(
        '--mode',
        choices=['explore', 'scale', 'train', 'optimize', 'evaluate', 'all'],
        default='all',
        help='Which part of the pipeline to run'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of neighbors for baseline model'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        run_complete_pipeline()
    else:
        pipeline = KNNPipeline(n_neighbors=args.k)
        X, y = pipeline.load_data()
        pipeline.split_and_scale(X, y)
        
        if args.mode == 'explore':
            print("Dataset exploration complete. Check output above.")
        elif args.mode == 'scale':
            print("Feature scaling complete. Check output above.")
        elif args.mode == 'train':
            pipeline.train_baseline()
        elif args.mode == 'optimize':
            pipeline.optimize_hyperparameters()
        elif args.mode == 'evaluate':
            pipeline.train_baseline()
            pipeline.evaluate_model()


if __name__ == '__main__':
    main()
