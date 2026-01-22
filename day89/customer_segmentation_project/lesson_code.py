"""
Day 89: Customer Segmentation System
Production-grade implementation for behavioral clustering and segment profiling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CustomerSegmentationEngine:
    """
    Production customer segmentation system using K-means clustering.
    Handles feature engineering, optimal cluster selection, and segment profiling.
    
    Similar to systems used by Netflix, Spotify, Amazon for user segmentation.
    """
    
    def __init__(self, min_clusters: int = 2, max_clusters: int = 10):
        """
        Initialize segmentation engine.
        
        Args:
            min_clusters: Minimum number of segments to evaluate
            max_clusters: Maximum number of segments to evaluate
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
        self.model = None
        self.optimal_k = None
        self.feature_names = None
        self.segment_profiles = None
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw customer data.
        Creates composite metrics capturing behavioral patterns.
        
        Args:
            df: DataFrame with customer behavior data
            
        Returns:
            DataFrame with engineered features
        """
        engineered = df.copy()
        
        # Value score: monetary value per transaction
        if 'total_spend' in df.columns and 'purchase_frequency' in df.columns:
            engineered['value_score'] = (
                df['total_spend'] / (df['purchase_frequency'] + 1)
            )
        
        # Engagement level: interaction between frequency and recency
        if 'purchase_frequency' in df.columns and 'days_since_last_purchase' in df.columns:
            # Lower recency (more recent) with higher frequency = higher engagement
            engineered['engagement_level'] = (
                df['purchase_frequency'] / (df['days_since_last_purchase'] + 1)
            )
        
        # Customer lifetime value proxy
        if 'total_spend' in df.columns and 'days_since_last_purchase' in df.columns:
            engineered['ltv_proxy'] = (
                df['total_spend'] * np.exp(-df['days_since_last_purchase'] / 365)
            )
        
        return engineered
    
    def find_optimal_clusters(self, X: np.ndarray) -> Tuple[int, Dict]:
        """
        Determine optimal number of clusters using Elbow Method and Silhouette Analysis.
        
        Args:
            X: Scaled feature matrix
            
        Returns:
            Tuple of (optimal_k, evaluation_metrics)
        """
        inertias = []
        silhouette_scores = []
        K_range = range(self.min_clusters, self.max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score for k > 1 and when we have multiple clusters
            if k > 1:
                # Check if we have at least 2 distinct clusters
                n_unique_labels = len(np.unique(labels))
                if n_unique_labels > 1:
                    try:
                        score = silhouette_score(X, labels)
                        silhouette_scores.append(score)
                    except ValueError:
                        # Fallback if silhouette score fails
                        silhouette_scores.append(0)
                else:
                    silhouette_scores.append(0)
            else:
                silhouette_scores.append(0)
        
        # Find elbow point using second derivative
        inertia_diffs = np.diff(inertias)
        if len(inertia_diffs) > 1:
            inertia_diffs2 = np.diff(inertia_diffs)
            if len(inertia_diffs2) > 0:
                elbow_point = np.argmax(inertia_diffs2) + self.min_clusters
            else:
                # Fallback: use first difference
                elbow_point = np.argmax(inertia_diffs) + self.min_clusters
        else:
            # Fallback: use minimum clusters if we can't compute elbow
            elbow_point = self.min_clusters
        
        # Find best silhouette score
        if len(silhouette_scores) > 0:
            best_silhouette_k = np.argmax(silhouette_scores) + self.min_clusters
        else:
            best_silhouette_k = self.min_clusters
        
        # Optimal k is the average, bounded by valid range
        optimal_k = int(np.mean([elbow_point, best_silhouette_k]))
        optimal_k = max(self.min_clusters, min(optimal_k, self.max_clusters))
        
        metrics = {
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'elbow_point': elbow_point,
            'best_silhouette_k': best_silhouette_k,
            'K_range': list(K_range)
        }
        
        return optimal_k, metrics
    
    def fit(self, df: pd.DataFrame, n_clusters: Optional[int] = None) -> 'CustomerSegmentationEngine':
        """
        Fit segmentation model on customer data.
        
        Args:
            df: DataFrame with customer features
            n_clusters: Optional fixed number of clusters (overrides optimal search)
            
        Returns:
            Self for method chaining
        """
        # Engineer features
        df_engineered = self.engineer_features(df)
        
        # Store feature names
        self.feature_names = df_engineered.columns.tolist()
        
        # Handle missing values
        df_engineered = df_engineered.fillna(df_engineered.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(df_engineered)
        
        # Find optimal clusters if not specified
        if n_clusters is None:
            self.optimal_k, self.metrics = self.find_optimal_clusters(X_scaled)
        else:
            self.optimal_k = n_clusters
            self.metrics = {}
        
        # Train final model
        self.model = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        self.model.fit(X_scaled)
        
        # Generate segment profiles
        self.segment_profiles = self._generate_segment_profiles(df_engineered, self.model.labels_)
        
        return self
    
    def _generate_segment_profiles(self, df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """
        Generate detailed profiles for each segment.
        
        Args:
            df: Feature DataFrame
            labels: Cluster assignments
            
        Returns:
            DataFrame with segment statistics
        """
        df_with_labels = df.copy()
        df_with_labels['segment'] = labels
        
        # Calculate statistics per segment
        profiles = df_with_labels.groupby('segment').agg(['mean', 'std', 'median', 'min', 'max'])
        
        # Add segment sizes
        segment_sizes = df_with_labels['segment'].value_counts().sort_index()
        profiles['size'] = segment_sizes.values
        
        return profiles
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign segments to new customers.
        
        Args:
            df: DataFrame with customer features
            
        Returns:
            Tuple of (segment_labels, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Engineer features
        df_engineered = self.engineer_features(df)
        
        # Ensure same features as training
        missing_features = set(self.feature_names) - set(df_engineered.columns)
        if missing_features:
            for feat in missing_features:
                df_engineered[feat] = 0
        
        df_engineered = df_engineered[self.feature_names]
        
        # Handle missing values
        df_engineered = df_engineered.fillna(df_engineered.median())
        
        # Scale features
        X_scaled = self.scaler.transform(df_engineered)
        
        # Predict segments
        labels = self.model.predict(X_scaled)
        
        # Calculate confidence scores (inverse distance to cluster center)
        distances = self.model.transform(X_scaled)
        min_distances = np.min(distances, axis=1)
        max_distance = np.max(min_distances)
        confidence_scores = 1 - (min_distances / max_distance)
        
        return labels, confidence_scores
    
    def get_segment_characteristics(self, segment_id: int) -> Dict:
        """
        Get detailed characteristics for a specific segment.
        
        Args:
            segment_id: Segment number
            
        Returns:
            Dictionary with segment characteristics
        """
        if self.segment_profiles is None:
            raise ValueError("No segment profiles available. Train model first.")
        
        if segment_id not in self.segment_profiles.index:
            raise ValueError(f"Segment {segment_id} not found.")
        
        segment_data = self.segment_profiles.loc[segment_id]
        
        # Handle size extraction - could be Series or scalar
        size_value = segment_data['size']
        if isinstance(size_value, pd.Series):
            size_value = size_value.iloc[0]
        
        characteristics = {
            'segment_id': segment_id,
            'size': int(size_value),
            'features': {}
        }
        
        # Extract feature statistics
        for feature in self.feature_names:
            if feature in segment_data.index.get_level_values(0):
                characteristics['features'][feature] = {
                    'mean': float(segment_data[(feature, 'mean')]),
                    'median': float(segment_data[(feature, 'median')]),
                    'std': float(segment_data[(feature, 'std')])
                }
        
        return characteristics
    
    def visualize_segments(self, df: pd.DataFrame, feature_x: str, feature_y: str):
        """
        Visualize segments in 2D feature space.
        
        Args:
            df: DataFrame with customer features
            feature_x: Feature for x-axis
            feature_y: Feature for y-axis
        """
        df_engineered = self.engineer_features(df)
        labels, _ = self.predict(df)
        
        plt.figure(figsize=(12, 8))
        
        # Plot each segment
        for segment_id in range(self.optimal_k):
            mask = labels == segment_id
            plt.scatter(
                df_engineered.loc[mask, feature_x],
                df_engineered.loc[mask, feature_y],
                label=f'Segment {segment_id}',
                alpha=0.6,
                s=100
            )
        
        # Plot cluster centers (transform back to original space)
        centers_scaled = self.model.cluster_centers_
        feature_x_idx = self.feature_names.index(feature_x)
        feature_y_idx = self.feature_names.index(feature_y)
        
        plt.scatter(
            centers_scaled[:, feature_x_idx],
            centers_scaled[:, feature_y_idx],
            c='black',
            marker='X',
            s=300,
            edgecolors='white',
            linewidths=2,
            label='Centroids',
            zorder=10
        )
        
        plt.xlabel(feature_x.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(feature_y.replace('_', ' ').title(), fontsize=12)
        plt.title('Customer Segments Visualization', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('segment_visualization.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'segment_visualization.png'")
    
    def save_model(self, filepath: str):
        """Save trained model and preprocessors."""
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'optimal_k': self.optimal_k,
            'feature_names': self.feature_names,
            'segment_profiles': self.segment_profiles
        }
        joblib.dump(model_artifacts, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'CustomerSegmentationEngine':
        """Load trained model and preprocessors."""
        artifacts = joblib.load(filepath)
        
        engine = cls()
        engine.model = artifacts['model']
        engine.scaler = artifacts['scaler']
        engine.optimal_k = artifacts['optimal_k']
        engine.feature_names = artifacts['feature_names']
        engine.segment_profiles = artifacts['segment_profiles']
        
        return engine


def generate_sample_data(n_customers: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic customer behavior data for demonstration.
    
    Args:
        n_customers: Number of customers to generate
        
    Returns:
        DataFrame with customer features
    """
    np.random.seed(42)
    
    # Create distinct customer types
    # Type 1: High-value engaged (20%)
    n_high_value = int(n_customers * 0.2)
    high_value = pd.DataFrame({
        'total_spend': np.random.normal(5000, 1000, n_high_value),
        'purchase_frequency': np.random.normal(25, 5, n_high_value),
        'days_since_last_purchase': np.random.normal(15, 10, n_high_value)
    })
    
    # Type 2: Medium-value regular (40%)
    n_medium = int(n_customers * 0.4)
    medium_value = pd.DataFrame({
        'total_spend': np.random.normal(2000, 500, n_medium),
        'purchase_frequency': np.random.normal(12, 3, n_medium),
        'days_since_last_purchase': np.random.normal(30, 15, n_medium)
    })
    
    # Type 3: Low-value occasional (25%)
    n_low = int(n_customers * 0.25)
    low_value = pd.DataFrame({
        'total_spend': np.random.normal(500, 200, n_low),
        'purchase_frequency': np.random.normal(3, 1, n_low),
        'days_since_last_purchase': np.random.normal(90, 30, n_low)
    })
    
    # Type 4: Dormant customers (15%)
    n_dormant = n_customers - n_high_value - n_medium - n_low
    dormant = pd.DataFrame({
        'total_spend': np.random.normal(1000, 300, n_dormant),
        'purchase_frequency': np.random.normal(5, 2, n_dormant),
        'days_since_last_purchase': np.random.normal(180, 60, n_dormant)
    })
    
    # Combine all customer types
    df = pd.concat([high_value, medium_value, low_value, dormant], ignore_index=True)
    
    # Ensure non-negative values
    df = df.clip(lower=0)
    
    return df


def main():
    """Demonstration of customer segmentation system."""
    print("=" * 70)
    print("Day 89: Customer Segmentation System Demonstration")
    print("=" * 70)
    print()
    
    # Generate sample data
    print("1. Generating sample customer data...")
    df = generate_sample_data(n_customers=1000)
    print(f"   Created dataset with {len(df)} customers")
    print(f"   Features: {', '.join(df.columns)}")
    print()
    
    # Initialize and train segmentation engine
    print("2. Training segmentation engine...")
    engine = CustomerSegmentationEngine(min_clusters=2, max_clusters=8)
    engine.fit(df)
    print(f"   Optimal number of segments: {engine.optimal_k}")
    print()
    
    # Display segment profiles
    print("3. Segment Profiles:")
    print("-" * 70)
    for segment_id in range(engine.optimal_k):
        characteristics = engine.get_segment_characteristics(segment_id)
        print(f"\n   Segment {segment_id} (n={characteristics['size']} customers):")
        
        for feature, stats in characteristics['features'].items():
            if feature in ['total_spend', 'purchase_frequency', 'days_since_last_purchase']:
                print(f"     • {feature.replace('_', ' ').title()}: "
                      f"${stats['mean']:.2f} (median: ${stats['median']:.2f})")
    print()
    
    # Predict on new customers
    print("4. Predicting segments for new customers...")
    new_customers = pd.DataFrame({
        'total_spend': [6000, 800, 2500],
        'purchase_frequency': [30, 2, 15],
        'days_since_last_purchase': [5, 120, 25]
    })
    
    labels, confidence = engine.predict(new_customers)
    
    for i, (label, conf) in enumerate(zip(labels, confidence)):
        print(f"   Customer {i+1}: Segment {label} (confidence: {conf:.2%})")
    print()
    
    # Save model
    print("5. Saving trained model...")
    engine.save_model('customer_segmentation_model.pkl')
    print()
    
    # Visualize segments
    print("6. Generating visualization...")
    engine.visualize_segments(df, 'total_spend', 'purchase_frequency')
    print()
    
    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
