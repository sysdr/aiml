"""
Day 85: Introduction to Unsupervised Learning
Customer Segmentation System

This implementation demonstrates production-grade unsupervised learning
techniques used by companies like Amazon, Netflix, and Spotify for 
customer/user segmentation without labeled data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
import argparse
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style for production-quality visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class CustomerSegmentationPipeline:
    """
    Production-grade customer segmentation system using unsupervised learning.
    
    Architecture:
    1. Data Ingestion & Validation
    2. Feature Engineering
    3. Dimensionality Reduction (PCA)
    4. Clustering (K-Means)
    5. Segment Analysis & Profiling
    
    Used by: E-commerce platforms, SaaS companies, retail analytics
    """
    
    def __init__(self, n_components=2, random_state=42):
        """
        Initialize the segmentation pipeline.
        
        Args:
            n_components: Number of PCA components for dimensionality reduction
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.best_kmeans = None
        self.optimal_k = None
        self.data = None
        self.features = None
        self.scaled_features = None
        self.pca_features = None
        self.cluster_labels = None
        
    def generate_sample_data(self, n_samples=1000):
        """
        Generate synthetic customer transaction data.
        
        In production, this would connect to data warehouse (Snowflake, BigQuery)
        or stream from Kafka/Kinesis for real-time segmentation.
        
        Features mirror real e-commerce analytics:
        - Recency: Days since last purchase
        - Frequency: Number of purchases
        - Monetary: Total spend
        - AOV: Average order value
        - Product diversity: Number of unique categories purchased
        """
        np.random.seed(self.random_state)
        
        # Create three natural customer segments
        # Segment 1: VIP customers (10%)
        n_vip = int(n_samples * 0.10)
        vip_recency = np.random.randint(1, 30, n_vip)
        vip_frequency = np.random.randint(20, 50, n_vip)
        vip_monetary = np.random.uniform(5000, 20000, n_vip)
        vip_aov = vip_monetary / vip_frequency
        vip_diversity = np.random.randint(8, 15, n_vip)
        
        # Segment 2: Regular customers (60%)
        n_regular = int(n_samples * 0.60)
        regular_recency = np.random.randint(1, 90, n_regular)
        regular_frequency = np.random.randint(5, 20, n_regular)
        regular_monetary = np.random.uniform(500, 5000, n_regular)
        regular_aov = regular_monetary / regular_frequency
        regular_diversity = np.random.randint(3, 8, n_regular)
        
        # Segment 3: At-risk/Churning customers (30%)
        n_churn = n_samples - n_vip - n_regular
        churn_recency = np.random.randint(90, 365, n_churn)
        churn_frequency = np.random.randint(1, 5, n_churn)
        churn_monetary = np.random.uniform(50, 500, n_churn)
        churn_aov = churn_monetary / churn_frequency
        churn_diversity = np.random.randint(1, 3, n_churn)
        
        # Combine all segments
        recency = np.concatenate([vip_recency, regular_recency, churn_recency])
        frequency = np.concatenate([vip_frequency, regular_frequency, churn_frequency])
        monetary = np.concatenate([vip_monetary, regular_monetary, churn_monetary])
        aov = np.concatenate([vip_aov, regular_aov, churn_aov])
        diversity = np.concatenate([vip_diversity, regular_diversity, churn_diversity])
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'customer_id': range(1, n_samples + 1),
            'recency_days': recency,
            'purchase_frequency': frequency,
            'total_monetary': monetary,
            'avg_order_value': aov,
            'product_diversity': diversity
        })
        
        # Add engagement score (derived feature)
        self.data['engagement_score'] = (
            (365 - self.data['recency_days']) / 365 * 0.3 +
            np.log1p(self.data['purchase_frequency']) / np.log1p(50) * 0.3 +
            np.log1p(self.data['total_monetary']) / np.log1p(20000) * 0.4
        )
        
        return self.data
    
    def exploratory_analysis(self, save_plots=True):
        """
        Perform exploratory data analysis to understand data distributions.
        
        Production systems use this for:
        - Data quality monitoring
        - Drift detection
        - Feature importance assessment
        """
        print("=" * 70)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 70)
        
        # Summary statistics
        print("\nDataset Shape:", self.data.shape)
        print("\nSummary Statistics:")
        print(self.data.describe())
        
        # Check for missing values
        missing = self.data.isnull().sum()
        if missing.any():
            print("\nMissing Values:")
            print(missing[missing > 0])
        else:
            print("\n✓ No missing values detected")
        
        if save_plots:
            # Create visualization directory
            os.makedirs('visualizations', exist_ok=True)
            
            # Distribution plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
            
            features = ['recency_days', 'purchase_frequency', 'total_monetary', 
                       'avg_order_value', 'product_diversity', 'engagement_score']
            
            for idx, feature in enumerate(features):
                ax = axes[idx // 3, idx % 3]
                self.data[feature].hist(bins=50, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f'{feature.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('visualizations/01_feature_distributions.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: visualizations/01_feature_distributions.png")
            plt.close()
            
            # Correlation heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.data[features].corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=1)
            plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('visualizations/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: visualizations/02_correlation_heatmap.png")
            plt.close()
            
            # Pairplot for key features (subset for clarity)
            key_features = ['purchase_frequency', 'total_monetary', 'recency_days']
            sns.pairplot(self.data[key_features], diag_kind='hist', 
                        plot_kws={'alpha': 0.6}, height=3)
            plt.savefig('visualizations/03_feature_relationships.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: visualizations/03_feature_relationships.png")
            plt.close()
    
    def preprocess_features(self):
        """
        Feature scaling for unsupervised learning.
        
        Critical for distance-based algorithms (K-Means, hierarchical clustering).
        Without scaling, features with larger ranges dominate distance calculations.
        
        Production note: Scalers are persisted for consistent inference-time preprocessing.
        """
        print("\n" + "=" * 70)
        print("FEATURE PREPROCESSING")
        print("=" * 70)
        
        # Select features for clustering (exclude customer_id)
        feature_columns = ['recency_days', 'purchase_frequency', 'total_monetary',
                          'avg_order_value', 'product_diversity', 'engagement_score']
        
        self.features = self.data[feature_columns].copy()
        
        # Standardize features (mean=0, std=1)
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        print(f"\n✓ Scaled {len(feature_columns)} features")
        print(f"  Original feature ranges: {self.features.min().min():.2f} to {self.features.max().max():.2f}")
        print(f"  Scaled feature ranges: {self.scaled_features.min():.2f} to {self.scaled_features.max():.2f}")
        
        return self.scaled_features
    
    def dimensionality_reduction(self, save_plots=True):
        """
        Apply PCA for dimensionality reduction and visualization.
        
        Production use cases:
        - Reduce computational complexity for large-scale clustering
        - Noise reduction (minor components often capture noise)
        - Visualization of high-dimensional customer spaces
        
        Netflix: Reduces thousands of viewing features to ~50 latent factors
        Spotify: Compresses audio features for real-time playlist generation
        """
        print("\n" + "=" * 70)
        print("DIMENSIONALITY REDUCTION (PCA)")
        print("=" * 70)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_features = self.pca.fit_transform(self.scaled_features)
        
        # Calculate explained variance
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"\n✓ Reduced from {self.scaled_features.shape[1]} to {self.n_components} dimensions")
        print(f"\nExplained Variance Ratio:")
        for i, var in enumerate(explained_variance):
            print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        print(f"\nCumulative Variance: {cumulative_variance[-1]:.4f} ({cumulative_variance[-1]*100:.2f}%)")
        
        # Component loadings (feature importance)
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=self.features.columns
        )
        print("\nPrincipal Component Loadings:")
        print(loadings.round(3))
        
        if save_plots:
            # Scree plot
            plt.figure(figsize=(10, 6))
            components = range(1, len(explained_variance) + 1)
            plt.bar(components, explained_variance, alpha=0.7, color='steelblue', 
                   label='Individual Variance')
            plt.plot(components, cumulative_variance, marker='o', color='darkred', 
                    linewidth=2, label='Cumulative Variance')
            plt.xlabel('Principal Component', fontweight='bold')
            plt.ylabel('Explained Variance Ratio', fontweight='bold')
            plt.title('PCA Explained Variance (Scree Plot)', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/04_pca_variance.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: visualizations/04_pca_variance.png")
            plt.close()
            
            # 2D PCA visualization
            plt.figure(figsize=(10, 8))
            plt.scatter(self.pca_features[:, 0], self.pca_features[:, 1], 
                       alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.5)
            plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', fontweight='bold')
            plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', fontweight='bold')
            plt.title('Customer Space (PCA Projection)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/05_pca_projection.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: visualizations/05_pca_projection.png")
            plt.close()
        
        return self.pca_features
    
    def find_optimal_clusters(self, max_k=10, save_plots=True):
        """
        Determine optimal number of clusters using elbow method and silhouette analysis.
        
        Production strategy:
        - Test range of K values
        - Evaluate using multiple metrics (inertia, silhouette, business KPIs)
        - Consider interpretability (3-5 segments often most actionable)
        
        Real-world: Marketing teams prefer 3-5 clear segments over 10+ complex ones
        """
        print("\n" + "=" * 70)
        print("OPTIMAL CLUSTER DISCOVERY")
        print("=" * 70)
        
        inertias = []
        silhouette_scores_list = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, 
                           n_init=10, max_iter=300)
            labels = kmeans.fit_predict(self.pca_features)
            
            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(self.pca_features, labels)
            silhouette_scores_list.append(silhouette_avg)
            
            print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_avg:.3f}")
        
        # Find optimal K (maximum silhouette score)
        self.optimal_k = k_range[np.argmax(silhouette_scores_list)]
        print(f"\n✓ Optimal K (by silhouette): {self.optimal_k}")
        
        if save_plots:
            # Elbow plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            ax1.plot(k_range, inertias, marker='o', linewidth=2, markersize=8, color='steelblue')
            ax1.set_xlabel('Number of Clusters (K)', fontweight='bold')
            ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontweight='bold')
            ax1.set_title('Elbow Method for Optimal K', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=self.optimal_k, color='red', linestyle='--', 
                       label=f'Optimal K={self.optimal_k}', linewidth=2)
            ax1.legend()
            
            # Silhouette plot
            ax2.plot(k_range, silhouette_scores_list, marker='s', linewidth=2, 
                    markersize=8, color='darkgreen')
            ax2.set_xlabel('Number of Clusters (K)', fontweight='bold')
            ax2.set_ylabel('Average Silhouette Score', fontweight='bold')
            ax2.set_title('Silhouette Analysis for Optimal K', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=self.optimal_k, color='red', linestyle='--', 
                       label=f'Optimal K={self.optimal_k}', linewidth=2)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('visualizations/06_optimal_k_analysis.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: visualizations/06_optimal_k_analysis.png")
            plt.close()
        
        return self.optimal_k
    
    def perform_clustering(self, n_clusters=None, save_plots=True):
        """
        Execute K-Means clustering with optimal parameters.
        
        Production optimizations:
        - K-Means++ initialization (better convergence)
        - Multiple initializations (n_init=10)
        - Parallelization for large datasets
        
        Scale: Google's clustering systems process billions of data points using
        distributed K-Means on MapReduce/Spark infrastructure
        """
        print("\n" + "=" * 70)
        print("K-MEANS CLUSTERING")
        print("=" * 70)
        
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 3
        
        # Fit K-Means
        self.best_kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',  # Smart initialization
            n_init=10,          # Multiple runs with different initializations
            max_iter=300,
            random_state=self.random_state
        )
        
        self.cluster_labels = self.best_kmeans.fit_predict(self.pca_features)
        
        # Add cluster labels to original data
        self.data['cluster'] = self.cluster_labels
        
        # Calculate metrics
        inertia = self.best_kmeans.inertia_
        silhouette_avg = silhouette_score(self.pca_features, self.cluster_labels)
        
        print(f"\n✓ Clustered {len(self.data)} customers into {n_clusters} segments")
        print(f"  Inertia: {inertia:.2f}")
        print(f"  Silhouette Score: {silhouette_avg:.3f}")
        
        # Cluster size distribution
        cluster_sizes = pd.Series(self.cluster_labels).value_counts().sort_index()
        print("\nCluster Sizes:")
        for cluster_id, size in cluster_sizes.items():
            print(f"  Cluster {cluster_id}: {size} customers ({size/len(self.data)*100:.1f}%)")
        
        if save_plots:
            # Cluster visualization
            plt.figure(figsize=(12, 10))
            
            # Get unique clusters for color mapping
            unique_clusters = np.unique(self.cluster_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            
            for cluster_id, color in zip(unique_clusters, colors):
                mask = self.cluster_labels == cluster_id
                plt.scatter(self.pca_features[mask, 0], 
                          self.pca_features[mask, 1],
                          c=[color], label=f'Cluster {cluster_id}',
                          alpha=0.6, edgecolors='black', linewidth=0.5, s=100)
            
            # Plot centroids
            centroids = self.best_kmeans.cluster_centers_
            plt.scatter(centroids[:, 0], centroids[:, 1], 
                       c='red', marker='X', s=300, edgecolors='black', 
                       linewidth=2, label='Centroids')
            
            plt.xlabel('Principal Component 1', fontweight='bold')
            plt.ylabel('Principal Component 2', fontweight='bold')
            plt.title(f'Customer Segmentation ({n_clusters} Clusters)', 
                     fontsize=14, fontweight='bold')
            plt.legend(loc='best', frameon=True, shadow=True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/07_cluster_visualization.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: visualizations/07_cluster_visualization.png")
            plt.close()
        
        return self.cluster_labels
    
    def analyze_segments(self, save_plots=True):
        """
        Extract actionable business insights from discovered clusters.
        
        Production output:
        - Segment profiles for marketing teams
        - Targeting rules for recommendation engines
        - Monitoring dashboards for segment drift
        
        Business value: Personalized campaigns, retention strategies, upsell opportunities
        """
        print("\n" + "=" * 70)
        print("SEGMENT ANALYSIS & PROFILING")
        print("=" * 70)
        
        # Calculate segment statistics
        segment_profiles = self.data.groupby('cluster').agg({
            'recency_days': ['mean', 'median'],
            'purchase_frequency': ['mean', 'median'],
            'total_monetary': ['mean', 'median'],
            'avg_order_value': ['mean', 'median'],
            'product_diversity': ['mean', 'median'],
            'engagement_score': ['mean', 'median']
        }).round(2)
        
        print("\nSegment Profiles:")
        print(segment_profiles)
        
        # Assign business labels based on characteristics
        segment_labels = {}
        for cluster_id in self.data['cluster'].unique():
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            avg_monetary = cluster_data['total_monetary'].mean()
            avg_frequency = cluster_data['purchase_frequency'].mean()
            avg_recency = cluster_data['recency_days'].mean()
            
            if avg_monetary > 5000 and avg_frequency > 15:
                label = "VIP Customers"
            elif avg_recency > 90:
                label = "At-Risk / Churning"
            else:
                label = "Regular Customers"
            
            segment_labels[cluster_id] = label
        
        print("\n" + "=" * 70)
        print("BUSINESS SEGMENT INTERPRETATION")
        print("=" * 70)
        
        for cluster_id, label in segment_labels.items():
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            print(f"\nCluster {cluster_id}: {label}")
            print(f"  Size: {len(cluster_data)} customers ({len(cluster_data)/len(self.data)*100:.1f}%)")
            print(f"  Avg Recency: {cluster_data['recency_days'].mean():.0f} days")
            print(f"  Avg Frequency: {cluster_data['purchase_frequency'].mean():.1f} purchases")
            print(f"  Avg Monetary: ${cluster_data['total_monetary'].mean():.2f}")
            print(f"  Avg Order Value: ${cluster_data['avg_order_value'].mean():.2f}")
            print(f"  Product Diversity: {cluster_data['product_diversity'].mean():.1f} categories")
        
        if save_plots:
            # Radar chart for segment comparison
            features_for_radar = ['recency_days', 'purchase_frequency', 'total_monetary', 
                                 'product_diversity', 'engagement_score']
            
            # Normalize features for radar chart
            normalized_profiles = self.data.groupby('cluster')[features_for_radar].mean()
            for col in features_for_radar:
                max_val = normalized_profiles[col].max()
                if max_val > 0:
                    normalized_profiles[col] = normalized_profiles[col] / max_val
            
            # Create radar chart
            num_vars = len(features_for_radar)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(normalized_profiles)))
            
            for idx, (cluster_id, color) in enumerate(zip(normalized_profiles.index, colors)):
                values = normalized_profiles.loc[cluster_id].tolist()
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}', 
                       color=color)
                ax.fill(angles, values, alpha=0.25, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([f.replace('_', ' ').title() for f in features_for_radar])
            ax.set_ylim(0, 1)
            ax.set_title('Segment Profile Comparison', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig('visualizations/08_segment_radar_chart.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: visualizations/08_segment_radar_chart.png")
            plt.close()
        
        return segment_profiles
    
    def export_results(self, filename='customer_segments.csv'):
        """
        Export segmented customer data for downstream systems.
        
        Production integration:
        - Export to data warehouse for BI dashboards
        - Stream to CRM for personalized campaigns
        - Feed to recommendation engines
        """
        export_data = self.data[['customer_id', 'cluster', 'recency_days', 
                                 'purchase_frequency', 'total_monetary', 
                                 'engagement_score']].copy()
        
        export_data.to_csv(filename, index=False)
        print(f"\n✓ Exported segmentation results to: {filename}")
        
        return filename


def main():
    """Main execution pipeline for unsupervised learning demonstration."""
    
    parser = argparse.ArgumentParser(description='Day 85: Unsupervised Learning Pipeline')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'eda', 'pca', 'cluster'],
                       help='Execution mode: full pipeline or specific stage')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of customer samples to generate')
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='Number of clusters (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("DAY 85: INTRODUCTION TO UNSUPERVISED LEARNING")
    print("Customer Segmentation System")
    print("=" * 70)
    print(f"\nExecution Mode: {args.mode.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = CustomerSegmentationPipeline(n_components=2, random_state=42)
    
    # Generate data
    print("\nGenerating customer transaction data...")
    pipeline.generate_sample_data(n_samples=args.samples)
    
    if args.mode in ['full', 'eda']:
        # Exploratory analysis
        pipeline.exploratory_analysis(save_plots=True)
    
    if args.mode in ['full', 'pca', 'cluster']:
        # Preprocessing
        pipeline.preprocess_features()
        
        # Dimensionality reduction
        pipeline.dimensionality_reduction(save_plots=True)
    
    if args.mode in ['full', 'cluster']:
        # Find optimal clusters
        pipeline.find_optimal_clusters(max_k=10, save_plots=True)
        
        # Perform clustering
        pipeline.perform_clustering(n_clusters=args.n_clusters, save_plots=True)
        
        # Analyze segments
        pipeline.analyze_segments(save_plots=True)
        
        # Export results
        pipeline.export_results()
    
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print("\nGenerated Outputs:")
    print("  📊 Visualizations: visualizations/ directory")
    print("  📄 Segmentation data: customer_segments.csv")
    print("\nNext Steps:")
    print("  1. Review segment profiles for business insights")
    print("  2. Validate segments with domain experts")
    print("  3. Integrate with CRM/marketing automation")
    print("  4. Monitor segment drift over time")
    print("\nDay 86 Preview: K-Means Algorithm Deep Dive")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

