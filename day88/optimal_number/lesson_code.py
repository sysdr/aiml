"""
Day 88: How to Choose the Optimal Number of Clusters
Production-grade cluster evaluation using Elbow, Silhouette, and Gap Statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')


class ClusterEvaluator:
    """
    Comprehensive cluster evaluation using multiple methods.
    """
    
    def __init__(self, k_range=(2, 15), random_state=42):
        self.k_range = range(k_range[0], k_range[1] + 1)
        self.random_state = random_state
        self.results = {
            'elbow': {},
            'silhouette': {},
            'gap': {}
        }
        self.scaler = StandardScaler()
        
    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        print(f"Evaluating clusters from k={min(self.k_range)} to k={max(self.k_range)}...")
        self._compute_elbow_method(X_scaled)
        self._compute_silhouette_scores(X_scaled)
        self._compute_gap_statistic(X_scaled)
        return self
    
    def _compute_elbow_method(self, X):
        wcss_values = []
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            wcss_values.append(kmeans.inertia_)
        self.results['elbow'] = {
            'k_values': list(self.k_range),
            'wcss': wcss_values,
            'optimal_k': self._find_elbow_point(wcss_values)
        }
        
    def _find_elbow_point(self, wcss_values):
        x = np.arange(len(wcss_values))
        y = np.array(wcss_values)
        m = (y[-1] - y[0]) / (x[-1] - x[0])
        b = y[0] - m * x[0]
        distances = np.abs(y - (m * x + b)) / np.sqrt(m**2 + 1)
        elbow_idx = np.argmax(distances)
        return list(self.k_range)[elbow_idx]
    
    def _compute_silhouette_scores(self, X):
        avg_scores = []
        sample_scores = {}
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            avg_score = silhouette_score(X, cluster_labels)
            avg_scores.append(avg_score)
            sample_scores[k] = silhouette_samples(X, cluster_labels)
        self.results['silhouette'] = {
            'k_values': list(self.k_range),
            'avg_scores': avg_scores,
            'sample_scores': sample_scores,
            'optimal_k': list(self.k_range)[np.argmax(avg_scores)]
        }
    
    def _compute_gap_statistic(self, X, n_refs=50):
        gaps = []
        gap_stds = []
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            real_wcss = kmeans.inertia_
            ref_wcss = []
            for _ in range(n_refs):
                random_data = np.random.uniform(low=mins, high=maxs, size=X.shape)
                kmeans_ref = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                kmeans_ref.fit(random_data)
                ref_wcss.append(kmeans_ref.inertia_)
            gap = np.mean(np.log(ref_wcss)) - np.log(real_wcss)
            gap_std = np.std(np.log(ref_wcss))
            gaps.append(gap)
            gap_stds.append(gap_std)
        optimal_k = self._find_optimal_gap_k(gaps, gap_stds)
        self.results['gap'] = {
            'k_values': list(self.k_range),
            'gaps': gaps,
            'gap_stds': gap_stds,
            'optimal_k': optimal_k
        }
    
    def _find_optimal_gap_k(self, gaps, gap_stds):
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - gap_stds[i + 1]:
                return list(self.k_range)[i]
        return list(self.k_range)[np.argmax(gaps)]
    
    def get_recommendations(self):
        recommendations = {
            'elbow_method': self.results['elbow']['optimal_k'],
            'silhouette_analysis': self.results['silhouette']['optimal_k'],
            'gap_statistic': self.results['gap']['optimal_k']
        }
        k_values = list(recommendations.values())
        consensus = max(set(k_values), key=k_values.count)
        recommendations['consensus'] = consensus
        recommendations['agreement'] = (
            '✓ Strong agreement' if len(set(k_values)) == 1 
            else '~ Moderate agreement' if len(set(k_values)) == 2
            else '✗ Methods disagree - use domain knowledge'
        )
        return recommendations
    
    def plot_results(self, figsize=(16, 10)):
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Cluster Evaluation Dashboard', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        k_vals = self.results['elbow']['k_values']
        wcss = self.results['elbow']['wcss']
        optimal_k = self.results['elbow']['optimal_k']
        ax1.plot(k_vals, wcss, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Elbow at k={optimal_k}')
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Within-Cluster Sum of Squares', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        k_vals = self.results['silhouette']['k_values']
        scores = self.results['silhouette']['avg_scores']
        optimal_k = self.results['silhouette']['optimal_k']
        ax2.plot(k_vals, scores, 'go-', linewidth=2, markersize=8)
        ax2.axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Best score at k={optimal_k}')
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Average Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        ax3 = axes[1, 0]
        k_vals = self.results['gap']['k_values']
        gaps = self.results['gap']['gaps']
        gap_stds = self.results['gap']['gap_stds']
        optimal_k = self.results['gap']['optimal_k']
        ax3.errorbar(k_vals, gaps, yerr=gap_stds, fmt='mo-', linewidth=2, markersize=8, capsize=5)
        ax3.axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
        ax3.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax3.set_ylabel('Gap Statistic', fontsize=12)
        ax3.set_title('Gap Statistic', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        recommendations = self.get_recommendations()
        summary_text = f"""
        RECOMMENDATIONS SUMMARY
        {'='*40}
        
        Elbow Method:           k = {recommendations['elbow_method']}
        Silhouette Analysis:    k = {recommendations['silhouette_analysis']}
        Gap Statistic:          k = {recommendations['gap_statistic']}
        
        {'='*40}
        CONSENSUS:              k = {recommendations['consensus']}
        {'='*40}
        
        Agreement Level: {recommendations['agreement']}
        
        PRODUCTION GUIDANCE:
        • If methods agree: Use consensus k
        • If methods disagree by ±1: Test both values
        • If large disagreement: Consider domain knowledge
          and business constraints
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('cluster_evaluation_dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Dashboard saved as 'cluster_evaluation_dashboard.png'")
        plt.show()
        return fig


def generate_sample_data(n_samples=1000, n_features=5, n_clusters=4, random_state=42):
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.5,
        random_state=random_state
    )
    feature_names = [
        'session_duration_hrs',
        'purchase_frequency',
        'avg_transaction_value',
        'support_tickets',
        'days_since_last_visit'
    ]
    df = pd.DataFrame(X, columns=feature_names[:n_features])
    return df, y_true


def main():
    print("=" * 60)
    print("Day 88: How to Choose the Optimal Number of Clusters")
    print("=" * 60)
    print()
    print("1. Generating sample customer behavioral data...")
    X, y_true = generate_sample_data(n_samples=1000, n_features=5, n_clusters=4)
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   True clusters (hidden in real scenarios): {len(np.unique(y_true))}")
    print()
    print("2. Initializing cluster evaluator (k=2 to k=10)...")
    evaluator = ClusterEvaluator(k_range=(2, 10))
    print()
    print("3. Running comprehensive evaluation...")
    print("   - Computing Elbow Method (WCSS)...")
    print("   - Computing Silhouette scores...")
    print("   - Computing Gap Statistics (50 bootstrap samples)...")
    evaluator.fit(X)
    print("   ✓ Evaluation complete!")
    print()
    print("4. Analyzing results...")
    recommendations = evaluator.get_recommendations()
    print()
    print("RECOMMENDATIONS:")
    print(f"  Elbow Method:        k = {recommendations['elbow_method']}")
    print(f"  Silhouette Analysis: k = {recommendations['silhouette_analysis']}")
    print(f"  Gap Statistic:       k = {recommendations['gap_statistic']}")
    print()
    print(f"  CONSENSUS:           k = {recommendations['consensus']}")
    print(f"  Agreement:           {recommendations['agreement']}")
    print()
    print("5. Generating visualization dashboard...")
    evaluator.plot_results()
    print()
    print("=" * 60)
    print("Next Steps:")
    print("  • Examine cluster_evaluation_dashboard.png")
    print("  • Run tests: pytest test_lesson.py -v")
    print("  • Apply to your own data by modifying generate_sample_data()")
    print("=" * 60)


if __name__ == "__main__":
    main()
