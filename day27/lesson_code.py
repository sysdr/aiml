"""
Day 27: Measures of Spread - Variance and Standard Deviation
Understanding data variability for AI/ML applications
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


class DataQualityChecker:
    """Analyzes data spread to determine ML readiness"""
    
    def __init__(self, data, feature_name="Feature"):
        self.data = np.array(data)
        self.feature_name = feature_name
        self.mean = np.mean(data)
        self.variance = np.var(data, ddof=1)  # Sample variance (n-1)
        self.std = np.std(data, ddof=1)  # Sample standard deviation
        self.population_var = np.var(data, ddof=0)  # Population variance (n)
        self.population_std = np.std(data, ddof=0)
    
    def detect_outliers(self, threshold=3):
        """Find data points beyond threshold standard deviations"""
        z_scores = (self.data - self.mean) / self.std
        outlier_mask = np.abs(z_scores) > threshold
        outliers = self.data[outlier_mask]
        outlier_indices = np.where(outlier_mask)[0]
        return outliers, len(outliers), outlier_indices
    
    def coefficient_of_variation(self):
        """Relative variability: (std/mean) * 100%"""
        if self.mean == 0:
            return float('inf')
        return (self.std / self.mean) * 100
    
    def calculate_iqr(self):
        """Interquartile Range: another measure of spread"""
        q75, q25 = np.percentile(self.data, [75, 25])
        iqr = q75 - q25
        return iqr, q25, q75
    
    def quality_report(self):
        """Comprehensive data quality analysis"""
        outliers, outlier_count, outlier_idx = self.detect_outliers()
        cv = self.coefficient_of_variation()
        iqr, q25, q75 = self.calculate_iqr()
        
        print(f"\n{'='*60}")
        print(f"📊 DATA QUALITY REPORT: {self.feature_name}")
        print(f"{'='*60}")
        print(f"Sample Size: {len(self.data)}")
        print(f"Range: [{self.data.min():.2f}, {self.data.max():.2f}]")
        print(f"\nCentral Tendency:")
        print(f"  Mean: {self.mean:.2f}")
        print(f"  Median: {np.median(self.data):.2f}")
        print(f"\nMeasures of Spread:")
        print(f"  Sample Variance: {self.variance:.2f}")
        print(f"  Sample Std Dev: {self.std:.2f}")
        print(f"  Population Variance: {self.population_var:.2f}")
        print(f"  Population Std Dev: {self.population_std:.2f}")
        print(f"  IQR: {iqr:.2f} (Q1={q25:.2f}, Q3={q75:.2f})")
        print(f"  Coefficient of Variation: {cv:.2f}%")
        print(f"\nOutlier Detection (3σ rule):")
        print(f"  Outliers Found: {outlier_count} ({outlier_count/len(self.data)*100:.1f}%)")
        if outlier_count > 0:
            print(f"  Outlier Values: {outliers}")
        
        # Quality assessment
        print(f"\n{'='*60}")
        if cv < 15:
            status = "✅ EXCELLENT"
            advice = "Low variance, consistent data. Great for model training!"
        elif cv < 30:
            status = "⚠️  MODERATE"
            advice = "Some spread present. Consider feature scaling."
        else:
            status = "❌ HIGH VARIANCE"
            advice = "High variability detected. Normalization or log transform recommended."
        
        print(f"ML Readiness: {status}")
        print(f"Recommendation: {advice}")
        print(f"{'='*60}\n")
        
        return {
            'mean': self.mean,
            'std': self.std,
            'variance': self.variance,
            'cv': cv,
            'outliers': outlier_count
        }
    
    def visualize(self):
        """Create comprehensive visualization of data spread"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Data Spread Analysis: {self.feature_name}', fontsize=16, fontweight='bold')
        
        # 1. Histogram with normal curve
        ax1 = axes[0, 0]
        ax1.hist(self.data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Overlay normal distribution
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, self.mean, self.std)
        ax1.plot(x, p, 'r-', linewidth=2, label='Normal Distribution')
        ax1.axvline(self.mean, color='green', linestyle='--', linewidth=2, label=f'Mean: {self.mean:.2f}')
        ax1.axvline(self.mean - self.std, color='orange', linestyle='--', linewidth=1.5, label=f'±1σ')
        ax1.axvline(self.mean + self.std, color='orange', linestyle='--', linewidth=1.5)
        ax1.set_title('Distribution with Standard Deviation')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax2 = axes[0, 1]
        bp = ax2.boxplot(self.data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax2.set_title('Box Plot (IQR-based Spread)')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {self.mean:.2f}\nMedian: {np.median(self.data):.2f}\nStd: {self.std:.2f}'
        ax2.text(1.15, np.median(self.data), stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Scatter plot with outliers highlighted
        ax3 = axes[1, 0]
        outliers, outlier_count, outlier_idx = self.detect_outliers()
        colors = ['red' if i in outlier_idx else 'blue' for i in range(len(self.data))]
        ax3.scatter(range(len(self.data)), self.data, c=colors, alpha=0.6)
        ax3.axhline(self.mean, color='green', linestyle='--', linewidth=2, label='Mean')
        ax3.axhline(self.mean + 3*self.std, color='red', linestyle=':', linewidth=1.5, label='3σ boundary')
        ax3.axhline(self.mean - 3*self.std, color='red', linestyle=':', linewidth=1.5)
        ax3.set_title(f'Outlier Detection ({outlier_count} outliers found)')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Variance decomposition
        ax4 = axes[1, 1]
        deviations = (self.data - self.mean) ** 2
        sorted_idx = np.argsort(deviations)[::-1][:20]  # Top 20 contributors
        ax4.bar(range(len(sorted_idx)), deviations[sorted_idx], color='coral', alpha=0.7)
        ax4.set_title('Top 20 Variance Contributors')
        ax4.set_xlabel('Data Point')
        ax4.set_ylabel('Squared Deviation')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_spread_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Visualization saved as 'data_spread_analysis.png'")
        plt.close()  # Close to avoid display issues in headless mode


def compare_datasets():
    """Compare variance across different datasets"""
    print("\n" + "="*60)
    print("🔬 EXPERIMENT: Comparing Different Data Spreads")
    print("="*60)
    
    # Create three datasets with different spreads
    np.random.seed(42)
    
    # Low variance: Consistent ML predictions
    consistent_model = np.random.normal(100, 5, 100)
    
    # Medium variance: Typical real-world data
    typical_data = np.random.normal(100, 20, 100)
    
    # High variance: Unstable measurements
    noisy_data = np.random.normal(100, 50, 100)
    
    datasets = {
        'Consistent Model Predictions': consistent_model,
        'Typical User Behavior': typical_data,
        'Noisy Sensor Readings': noisy_data
    }
    
    results = []
    for name, data in datasets.items():
        checker = DataQualityChecker(data, name)
        result = checker.quality_report()
        results.append((name, result))
    
    # Summary comparison
    print("\n" + "="*60)
    print("📈 COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Dataset':<30} {'Mean':<10} {'Std Dev':<10} {'CV%':<10}")
    print("-"*60)
    for name, result in results:
        print(f"{name:<30} {result['mean']:<10.2f} {result['std']:<10.2f} {result['cv']:<10.2f}")


def ml_application_demo():
    """Demonstrate variance in practical ML scenarios"""
    print("\n" + "="*60)
    print("🤖 ML APPLICATION: Feature Scaling Impact")
    print("="*60)
    
    # Simulate features with different scales
    feature_1 = np.random.normal(50, 10, 100)  # Age (moderate variance)
    feature_2 = np.random.normal(50000, 20000, 100)  # Salary (high variance)
    
    print("\nBefore Scaling:")
    print(f"Feature 1 (Age): μ={np.mean(feature_1):.2f}, σ={np.std(feature_1):.2f}")
    print(f"Feature 2 (Salary): μ={np.mean(feature_2):.2f}, σ={np.std(feature_2):.2f}")
    print(f"Variance Ratio: {np.var(feature_2)/np.var(feature_1):.2f}x")
    print("⚠️  Feature 2 will dominate model learning!")
    
    # Standardization (z-score normalization)
    feature_1_scaled = (feature_1 - np.mean(feature_1)) / np.std(feature_1)
    feature_2_scaled = (feature_2 - np.mean(feature_2)) / np.std(feature_2)
    
    print("\nAfter Standardization:")
    print(f"Feature 1: μ={np.mean(feature_1_scaled):.2f}, σ={np.std(feature_1_scaled):.2f}")
    print(f"Feature 2: μ={np.mean(feature_2_scaled):.2f}, σ={np.std(feature_2_scaled):.2f}")
    print("✅ Features now on equal footing for ML!")


def main():
    """Main lesson execution"""
    print("\n" + "="*80)
    print("🎓 DAY 27: MEASURES OF SPREAD - VARIANCE AND STANDARD DEVIATION")
    print("="*80)
    
    # Example 1: Real-world dataset
    print("\n📊 Example 1: Analyzing Model Response Times")
    response_times = np.array([
        102, 98, 105, 101, 99, 103, 97, 104, 100, 102,
        101, 99, 103, 98, 105, 250, 102, 99, 101, 100  # Note: one outlier (250ms)
    ])
    
    checker = DataQualityChecker(response_times, "API Response Times (ms)")
    checker.quality_report()
    checker.visualize()
    
    # Example 2: Compare different spreads
    compare_datasets()
    
    # Example 3: ML application
    ml_application_demo()
    
    print("\n" + "="*80)
    print("✅ Lesson Complete!")
    print("Key Takeaways:")
    print("  1. Variance measures average squared deviation from mean")
    print("  2. Standard deviation is sqrt(variance) in original units")
    print("  3. High variance → need feature scaling for ML")
    print("  4. 3σ rule: 99.7% of data within 3 standard deviations")
    print("  5. Always use sample variance (n-1) for training data")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
