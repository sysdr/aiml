"""
Day 36: Exploratory Data Analysis Project
A complete EDA workflow for e-commerce user behavior data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EDAEngine:
    """
    Production-grade EDA engine for systematic data investigation.
    Used by data scientists at companies like Netflix, Uber, and Spotify.
    """
    
    def __init__(self, data: pd.DataFrame, name: str = "Dataset"):
        """
        Initialize EDA engine with a dataset.
        
        Args:
            data: Pandas DataFrame to analyze
            name: Name of the dataset for reporting
        """
        self.data = data.copy()
        self.name = name
        self.report = []
        self.output_dir = Path("eda_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def add_to_report(self, section: str, content: str):
        """Add findings to investigation report"""
        self.report.append(f"\n{'='*60}\n{section}\n{'='*60}\n{content}\n")
    
    def phase1_data_profiling(self):
        """
        Phase 1: Initial data profiling - the vital signs check.
        This is the first thing data scientists do at any company.
        """
        print("\n📊 PHASE 1: DATA PROFILING")
        print("-" * 60)
        
        # Basic shape and info
        n_rows, n_cols = self.data.shape
        memory_mb = self.data.memory_usage(deep=True).sum() / 1024**2
        
        profile = f"""
Dataset: {self.name}
Dimensions: {n_rows:,} rows × {n_cols} columns
Memory Usage: {memory_mb:.2f} MB
        
Column Overview:
{self.data.dtypes.to_string()}

First 3 Rows:
{self.data.head(3).to_string()}

Last 3 Rows:
{self.data.tail(3).to_string()}
        """
        
        print(profile)
        self.add_to_report("DATA PROFILING", profile)
        
        return {
            'n_rows': n_rows,
            'n_cols': n_cols,
            'memory_mb': memory_mb,
            'dtypes': self.data.dtypes.to_dict()
        }
    
    def phase2_data_quality(self):
        """
        Phase 2: Data quality assessment - finding the problems.
        At Uber, this prevents routing failures. At Netflix, this prevents bad recommendations.
        """
        print("\n🔍 PHASE 2: DATA QUALITY ASSESSMENT")
        print("-" * 60)
        
        # Missing values analysis
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data) * 100).round(2)
        
        quality_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percentage': missing_pct,
            'Data_Type': self.data.dtypes
        })
        quality_df = quality_df[quality_df['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        
        quality_report = f"""
Missing Values Summary:
Total Cells: {self.data.shape[0] * self.data.shape[1]:,}
Missing Cells: {self.data.isnull().sum().sum():,}
Overall Completeness: {(1 - self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100:.2f}%

Columns with Missing Data:
{quality_df.to_string() if not quality_df.empty else "No missing values found! ✓"}
        """
        
        print(quality_report)
        self.add_to_report("DATA QUALITY", quality_report)
        
        # Outlier detection for numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numerical_cols:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outlier_count = ((self.data[col] < lower_bound) | 
                                (self.data[col] > upper_bound)).sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        'count': outlier_count,
                        'percentage': (outlier_count / len(self.data) * 100).round(2),
                        'bounds': (lower_bound, upper_bound)
                    }
        
        outlier_report = "\nOutlier Detection (3×IQR method):\n"
        if outliers:
            for col, info in outliers.items():
                outlier_report += f"\n{col}:"
                outlier_report += f"\n  - {info['count']} outliers ({info['percentage']}%)"
                outlier_report += f"\n  - Normal range: [{info['bounds'][0]:.2f}, {info['bounds'][1]:.2f}]"
        else:
            outlier_report += "No significant outliers detected ✓"
        
        print(outlier_report)
        self.add_to_report("OUTLIER ANALYSIS", outlier_report)
        
        return {
            'missing_summary': quality_df.to_dict() if not quality_df.empty else {},
            'outliers': outliers
        }
    
    def phase3_statistical_analysis(self):
        """
        Phase 3: Statistical deep dive - understanding the patterns.
        This is where Netflix learns viewing behavior, Spotify understands listening patterns.
        """
        print("\n📈 PHASE 3: STATISTICAL ANALYSIS")
        print("-" * 60)
        
        # Summary statistics
        stats = self.data.describe(include='all').round(2)
        
        stats_report = f"""
Summary Statistics:
{stats.to_string()}
        """
        
        print(stats_report)
        self.add_to_report("STATISTICAL SUMMARY", stats_report)
        
        # Distribution analysis for key numerical features
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns[:4]
        
        if len(numerical_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{self.name} - Distribution Analysis', 
                        fontsize=16, fontweight='bold')
            axes = axes.ravel()
            
            for idx, col in enumerate(numerical_cols):
                if idx < 4:
                    # Histogram with KDE
                    axes[idx].hist(self.data[col].dropna(), bins=30, 
                                 alpha=0.7, color='steelblue', edgecolor='black')
                    axes[idx].set_title(f'{col} Distribution', fontweight='bold')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
                    axes[idx].grid(True, alpha=0.3)
                    
                    # Add statistical annotations
                    mean_val = self.data[col].mean()
                    median_val = self.data[col].median()
                    axes[idx].axvline(mean_val, color='red', 
                                    linestyle='--', label=f'Mean: {mean_val:.2f}')
                    axes[idx].axvline(median_val, color='green', 
                                    linestyle='--', label=f'Median: {median_val:.2f}')
                    axes[idx].legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'distributions.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved distribution plots to {self.output_dir}/distributions.png")
            plt.close()
        
        return stats.to_dict()
    
    def phase4_correlation_analysis(self):
        """
        Phase 4: Relationship mapping - finding what connects.
        Amazon uses this for "customers who bought X also bought Y".
        """
        print("\n🔗 PHASE 4: CORRELATION ANALYSIS")
        print("-" * 60)
        
        # Select numerical columns only
        numerical_data = self.data.select_dtypes(include=[np.number])
        
        if len(numerical_data.columns) < 2:
            print("Insufficient numerical columns for correlation analysis")
            return {}
        
        # Correlation matrix
        corr_matrix = numerical_data.corr().round(3)
        
        corr_report = f"""
Correlation Matrix:
{corr_matrix.to_string()}

Strong Correlations (|r| > 0.7):
        """
        
        # Find strong correlations
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corrs.append(
                        f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}"
                    )
        
        if strong_corrs:
            corr_report += "\n" + "\n".join(strong_corrs)
        else:
            corr_report += "\nNo strong correlations found (this can be good - features are independent!)"
        
        print(corr_report)
        self.add_to_report("CORRELATION ANALYSIS", corr_report)
        
        # Visualize correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, 
                   linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title(f'{self.name} - Correlation Heatmap', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Saved correlation heatmap to {self.output_dir}/correlation_heatmap.png")
        plt.close()
        
        return corr_matrix.to_dict()
    
    def phase5_insights_synthesis(self):
        """
        Phase 5: Creating the final report - making it actionable.
        This is what you present to stakeholders at companies like Spotify.
        """
        print("\n📋 PHASE 5: INSIGHT SYNTHESIS")
        print("-" * 60)
        
        # Generate categorical analysis if applicable
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            cat_report = "\nCategorical Features Analysis:\n"
            
            for col in categorical_cols[:3]:  # Top 3 categorical columns
                value_counts = self.data[col].value_counts()
                cat_report += f"\n{col}:"
                cat_report += f"\n  - Unique values: {self.data[col].nunique()}"
                cat_report += f"\n  - Most common: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)"
                cat_report += f"\n  - Distribution: {value_counts.head(3).to_dict()}"
            
            print(cat_report)
            self.add_to_report("CATEGORICAL ANALYSIS", cat_report)
        
        # Save complete report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f'eda_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write(f"EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write(f"Dataset: {self.name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("".join(self.report))
        
        print(f"\n✅ Complete EDA report saved to {report_file}")
        
        return report_file
    
    def run_complete_eda(self):
        """
        Execute the full EDA pipeline - this is your production workflow.
        Call this one method to perform systematic data investigation.
        """
        print("\n" + "="*60)
        print(f"STARTING COMPLETE EDA: {self.name}")
        print("="*60)
        
        results = {}
        
        # Run all phases
        results['profiling'] = self.phase1_data_profiling()
        results['quality'] = self.phase2_data_quality()
        results['statistics'] = self.phase3_statistical_analysis()
        results['correlations'] = self.phase4_correlation_analysis()
        results['report_path'] = self.phase5_insights_synthesis()
        
        print("\n" + "="*60)
        print("✅ EDA COMPLETE!")
        print("="*60)
        print(f"\nAll outputs saved to: {self.output_dir}/")
        print(f"  - Distribution plots: distributions.png")
        print(f"  - Correlation heatmap: correlation_heatmap.png")
        print(f"  - Full report: {results['report_path'].name}")
        
        return results


def generate_ecommerce_dataset(n_samples=100000):
    """
    Generate realistic e-commerce user behavior dataset.
    This simulates data like what Amazon, Shopify, or Etsy analyze.
    """
    print(f"📦 Generating synthetic e-commerce dataset ({n_samples:,} samples)...")
    
    np.random.seed(42)
    
    # Generate timestamps over 90 days
    start_date = datetime.now() - timedelta(days=90)
    timestamps = [start_date + timedelta(
        seconds=np.random.randint(0, 90*24*3600)
    ) for _ in range(n_samples)]
    
    # User demographics
    ages = np.random.normal(35, 12, n_samples).clip(18, 75).astype(int)
    
    # Browsing behavior
    pages_viewed = np.random.lognormal(2, 1, n_samples).clip(1, 100).astype(int)
    time_on_site = pages_viewed * np.random.uniform(20, 120, n_samples)  # seconds
    
    # Purchase behavior (not everyone buys)
    purchase_probability = 0.15  # 15% conversion rate
    purchased = np.random.random(n_samples) < purchase_probability
    
    # Revenue (only if purchased)
    revenue = np.where(
        purchased,
        np.random.lognormal(3.5, 1.2, n_samples).clip(10, 2000),
        0
    )
    
    # Device type
    devices = np.random.choice(
        ['Mobile', 'Desktop', 'Tablet'],
        n_samples,
        p=[0.6, 0.35, 0.05]
    )
    
    # Traffic source
    sources = np.random.choice(
        ['Organic', 'Paid', 'Social', 'Email', 'Direct'],
        n_samples,
        p=[0.35, 0.25, 0.20, 0.10, 0.10]
    )
    
    # Customer segment (derived from behavior)
    def assign_segment(age, pages, purchased):
        if purchased and pages > 10:
            return 'High-Value'
        elif purchased:
            return 'Converter'
        elif pages > 15:
            return 'Browser'
        else:
            return 'Bouncer'
    
    segments = [assign_segment(age, pages, purch) 
                for age, pages, purch in zip(ages, pages_viewed, purchased)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'user_age': ages,
        'pages_viewed': pages_viewed,
        'time_on_site_seconds': time_on_site.round(2),
        'purchased': purchased,
        'revenue': revenue.round(2),
        'device_type': devices,
        'traffic_source': sources,
        'customer_segment': segments
    })
    
    # Add some realistic missing values (5% random missing)
    missing_mask = np.random.random((n_samples, len(df.columns))) < 0.05
    for col in ['user_age', 'pages_viewed']:
        df.loc[missing_mask[:, df.columns.get_loc(col)], col] = np.nan
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✓ Dataset generated: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


def main():
    """
    Main execution: Your first complete data investigation project.
    This is exactly how you'd start any AI/ML project in production.
    """
    print("\n" + "🎯 "*20)
    print("DAY 36: EXPLORATORY DATA ANALYSIS PROJECT")
    print("🎯 "*20 + "\n")
    
    # Generate dataset
    data = generate_ecommerce_dataset(n_samples=100000)
    
    # Save dataset
    output_dir = Path("eda_output")
    output_dir.mkdir(exist_ok=True)
    data.to_csv(output_dir / 'ecommerce_data.csv', index=False)
    print(f"✓ Dataset saved to {output_dir}/ecommerce_data.csv\n")
    
    # Initialize EDA Engine
    eda = EDAEngine(data, name="E-Commerce User Behavior")
    
    # Run complete EDA pipeline
    results = eda.run_complete_eda()
    
    # Key insights summary
    print("\n" + "🔑 "*20)
    print("KEY INSIGHTS FROM YOUR INVESTIGATION:")
    print("🔑 "*20)
    print(f"""
    1. Dataset Size: {results['profiling']['n_rows']:,} user sessions analyzed
    
    2. Conversion Rate: {(data['purchased'].sum() / len(data) * 100):.2f}%
       - Industry average: 2-3% (we're doing great!)
    
    3. High-Value Segment: {(data['customer_segment'] == 'High-Value').sum():,} users
       - These are your VIPs - they need special attention
    
    4. Average Revenue per Purchase: ${data[data['purchased']]['revenue'].mean():.2f}
       - Total Revenue: ${data['revenue'].sum():,.2f}
    
    5. Mobile Traffic: {(data['device_type'] == 'Mobile').sum() / len(data) * 100:.1f}%
       - Mobile-first design is critical!
    
    Next Steps for AI:
    • Use this data to train a purchase prediction model
    • Build a customer segmentation classifier
    • Create a recommendation engine for product suggestions
    • Predict customer lifetime value
    
    You've just completed professional-grade EDA! 🎉
    This is the foundation of every AI system in production.
    """)


if __name__ == "__main__":
    main()
