"""
Day 30: ML Dataset Analyzer
A production-ready tool for analyzing dataset quality before ML training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any
import warnings
from datetime import datetime
from tabulate import tabulate

warnings.filterwarnings('ignore')

class MLDatasetAnalyzer:
    """
    Comprehensive dataset analyzer for ML pipelines.
    Implements statistical profiling used at FAANG companies.
    """
    
    def __init__(self, dataframe: pd.DataFrame, target_column: str = None):
        """
        Initialize analyzer with a dataset.
        
        Args:
            dataframe: Input dataset to analyze
            target_column: Optional target variable for ML tasks
        """
        self.df = dataframe.copy()
        self.target_column = target_column
        self.numeric_features = []
        self.categorical_features = []
        self.analysis_results = {}
        
        self._identify_feature_types()
        
    def _identify_feature_types(self):
        """Classify features as numeric or categorical"""
        for col in self.df.columns:
            if col == self.target_column:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
    
    def profile_features(self) -> Dict[str, Dict]:
        """
        Generate comprehensive statistical profile for each numeric feature.
        Uses concepts from Days 26-27: descriptive statistics and measures of spread.
        
        Returns:
            Dictionary with detailed statistics for each feature
        """
        print("📊 Profiling features...")
        profiles = {}
        
        for feature in self.numeric_features:
            data = self.df[feature].dropna()
            
            if len(data) == 0:
                continue
            
            # Central tendency (Day 26)
            mean_val = data.mean()
            median_val = data.median()
            mode_result = data.mode()
            mode_val = mode_result[0] if len(mode_result) > 0 else None
            
            # Measures of spread (Day 27)
            std_val = data.std()
            var_val = data.var()
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            
            # Distribution shape
            skewness = data.skew()
            kurt = data.kurtosis()
            
            # Range
            min_val = data.min()
            max_val = data.max()
            range_val = max_val - min_val
            
            # Outlier detection using IQR method (Day 27)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(data)) * 100
            
            profiles[feature] = {
                'count': len(data),
                'missing': self.df[feature].isna().sum(),
                'missing_pct': (self.df[feature].isna().sum() / len(self.df)) * 100,
                'mean': mean_val,
                'median': median_val,
                'mode': mode_val,
                'std': std_val,
                'variance': var_val,
                'min': min_val,
                'max': max_val,
                'range': range_val,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'skewness': skewness,
                'kurtosis': kurt,
                'outlier_count': outlier_count,
                'outlier_pct': outlier_percentage,
                'cv': (std_val / mean_val * 100) if mean_val != 0 else 0  # Coefficient of variation
            }
        
        self.analysis_results['feature_profiles'] = profiles
        return profiles
    
    def detect_quality_issues(self) -> Dict[str, Any]:
        """
        Identify data quality issues that affect ML model performance.
        
        Returns:
            Dictionary of quality issues and severity
        """
        print("🔍 Detecting quality issues...")
        issues = {
            'high_missing': [],
            'high_outliers': [],
            'zero_variance': [],
            'highly_skewed': [],
            'imbalanced_target': None
        }
        
        profiles = self.analysis_results.get('feature_profiles', {})
        
        for feature, profile in profiles.items():
            # High missing data (>30% is concerning)
            if profile['missing_pct'] > 30:
                issues['high_missing'].append({
                    'feature': feature,
                    'missing_pct': profile['missing_pct']
                })
            
            # Too many outliers (>10% suggests data issues)
            if profile['outlier_pct'] > 10:
                issues['high_outliers'].append({
                    'feature': feature,
                    'outlier_pct': profile['outlier_pct']
                })
            
            # Zero or near-zero variance (useless for ML)
            if profile['std'] < 1e-10:
                issues['zero_variance'].append(feature)
            
            # Highly skewed distributions (|skewness| > 2)
            if abs(profile['skewness']) > 2:
                issues['highly_skewed'].append({
                    'feature': feature,
                    'skewness': profile['skewness'],
                    'recommendation': 'log' if profile['skewness'] > 0 else 'sqrt'
                })
        
        # Check target imbalance (if target provided)
        if self.target_column and self.target_column in self.df.columns:
            target_dist = self.df[self.target_column].value_counts(normalize=True)
            max_class_pct = target_dist.max() * 100
            if max_class_pct > 90:
                issues['imbalanced_target'] = {
                    'max_class_pct': max_class_pct,
                    'distribution': target_dist.to_dict()
                }
        
        self.analysis_results['quality_issues'] = issues
        return issues
    
    def analyze_correlations(self, threshold: float = 0.8) -> Tuple[pd.DataFrame, List[Tuple]]:
        """
        Calculate feature correlations and identify multicollinearity.
        Uses concepts from Day 28: correlation and covariance.
        
        Args:
            threshold: Correlation threshold for flagging redundant features
            
        Returns:
            Correlation matrix and list of highly correlated pairs
        """
        print("🔗 Analyzing correlations...")
        
        if len(self.numeric_features) < 2:
            return pd.DataFrame(), []
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_features].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        self.analysis_results['correlation_matrix'] = corr_matrix
        self.analysis_results['high_correlations'] = high_corr_pairs
        
        return corr_matrix, high_corr_pairs
    
    def test_normality(self) -> Dict[str, Dict]:
        """
        Test if features follow normal distribution.
        Connects to Day 29: Central Limit Theorem.
        
        Returns:
            Dictionary with normality test results for each feature
        """
        print("📈 Testing normality...")
        normality_results = {}
        
        for feature in self.numeric_features:
            data = self.df[feature].dropna()
            
            if len(data) < 20:  # Need sufficient data for test
                continue
            
            # Shapiro-Wilk test for normality
            stat, p_value = stats.shapiro(data[:5000])  # Limit to 5000 for performance
            
            normality_results[feature] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05,  # Common significance level
                'interpretation': 'Normal' if p_value > 0.05 else 'Not Normal'
            }
        
        self.analysis_results['normality_tests'] = normality_results
        return normality_results
    
    def calculate_ml_readiness_score(self) -> Dict[str, Any]:
        """
        Calculate overall dataset quality score for ML readiness.
        
        Returns:
            Score (0-100) and breakdown
        """
        print("⚡ Calculating ML readiness...")
        
        score = 100
        deductions = {}
        
        # Check missing data
        avg_missing = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100
        if avg_missing > 0:
            missing_penalty = min(avg_missing * 0.5, 25)
            score -= missing_penalty
            deductions['missing_data'] = missing_penalty
        
        # Check outliers
        profiles = self.analysis_results.get('feature_profiles', {})
        avg_outliers = np.mean([p['outlier_pct'] for p in profiles.values()]) if profiles else 0
        if avg_outliers > 5:
            outlier_penalty = min((avg_outliers - 5) * 0.3, 20)
            score -= outlier_penalty
            deductions['outliers'] = outlier_penalty
        
        # Check feature variance
        zero_var_count = len(self.analysis_results.get('quality_issues', {}).get('zero_variance', []))
        if zero_var_count > 0:
            var_penalty = min(zero_var_count * 5, 15)
            score -= var_penalty
            deductions['zero_variance'] = var_penalty
        
        # Check multicollinearity
        high_corr_count = len(self.analysis_results.get('high_correlations', []))
        if high_corr_count > 0:
            corr_penalty = min(high_corr_count * 3, 15)
            score -= corr_penalty
            deductions['multicollinearity'] = corr_penalty
        
        # Check imbalance
        if self.analysis_results.get('quality_issues', {}).get('imbalanced_target'):
            score -= 10
            deductions['class_imbalance'] = 10
        
        score = max(0, score)
        
        # Interpret score
        if score >= 90:
            interpretation = "Excellent - Ready for ML"
        elif score >= 75:
            interpretation = "Good - Minor preprocessing needed"
        elif score >= 60:
            interpretation = "Fair - Moderate preprocessing required"
        else:
            interpretation = "Poor - Significant data quality work needed"
        
        readiness = {
            'score': score,
            'interpretation': interpretation,
            'deductions': deductions
        }
        
        self.analysis_results['ml_readiness'] = readiness
        return readiness
    
    def generate_visualizations(self, output_dir: str = 'plots'):
        """Generate key visualizations for the analysis"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 Generating visualizations in '{output_dir}/'...")
        
        # 1. Correlation heatmap
        if len(self.numeric_features) >= 2:
            plt.figure(figsize=(10, 8))
            corr_matrix = self.analysis_results.get('correlation_matrix')
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=150)
            plt.close()
        
        # 2. Distribution plots for each feature
        n_features = len(self.numeric_features)
        if n_features > 0:
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]
            
            for idx, feature in enumerate(self.numeric_features):
                ax = axes[idx]
                data = self.df[feature].dropna()
                
                ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
                ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label='Median')
                ax.set_title(f'{feature} Distribution', fontweight='bold')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(alpha=0.3)
            
            # Hide unused subplots
            for idx in range(n_features, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_distributions.png', dpi=150)
            plt.close()
        
        # 3. Missing data visualization
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            plt.figure(figsize=(10, 6))
            missing_data[missing_data > 0].sort_values().plot(kind='barh', color='coral')
            plt.title('Missing Data by Feature', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Missing Values')
            plt.ylabel('Feature')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/missing_data.png', dpi=150)
            plt.close()
    
    def generate_report(self, output_file: str = 'analysis_report.html'):
        """
        Generate comprehensive HTML report with all analysis results.
        """
        print(f"📝 Generating report: {output_file}")
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ML Dataset Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                }}
                .section {{
                    background: white;
                    padding: 25px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .section h2 {{
                    color: #667eea;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }}
                .metric-card h3 {{
                    margin: 0 0 5px 0;
                    font-size: 0.9em;
                    color: #666;
                }}
                .metric-card .value {{
                    font-size: 1.8em;
                    font-weight: bold;
                    color: #333;
                }}
                .score {{
                    font-size: 4em;
                    font-weight: bold;
                    text-align: center;
                    margin: 20px 0;
                }}
                .score.excellent {{ color: #28a745; }}
                .score.good {{ color: #17a2b8; }}
                .score.fair {{ color: #ffc107; }}
                .score.poor {{ color: #dc3545; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #667eea;
                    color: white;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .issue-badge {{
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 0.85em;
                    font-weight: bold;
                    margin: 2px;
                }}
                .issue-critical {{ background-color: #dc3545; color: white; }}
                .issue-warning {{ background-color: #ffc107; color: #333; }}
                .issue-info {{ background-color: #17a2b8; color: white; }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 ML Dataset Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Dataset: {len(self.df)} rows × {len(self.df.columns)} columns</p>
            </div>
        """
        
        # ML Readiness Score
        readiness = self.analysis_results.get('ml_readiness', {})
        score = readiness.get('score', 0)
        score_class = 'excellent' if score >= 90 else 'good' if score >= 75 else 'fair' if score >= 60 else 'poor'
        
        html += f"""
            <div class="section">
                <h2>ML Readiness Score</h2>
                <div class="score {score_class}">{score:.1f}/100</div>
                <p style="text-align: center; font-size: 1.2em; color: #666;">
                    {readiness.get('interpretation', 'Unknown')}
                </p>
            </div>
        """
        
        # Dataset Overview
        html += f"""
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>Total Rows</h3>
                        <div class="value">{len(self.df):,}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Total Features</h3>
                        <div class="value">{len(self.df.columns)}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Numeric Features</h3>
                        <div class="value">{len(self.numeric_features)}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Categorical Features</h3>
                        <div class="value">{len(self.categorical_features)}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Missing Values</h3>
                        <div class="value">{self.df.isnull().sum().sum():,}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Memory Usage</h3>
                        <div class="value">{self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB</div>
                    </div>
                </div>
            </div>
        """
        
        # Feature Profiles
        profiles = self.analysis_results.get('feature_profiles', {})
        if profiles:
            html += """
                <div class="section">
                    <h2>Feature Statistical Profiles</h2>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Mean</th>
                            <th>Median</th>
                            <th>Std Dev</th>
                            <th>Skewness</th>
                            <th>Missing %</th>
                            <th>Outliers %</th>
                        </tr>
            """
            
            for feature, profile in profiles.items():
                html += f"""
                        <tr>
                            <td><strong>{feature}</strong></td>
                            <td>{profile['mean']:.3f}</td>
                            <td>{profile['median']:.3f}</td>
                            <td>{profile['std']:.3f}</td>
                            <td>{profile['skewness']:.3f}</td>
                            <td>{profile['missing_pct']:.1f}%</td>
                            <td>{profile['outlier_pct']:.1f}%</td>
                        </tr>
                """
            
            html += """
                    </table>
                </div>
            """
        
        # Quality Issues
        issues = self.analysis_results.get('quality_issues', {})
        html += """
            <div class="section">
                <h2>Data Quality Issues</h2>
        """
        
        has_issues = False
        
        if issues.get('high_missing'):
            has_issues = True
            html += "<h3>⚠️ High Missing Data</h3><ul>"
            for item in issues['high_missing']:
                html += f"<li><span class='issue-badge issue-warning'>{item['feature']}</span> - {item['missing_pct']:.1f}% missing</li>"
            html += "</ul>"
        
        if issues.get('high_outliers'):
            has_issues = True
            html += "<h3>⚠️ High Outlier Count</h3><ul>"
            for item in issues['high_outliers']:
                html += f"<li><span class='issue-badge issue-warning'>{item['feature']}</span> - {item['outlier_pct']:.1f}% outliers</li>"
            html += "</ul>"
        
        if issues.get('highly_skewed'):
            has_issues = True
            html += "<h3>📊 Highly Skewed Features</h3><ul>"
            for item in issues['highly_skewed']:
                html += f"<li><span class='issue-badge issue-info'>{item['feature']}</span> - Skewness: {item['skewness']:.2f} (Try {item['recommendation']} transform)</li>"
            html += "</ul>"
        
        if not has_issues:
            html += "<p style='color: #28a745; font-size: 1.1em;'>✅ No critical quality issues detected!</p>"
        
        html += "</div>"
        
        # Correlations
        high_corr = self.analysis_results.get('high_correlations', [])
        if high_corr:
            html += """
                <div class="section">
                    <h2>High Correlations (Potential Multicollinearity)</h2>
                    <table>
                        <tr>
                            <th>Feature 1</th>
                            <th>Feature 2</th>
                            <th>Correlation</th>
                        </tr>
            """
            
            for feat1, feat2, corr in high_corr:
                html += f"""
                        <tr>
                            <td>{feat1}</td>
                            <td>{feat2}</td>
                            <td>{corr:.3f}</td>
                        </tr>
                """
            
            html += """
                    </table>
                    <p><em>Consider removing one feature from each highly correlated pair.</em></p>
                </div>
            """
        
        html += """
            <div class="footer">
                <p>Report generated by ML Dataset Analyzer - Day 30 Project</p>
                <p>Part of the 180-Day AI/ML Course from Scratch</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"✅ Report saved to: {output_file}")
    
    def run_full_analysis(self, generate_viz: bool = True, generate_html: bool = True):
        """
        Run complete analysis pipeline.
        
        Args:
            generate_viz: Whether to generate visualization plots
            generate_html: Whether to generate HTML report
        """
        print("=" * 60)
        print("🚀 Starting ML Dataset Analysis Pipeline")
        print("=" * 60)
        
        # Run all analysis steps
        self.profile_features()
        self.detect_quality_issues()
        self.analyze_correlations()
        self.test_normality()
        self.calculate_ml_readiness_score()
        
        # Generate outputs
        if generate_viz:
            self.generate_visualizations()
        
        if generate_html:
            self.generate_report()
        
        # Print summary to console
        self._print_summary()
        
        print("\n" + "=" * 60)
        print("✅ Analysis Complete!")
        print("=" * 60)
    
    def _print_summary(self):
        """Print analysis summary to console"""
        print("\n" + "=" * 60)
        print("📋 ANALYSIS SUMMARY")
        print("=" * 60)
        
        readiness = self.analysis_results.get('ml_readiness', {})
        print(f"\n🎯 ML Readiness Score: {readiness.get('score', 0):.1f}/100")
        print(f"   {readiness.get('interpretation', 'Unknown')}")
        
        issues = self.analysis_results.get('quality_issues', {})
        print(f"\n📊 Data Quality:")
        print(f"   - High Missing Data: {len(issues.get('high_missing', []))} features")
        print(f"   - High Outliers: {len(issues.get('high_outliers', []))} features")
        print(f"   - Zero Variance: {len(issues.get('zero_variance', []))} features")
        print(f"   - Highly Skewed: {len(issues.get('highly_skewed', []))} features")
        
        high_corr = self.analysis_results.get('high_correlations', [])
        print(f"\n🔗 Multicollinearity: {len(high_corr)} highly correlated pairs")


def create_sample_datasets():
    """Create sample datasets for testing"""
    
    # Clean dataset
    np.random.seed(42)
    clean_data = {
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'credit_score': np.random.normal(700, 50, 1000),
        'loan_amount': np.random.normal(15000, 5000, 1000),
        'approved': np.random.choice([0, 1], 1000, p=[0.3, 0.7])
    }
    clean_df = pd.DataFrame(clean_data)
    clean_df.to_csv('clean_dataset.csv', index=False)
    
    # Messy dataset
    messy_data = {
        'age': np.concatenate([np.random.normal(35, 10, 997), [150, 200, -5]]),  # Outliers
        'income': np.concatenate([np.random.normal(50000, 15000, 950), [1000000] * 50]),  # Outliers
        'credit_score': np.random.exponential(500, 1000),  # Skewed
        'loan_amount': np.random.normal(15000, 5000, 1000),
        'approved': np.random.choice([0, 1], 1000, p=[0.95, 0.05])  # Imbalanced
    }
    messy_df = pd.DataFrame(messy_data)
    # Add missing values
    messy_df.loc[np.random.choice(1000, 200, replace=False), 'income'] = np.nan
    messy_df.loc[np.random.choice(1000, 150, replace=False), 'credit_score'] = np.nan
    messy_df.to_csv('messy_dataset.csv', index=False)
    
    print("✅ Sample datasets created:")
    print("   - clean_dataset.csv")
    print("   - messy_dataset.csv")


def main():
    """Main execution function"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        ML Dataset Analyzer - Day 30 Project             ║
    ║        180-Day AI/ML Course from Scratch                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Create sample datasets if they don't exist
    import os
    if not os.path.exists('clean_dataset.csv'):
        print("\n📦 Creating sample datasets...")
        create_sample_datasets()
    
    # Example 1: Analyze clean dataset
    print("\n\n" + "="*60)
    print("EXAMPLE 1: Analyzing Clean Dataset")
    print("="*60)
    
    clean_df = pd.read_csv('clean_dataset.csv')
    analyzer1 = MLDatasetAnalyzer(clean_df, target_column='approved')
    analyzer1.run_full_analysis(generate_viz=True, generate_html=True)
    
    # Example 2: Analyze messy dataset
    print("\n\n" + "="*60)
    print("EXAMPLE 2: Analyzing Messy Dataset")
    print("="*60)
    
    messy_df = pd.read_csv('messy_dataset.csv')
    analyzer2 = MLDatasetAnalyzer(messy_df, target_column='approved')
    analyzer2.run_full_analysis(generate_viz=True, generate_html=True)
    
    # Example 3: Using real ML dataset (Iris)
    print("\n\n" + "="*60)
    print("EXAMPLE 3: Analyzing Iris Dataset")
    print("="*60)
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    
    analyzer3 = MLDatasetAnalyzer(iris_df, target_column='species')
    analyzer3.run_full_analysis(generate_viz=True, generate_html=True)
    
    print("\n\n" + "="*60)
    print("🎉 All analyses complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  📊 Visualizations: plots/ directory")
    print("  📄 Reports: analysis_report.html")
    print("\nOpen analysis_report.html in your browser to view the full report!")


if __name__ == "__main__":
    main()
