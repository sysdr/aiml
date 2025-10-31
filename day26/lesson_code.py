"""
Day 26: Descriptive Statistics - Mean, Median, Mode
Building blocks for AI data understanding
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Any, Union
import matplotlib.pyplot as plt


class DescriptiveStats:
    """
    Core descriptive statistics calculator.
    Used in production AI systems for data profiling and quality checks.
    """
    
    def __init__(self, data: List[Union[int, float]]):
        """
        Initialize with numerical data.
        
        Args:
            data: List of numerical values to analyze
        """
        if not data:
            raise ValueError("Data cannot be empty")
        self.data = data
        self.sorted_data = sorted(data)
        self.n = len(data)
    
    def mean(self) -> float:
        """
        Calculate arithmetic mean (average).
        
        In AI: Used for feature normalization and centering.
        Sensitive to outliers - can mislead model training.
        
        Returns:
            Mean value
        """
        return sum(self.data) / self.n
    
    def median(self) -> float:
        """
        Calculate median (middle value).
        
        In AI: Robust alternative to mean for outlier-heavy data.
        Used in preprocessing pipelines for financial/medical AI.
        
        Returns:
            Median value
        """
        mid = self.n // 2
        if self.n % 2 == 0:
            # Even number of elements: average of two middle values
            return (self.sorted_data[mid - 1] + self.sorted_data[mid]) / 2.0
        else:
            # Odd number: middle element
            return self.sorted_data[mid]
    
    def mode(self) -> List[Union[int, float]]:
        """
        Calculate mode (most frequent value).
        
        In AI: Critical for categorical features and detecting data patterns.
        Used in chatbot intent analysis and user behavior modeling.
        
        Returns:
            List of mode values (can be multiple if tie)
        """
        counts = Counter(self.data)
        if not counts:
            return []
        
        max_count = max(counts.values())
        modes = [value for value, count in counts.items() if count == max_count]
        
        # If all values appear once, no meaningful mode
        if max_count == 1 and len(modes) == self.n:
            return []
        
        return sorted(modes)
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary.
        
        Returns:
            Dictionary with all key statistics
        """
        mean_val = self.mean()
        median_val = self.median()
        mode_vals = self.mode()
        
        return {
            'count': self.n,
            'mean': round(mean_val, 2),
            'median': round(median_val, 2),
            'mode': mode_vals,
            'min': min(self.data),
            'max': max(self.data),
            'range': max(self.data) - min(self.data)
        }


class DataProfiler:
    """
    Advanced data profiler for AI/ML preprocessing.
    Detects data quality issues before they break models.
    """
    
    def __init__(self, data: List[Union[int, float]], feature_name: str = "feature"):
        """
        Initialize profiler with data and optional feature name.
        
        Args:
            data: Numerical data to profile
            feature_name: Name of the feature (for reporting)
        """
        self.data = data
        self.feature_name = feature_name
        self.stats = DescriptiveStats(data)
    
    def detect_skew(self) -> Dict[str, Any]:
        """
        Detect data skewness - critical for AI model selection.
        
        Skewed data often needs transformation (log, sqrt) before modeling.
        Symmetric data works well with most ML algorithms.
        
        Returns:
            Dictionary with skew analysis
        """
        mean = self.stats.mean()
        median = self.stats.median()
        
        # Calculate percentage difference
        if median != 0:
            diff_pct = abs(mean - median) / abs(median) * 100
        else:
            diff_pct = 0 if mean == 0 else float('inf')
        
        # Interpret skewness
        if diff_pct < 5:
            skew_type = "symmetric"
            recommendation = "Good for most AI models - no transformation needed"
        elif mean > median:
            skew_type = "right-skewed"
            recommendation = "Consider log transformation or robust scaling"
        else:
            skew_type = "left-skewed"
            recommendation = "Check for data collection issues or negative outliers"
        
        return {
            'feature': self.feature_name,
            'mean': round(mean, 2),
            'median': round(median, 2),
            'difference_pct': round(diff_pct, 2),
            'skew_type': skew_type,
            'recommendation': recommendation
        }
    
    def detect_outliers_iqr(self) -> Dict[str, Any]:
        """
        Detect outliers using IQR method (preview for later lessons).
        
        Returns:
            Dictionary with outlier information
        """
        q1 = np.percentile(self.data, 25)
        q3 = np.percentile(self.data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [x for x in self.data if x < lower_bound or x > upper_bound]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': round(len(outliers) / len(self.data) * 100, 2),
            'outliers': sorted(outliers)[:10],  # Show first 10
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2)
        }
    
    def generate_report(self) -> str:
        """
        Generate human-readable data quality report.
        
        Returns:
            Formatted report string
        """
        summary = self.stats.summary()
        skew_info = self.detect_skew()
        outlier_info = self.detect_outliers_iqr()
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║  DATA PROFILE REPORT: {self.feature_name.upper():<33}  ║
╚══════════════════════════════════════════════════════════╝

📊 BASIC STATISTICS
   Count:      {summary['count']}
   Mean:       {summary['mean']}
   Median:     {summary['median']}
   Mode:       {summary['mode'] if summary['mode'] else 'No dominant mode'}
   Range:      {summary['min']} to {summary['max']} (span: {summary['range']})

📈 DISTRIBUTION ANALYSIS
   Skew Type:  {skew_info['skew_type']}
   Mean-Median Difference: {skew_info['difference_pct']}%
   
   💡 AI Recommendation:
   {skew_info['recommendation']}

⚠️  OUTLIER DETECTION
   Found:      {outlier_info['outlier_count']} outliers ({outlier_info['outlier_percentage']}% of data)
   Bounds:     [{outlier_info['lower_bound']}, {outlier_info['upper_bound']}]
   
   {f"Sample outliers: {outlier_info['outliers']}" if outlier_info['outliers'] else "No outliers detected"}

════════════════════════════════════════════════════════════
"""
        return report


def demo_real_world_scenarios():
    """
    Demonstrate descriptive statistics on real AI/ML scenarios.
    """
    print("=" * 60)
    print("REAL-WORLD AI/ML SCENARIOS - DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    # Scenario 1: User Engagement Analysis
    print("\n📱 SCENARIO 1: User Session Duration (minutes)")
    print("Context: Mobile app analytics before building recommendation model")
    
    session_data = [2, 3, 3, 4, 5, 3, 2, 120, 4, 3, 150, 3, 4, 3, 2, 5, 4, 3]
    profiler1 = DataProfiler(session_data, "session_duration_minutes")
    print(profiler1.generate_report())
    
    # Scenario 2: Customer Transaction Amounts
    print("\n💳 SCENARIO 2: Transaction Amounts ($)")
    print("Context: Fraud detection model - profiling normal transactions")
    
    transaction_data = [45, 67, 52, 48, 51, 49, 53, 47, 50, 48, 
                       52, 3500, 48, 51, 49, 46, 52, 48]
    profiler2 = DataProfiler(transaction_data, "transaction_amount_usd")
    print(profiler2.generate_report())
    
    # Scenario 3: Model Prediction Latency
    print("\n⚡ SCENARIO 3: AI Model Response Time (ms)")
    print("Context: Production monitoring - ensuring SLA compliance")
    
    latency_data = [45, 47, 46, 48, 44, 46, 47, 45, 49, 46, 
                   48, 45, 47, 280, 46, 47, 45, 48]
    profiler3 = DataProfiler(latency_data, "response_latency_ms")
    print(profiler3.generate_report())
    
    # Key Learning: Compare scenarios
    print("\n" + "=" * 60)
    print("🎯 KEY INSIGHT: When Mean ≠ Median")
    print("=" * 60)
    print("""
When mean and median differ significantly:
- Your data has outliers (good or bad)
- Distribution is skewed (not normal)
- One summary statistic lies - you need both!

AI Impact:
- Models trained on mean-centered features: Sensitive to outliers
- Models trained on median-centered features: More robust
- Best practice: Always check both before model training
    """)


def interactive_calculator():
    """
    Interactive calculator for learning.
    """
    print("\n" + "=" * 60)
    print("🧮 INTERACTIVE DESCRIPTIVE STATISTICS CALCULATOR")
    print("=" * 60)
    
    # Example datasets for experimentation
    examples = {
        '1': ([10, 20, 30, 40, 50], "Symmetric distribution"),
        '2': ([1, 2, 3, 4, 100], "Right-skewed with outlier"),
        '3': ([100, 2, 3, 4, 5], "Left-skewed with outlier"),
        '4': ([5, 5, 5, 5, 5], "No variation (all same)"),
        '5': ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Uniform distribution")
    }
    
    print("\nChoose an example dataset:")
    for key, (data, desc) in examples.items():
        print(f"  {key}. {desc}: {data}")
    
    print("\nAnalyzing all examples...")
    for key, (data, desc) in examples.items():
        print(f"\n--- Example {key}: {desc} ---")
        stats = DescriptiveStats(data)
        summary = stats.summary()
        print(f"Data: {data}")
        print(f"Mean:   {summary['mean']}")
        print(f"Median: {summary['median']}")
        print(f"Mode:   {summary['mode'] if summary['mode'] else 'No mode'}")


if __name__ == "__main__":
    # Run demonstrations
    demo_real_world_scenarios()
    interactive_calculator()
    
    print("\n" + "=" * 60)
    print("✅ Lesson Complete!")
    print("=" * 60)
    print("""
Next Steps:
1. Run test_lesson.py to verify your understanding
2. Experiment with your own datasets
3. Tomorrow: Measures of Spread (Variance & Standard Deviation)

Remember: Descriptive statistics are your first conversation with data.
Listen carefully - they reveal what your AI models will learn.
    """)
