#!/bin/bash

# Day 26: Descriptive Statistics - Implementation Package Generator
# This script creates all necessary files for the lesson

echo "🚀 Generating Day 26: Descriptive Statistics Implementation Files..."

# Create environment setup script
cat > env_setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 26: Descriptive Statistics Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "To activate environment: source venv/bin/activate"
echo "To run lesson: python lesson_code.py"
echo "To run tests: python test_lesson.py"
echo "To start dashboard: python app.py"
EOF

chmod +x env_setup.sh

# Create requirements file
cat > requirements.txt << 'EOF'
numpy==1.26.4
matplotlib==3.8.4
scipy==1.13.0
jupyter==1.0.0
pandas==2.2.2
flask==3.0.0
flask-cors==4.0.0
EOF

# Create main lesson code
cat > lesson_code.py << 'EOF'
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
EOF

# Create test file
cat > test_lesson.py << 'EOF'
"""
Test suite for Day 26: Descriptive Statistics
Run this to verify your understanding
"""

import unittest
from lesson_code import DescriptiveStats, DataProfiler


class TestDescriptiveStats(unittest.TestCase):
    """Test basic descriptive statistics calculations."""
    
    def test_mean_calculation(self):
        """Test mean calculation with known values."""
        data = [1, 2, 3, 4, 5]
        stats = DescriptiveStats(data)
        self.assertEqual(stats.mean(), 3.0)
        
    def test_median_odd_count(self):
        """Test median with odd number of elements."""
        data = [1, 3, 5, 7, 9]
        stats = DescriptiveStats(data)
        self.assertEqual(stats.median(), 5)
    
    def test_median_even_count(self):
        """Test median with even number of elements."""
        data = [1, 2, 3, 4]
        stats = DescriptiveStats(data)
        self.assertEqual(stats.median(), 2.5)
    
    def test_mode_single(self):
        """Test mode with single most frequent value."""
        data = [1, 2, 2, 3, 4]
        stats = DescriptiveStats(data)
        self.assertEqual(stats.mode(), [2])
    
    def test_mode_multiple(self):
        """Test mode with multiple most frequent values."""
        data = [1, 1, 2, 2, 3]
        stats = DescriptiveStats(data)
        self.assertIn(1, stats.mode())
        self.assertIn(2, stats.mode())
    
    def test_mode_no_repeat(self):
        """Test mode when all values appear once."""
        data = [1, 2, 3, 4, 5]
        stats = DescriptiveStats(data)
        self.assertEqual(stats.mode(), [])
    
    def test_outlier_impact_on_mean(self):
        """Test that outliers significantly affect mean but not median."""
        normal_data = [10, 12, 11, 13, 12]
        with_outlier = [10, 12, 11, 13, 1000]
        
        normal_stats = DescriptiveStats(normal_data)
        outlier_stats = DescriptiveStats(with_outlier)
        
        # Mean should change dramatically
        self.assertGreater(outlier_stats.mean(), normal_stats.mean() * 2)
        
        # Median should be similar
        self.assertAlmostEqual(normal_stats.median(), outlier_stats.median(), delta=2)


class TestDataProfiler(unittest.TestCase):
    """Test data profiling and skew detection."""
    
    def test_symmetric_detection(self):
        """Test detection of symmetric distribution."""
        data = [10, 20, 30, 40, 50]
        profiler = DataProfiler(data)
        skew_info = profiler.detect_skew()
        self.assertEqual(skew_info['skew_type'], 'symmetric')
    
    def test_right_skew_detection(self):
        """Test detection of right-skewed distribution."""
        data = [1, 2, 3, 4, 100]
        profiler = DataProfiler(data)
        skew_info = profiler.detect_skew()
        self.assertEqual(skew_info['skew_type'], 'right-skewed')
    
    def test_left_skew_detection(self):
        """Test detection of left-skewed distribution."""
        data = [100, 2, 3, 4, 5]
        profiler = DataProfiler(data)
        skew_info = profiler.detect_skew()
        self.assertEqual(skew_info['skew_type'], 'left-skewed')
    
    def test_outlier_detection(self):
        """Test outlier detection using IQR method."""
        data = [10, 12, 11, 13, 12, 100]  # 100 is clear outlier
        profiler = DataProfiler(data)
        outlier_info = profiler.detect_outliers_iqr()
        self.assertGreater(outlier_info['outlier_count'], 0)
        self.assertIn(100, outlier_info['outliers'])


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world AI/ML scenarios."""
    
    def test_user_session_analysis(self):
        """Test user session duration analysis (recommendation systems)."""
        # Most users: 2-5 minutes, few power users: 120-150 minutes
        sessions = [2, 3, 3, 4, 5, 3, 2, 120, 4, 3, 150, 3, 4]
        stats = DescriptiveStats(sessions)
        
        mean = stats.mean()
        median = stats.median()
        
        # Mean should be much higher than median (outliers pull it up)
        self.assertGreater(mean, median * 2)
        self.assertLess(median, 5)  # Typical user < 5 minutes
    
    def test_transaction_fraud_detection(self):
        """Test transaction amount analysis (fraud detection)."""
        # Normal transactions: ~$50, fraudulent: $3500
        transactions = [45, 50, 48, 52, 3500, 49, 51]
        profiler = DataProfiler(transactions)
        outliers = profiler.detect_outliers_iqr()
        
        # Should detect the fraudulent transaction
        self.assertIn(3500, outliers['outliers'])
    
    def test_model_latency_monitoring(self):
        """Test AI model latency monitoring."""
        # Normal: 45-50ms, spike: 280ms
        latencies = [45, 47, 46, 48, 280, 46, 47]
        stats = DescriptiveStats(latencies)
        
        # Median should represent normal performance
        self.assertLess(stats.median(), 50)
        # Mean should be affected by spike
        self.assertGreater(stats.mean(), stats.median())


def run_tests():
    """Run all tests and display results."""
    print("=" * 60)
    print("RUNNING TESTS FOR DAY 26: DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestDescriptiveStats))
    suite.addTests(loader.loadTestsFromTestCase(TestDataProfiler))
    suite.addTests(loader.loadTestsFromTestCase(TestRealWorldScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED! You understand descriptive statistics!")
        print("\nYou've mastered:")
        print("  • Mean, median, and mode calculations")
        print("  • Skew detection and interpretation")
        print("  • Real-world AI/ML applications")
        print("  • Data quality profiling")
        print("\nReady for Day 27: Variance and Standard Deviation!")
    else:
        print("\n❌ Some tests failed. Review the lesson and try again.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
EOF

# Create README
cat > README.md << 'EOF'
# Day 26: Descriptive Statistics (Mean, Median, Mode)

## Quick Start

```bash
# 1. Run setup (creates virtual environment and installs dependencies)
chmod +x setup.sh
./setup.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run the lesson
python lesson_code.py

# 4. Test your understanding
python test_lesson.py
```

## What You'll Learn

- Calculate mean, median, and mode in Python
- Detect data skewness and outliers
- Understand how these metrics guide AI model selection
- Build a data profiler used in production AI systems

## Files Included

- `lesson_code.py` - Main implementation with real-world examples
- `test_lesson.py` - Comprehensive test suite to verify understanding
- `setup.sh` - Automated environment setup
- `requirements.txt` - Python dependencies

## Learning Objectives

After completing this lesson, you'll be able to:

1. Calculate descriptive statistics on any dataset
2. Interpret what mean vs median differences reveal about data
3. Detect data quality issues before model training
4. Profile datasets like production AI engineers

## Real-World Applications

- **User Behavior Analysis**: Understanding typical vs power user patterns
- **Fraud Detection**: Identifying outlier transactions
- **Model Performance Monitoring**: Tracking latency and accuracy
- **Feature Engineering**: Deciding on data transformations

## Time Estimate

2-3 hours including:
- Reading main article (30 min)
- Running code examples (45 min)
- Completing tests (45 min)
- Experimentation (30 min)

## Next Lesson

Day 27: Measures of Spread (Variance and Standard Deviation)

## Need Help?

Common issues:
- **Import errors**: Run `setup.sh` again
- **Python version**: Requires Python 3.11+
- **Test failures**: Review lesson_code.py comments

## Key Takeaway

Descriptive statistics are your first conversation with data. Before training any AI model, always calculate mean, median, and mode. When they disagree, your data has a story to tell - listen carefully!
EOF

# Create Jupyter notebook
cat > descriptive_statistics.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Day 26: Descriptive Statistics - Interactive Exploration\n",
    "\n",
    "This notebook lets you experiment with descriptive statistics interactively."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import our lesson code\n",
    "from lesson_code import DescriptiveStats, DataProfiler\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "print(\"✅ Imports successful!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Experiment 1: Understanding Mean vs Median"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Try different datasets\n",
    "datasets = {\n",
    "    'Symmetric': [10, 20, 30, 40, 50],\n",
    "    'Right-skewed': [1, 2, 3, 4, 100],\n",
    "    'Left-skewed': [100, 2, 3, 4, 5],\n",
    "    'With outliers': [45, 47, 46, 48, 280, 46, 47]\n",
    "}\n",
    "\n",
    "for name, data in datasets.items():\n",
    "    stats = DescriptiveStats(data)\n",
    "    print(f\"\\n{name}:\")\n",
    "    print(f\"  Data: {data}\")\n",
    "    print(f\"  Mean: {stats.mean():.2f}\")\n",
    "    print(f\"  Median: {stats.median():.2f}\")\n",
    "    print(f\"  Mode: {stats.mode()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Experiment 2: Your Own Data\n",
    "\n",
    "Try your own dataset! Replace the list below:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Your data here\n",
    "my_data = [12, 15, 12, 18, 22, 12, 25, 28, 12, 30]\n",
    "\n",
    "profiler = DataProfiler(my_data, \"my_feature\")\n",
    "print(profiler.generate_report())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Experiment 3: Visualizing Distributions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize how outliers affect statistics\n",
    "normal_data = [10, 12, 11, 13, 12, 11, 14, 12, 13, 11]\n",
    "with_outlier = normal_data + [100]\n",
    "\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
    "\n",
    "# Normal data\n",
    "stats1 = DescriptiveStats(normal_data)\n",
    "ax1.hist(normal_data, bins=10, edgecolor='black')\n",
    "ax1.axvline(stats1.mean(), color='red', linestyle='--', label=f'Mean: {stats1.mean():.1f}')\n",
    "ax1.axvline(stats1.median(), color='blue', linestyle='--', label=f'Median: {stats1.median():.1f}')\n",
    "ax1.set_title('Normal Data')\n",
    "ax1.legend()\n",
    "\n",
    "# With outlier\n",
    "stats2 = DescriptiveStats(with_outlier)\n",
    "ax2.hist(with_outlier, bins=10, edgecolor='black')\n",
    "ax2.axvline(stats2.mean(), color='red', linestyle='--', label=f'Mean: {stats2.mean():.1f}')\n",
    "ax2.axvline(stats2.median(), color='blue', linestyle='--', label=f'Median: {stats2.median():.1f}')\n",
    "ax2.set_title('With Outlier')\n",
    "ax2.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"Notice how the outlier pulls the mean far from the median!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create dashboard HTML
cat > dashboard.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day 26 Dashboard — Descriptive Statistics Metrics</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --muted: #9ca3af;
            --text: #f3f4f6;
            --accent: #16a34a;
            --accent-2: #f59e0b;
            --danger: #ef4444;
            --card: #0b1220;
            --border: #1f2937;
            --shadow: 0 10px 30px rgba(0,0,0,0.35);
        }

        * { box-sizing: border-box; }
        html, body { height: 100%; }
        body {
            margin: 0;
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
            background: radial-gradient(1200px 600px at 10% -10%, rgba(22,163,74,0.08), transparent 60%),
                        radial-gradient(900px 500px at 110% 10%, rgba(245,158,11,0.08), transparent 60%),
                        var(--bg);
            color: var(--text);
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 20px 80px;
        }

        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 28px;
        }
        .title {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .title h1 {
            margin: 0;
            font-size: 28px;
            letter-spacing: 0.2px;
        }
        .title p {
            margin: 0;
            color: var(--muted);
            font-size: 14px;
        }
        .actions {
            display: flex;
            gap: 10px;
        }
        .btn {
            appearance: none;
            border: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.05));
            color: var(--text);
            padding: 10px 14px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn:hover { transform: translateY(-1px); box-shadow: var(--shadow); }
        .btn-primary { border-color: rgba(22,163,74,0.4); background: linear-gradient(180deg, rgba(22,163,74,0.18), rgba(22,163,74,0.08)); }

        .grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 16px;
        }
        .card {
            grid-column: span 12;
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.1)), var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px;
            box-shadow: var(--shadow);
        }
        .card h3 { margin: 0 0 16px; font-size: 18px; font-weight: 700; }
        .card p { margin: 0; color: var(--muted); font-size: 14px; }

        @media (min-width: 900px) {
            .span-3 { grid-column: span 3; }
            .span-4 { grid-column: span 4; }
            .span-6 { grid-column: span 6; }
            .span-8 { grid-column: span 8; }
            .span-12 { grid-column: span 12; }
        }

        .metric {
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-top: 12px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: 800;
            color: var(--accent);
        }
        .metric-sub {
            color: var(--muted);
            font-size: 13px;
        }

        .scenario-card {
            border-left: 3px solid var(--accent);
            padding-left: 16px;
            margin-top: 16px;
        }
        .scenario-card h4 {
            margin: 0 0 8px;
            font-size: 16px;
            color: var(--text);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 12px;
            margin-top: 12px;
        }
        .stat-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .stat-label {
            font-size: 11px;
            color: var(--muted);
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 20px;
            font-weight: 700;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            border: 1px solid var(--border);
            padding: 6px 10px;
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 12px;
        }
        .badge .dot { width: 8px; height: 8px; border-radius: 999px; background: var(--accent); }

        footer { margin-top: 32px; color: var(--muted); font-size: 12px; text-align: center; }
        code { background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 6px; border: 1px solid var(--border); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title">
                <h1>Day 26 Dashboard</h1>
                <p>Descriptive Statistics — Real-Time Metrics & Demo Results</p>
            </div>
            <div class="actions">
                <button class="btn btn-primary" onclick="refreshMetrics()">🔄 Refresh Metrics</button>
                <button class="btn" onclick="runDemo()">▶️ Run Demo</button>
            </div>
        </header>

        <div class="grid">
            <section class="card span-12">
                <div class="badge"><span class="dot"></span> Live Metrics</div>
                <h3>Overview Statistics</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Total Scenarios</div>
                        <div class="stat-value" id="total-scenarios">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Total Data Points</div>
                        <div class="stat-value" id="total-datapoints">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Outliers Detected</div>
                        <div class="stat-value" id="total-outliers">0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Skewed Distributions</div>
                        <div class="stat-value" id="total-skewed">0</div>
                    </div>
                </div>
            </section>

            <section class="card span-6" id="scenario-1">
                <h3>📱 Scenario 1: User Session Duration</h3>
                <div class="scenario-card">
                    <h4>Statistics</h4>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-label">Mean</div>
                            <div class="stat-value" id="s1-mean">0</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Median</div>
                            <div class="stat-value" id="s1-median">0</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Mode</div>
                            <div class="stat-value" id="s1-mode">-</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Skew Type</div>
                            <div class="stat-value" id="s1-skew" style="font-size: 14px;">-</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Outliers</div>
                            <div class="stat-value" id="s1-outliers">0</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Count</div>
                            <div class="stat-value" id="s1-count">0</div>
                        </div>
                    </div>
                </div>
            </section>

            <section class="card span-6" id="scenario-2">
                <h3>💳 Scenario 2: Transaction Amounts</h3>
                <div class="scenario-card">
                    <h4>Statistics</h4>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-label">Mean</div>
                            <div class="stat-value" id="s2-mean">0</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Median</div>
                            <div class="stat-value" id="s2-median">0</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Mode</div>
                            <div class="stat-value" id="s2-mode">-</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Skew Type</div>
                            <div class="stat-value" id="s2-skew" style="font-size: 14px;">-</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Outliers</div>
                            <div class="stat-value" id="s2-outliers">0</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Count</div>
                            <div class="stat-value" id="s2-count">0</div>
                        </div>
                    </div>
                </div>
            </section>

            <section class="card span-6" id="scenario-3">
                <h3>⚡ Scenario 3: AI Model Latency</h3>
                <div class="scenario-card">
                    <h4>Statistics</h4>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-label">Mean</div>
                            <div class="stat-value" id="s3-mean">0</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Median</div>
                            <div class="stat-value" id="s3-median">0</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Mode</div>
                            <div class="stat-value" id="s3-mode">-</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Skew Type</div>
                            <div class="stat-value" id="s3-skew" style="font-size: 14px;">-</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Outliers</div>
                            <div class="stat-value" id="s3-outliers">0</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Count</div>
                            <div class="stat-value" id="s3-count">0</div>
                        </div>
                    </div>
                </div>
            </section>

            <section class="card span-6">
                <div class="badge"><span class="dot"></span> Status</div>
                <h3>System Status</h3>
                <div class="metric">
                    <div class="metric-label">Last Update</div>
                    <div class="metric-value" id="last-update" style="font-size: 18px;">Never</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Demo Status</div>
                    <div class="metric-value" id="demo-status" style="font-size: 18px; color: var(--muted);">Not Run</div>
                </div>
                <div style="margin-top: 16px;">
                    <button class="btn btn-primary" onclick="refreshMetrics()" style="width: 100%;">🔄 Refresh Now</button>
                </div>
            </section>
        </div>

        <footer>
            Built for the 180‑Day AI/ML Course · Day 26 · Descriptive Statistics Dashboard
        </footer>
    </div>

    <script>
        function updateMetrics(data) {
            // Overview
            document.getElementById('total-scenarios').textContent = data.total_scenarios || 3;
            document.getElementById('total-datapoints').textContent = data.total_datapoints || 0;
            document.getElementById('total-outliers').textContent = data.total_outliers || 0;
            document.getElementById('total-skewed').textContent = data.total_skewed || 0;

            // Scenario 1
            if (data.scenario1) {
                document.getElementById('s1-mean').textContent = data.scenario1.mean || 0;
                document.getElementById('s1-median').textContent = data.scenario1.median || 0;
                document.getElementById('s1-mode').textContent = data.scenario1.mode || '-';
                document.getElementById('s1-skew').textContent = data.scenario1.skew_type || '-';
                document.getElementById('s1-outliers').textContent = data.scenario1.outliers || 0;
                document.getElementById('s1-count').textContent = data.scenario1.count || 0;
            }

            // Scenario 2
            if (data.scenario2) {
                document.getElementById('s2-mean').textContent = data.scenario2.mean || 0;
                document.getElementById('s2-median').textContent = data.scenario2.median || 0;
                document.getElementById('s2-mode').textContent = data.scenario2.mode || '-';
                document.getElementById('s2-skew').textContent = data.scenario2.skew_type || '-';
                document.getElementById('s2-outliers').textContent = data.scenario2.outliers || 0;
                document.getElementById('s2-count').textContent = data.scenario2.count || 0;
            }

            // Scenario 3
            if (data.scenario3) {
                document.getElementById('s3-mean').textContent = data.scenario3.mean || 0;
                document.getElementById('s3-median').textContent = data.scenario3.median || 0;
                document.getElementById('s3-mode').textContent = data.scenario3.mode || '-';
                document.getElementById('s3-skew').textContent = data.scenario3.skew_type || '-';
                document.getElementById('s3-outliers').textContent = data.scenario3.outliers || 0;
                document.getElementById('s3-count').textContent = data.scenario3.count || 0;
            }

            // Status
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            document.getElementById('demo-status').textContent = data.demo_status || 'Not Run';
            document.getElementById('demo-status').style.color = data.demo_status === 'Running' ? 'var(--accent)' : 'var(--muted)';
        }

        function refreshMetrics() {
            fetch('/api/metrics')
                .then(res => res.json())
                .then(data => updateMetrics(data))
                .catch(err => {
                    console.error('Error fetching metrics:', err);
                    alert('Could not fetch metrics. Make sure the server is running and demo has been executed.');
                });
        }

        function runDemo() {
            fetch('/api/run-demo', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    alert('Demo executed successfully! Refreshing metrics...');
                    refreshMetrics();
                })
                .catch(err => {
                    console.error('Error running demo:', err);
                    alert('Could not run demo. Check server logs.');
                });
        }

        // Auto-refresh every 5 seconds
        setInterval(refreshMetrics, 5000);
        
        // Initial load
        refreshMetrics();
    </script>
</body>
</html>
EOF

# Create Flask app for dashboard server
cat > app.py << 'EOF'
#!/usr/bin/env python3
"""
Dashboard server for Day 26: Descriptive Statistics
Serves metrics and runs demo calculations
"""

import json
import sys
import os
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

# Add current directory to path to import lesson_code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from lesson_code import DescriptiveStats, DataProfiler, demo_real_world_scenarios
except ImportError as e:
    print(f"Error importing lesson_code: {e}")
    print("Make sure you have run: python lesson_code.py first")
    sys.exit(1)

app = Flask(__name__, static_folder='.')
CORS(app)

# Store last computed metrics
last_metrics = {
    'total_scenarios': 3,
    'total_datapoints': 0,
    'total_outliers': 0,
    'total_skewed': 0,
    'scenario1': None,
    'scenario2': None,
    'scenario3': None,
    'demo_status': 'Not Run',
    'last_run': None
}

def compute_metrics():
    """Compute statistics for all three scenarios"""
    global last_metrics
    
    # Scenario 1: User Session Duration
    session_data = [2, 3, 3, 4, 5, 3, 2, 120, 4, 3, 150, 3, 4, 3, 2, 5, 4, 3]
    profiler1 = DataProfiler(session_data, "session_duration_minutes")
    stats1 = profiler1.stats
    skew1 = profiler1.detect_skew()
    outliers1 = profiler1.detect_outliers_iqr()
    
    # Scenario 2: Transaction Amounts
    transaction_data = [45, 67, 52, 48, 51, 49, 53, 47, 50, 48, 
                       52, 3500, 48, 51, 49, 46, 52, 48]
    profiler2 = DataProfiler(transaction_data, "transaction_amount_usd")
    stats2 = profiler2.stats
    skew2 = profiler2.detect_skew()
    outliers2 = profiler2.detect_outliers_iqr()
    
    # Scenario 3: Model Latency
    latency_data = [45, 47, 46, 48, 44, 46, 47, 45, 49, 46, 
                   48, 45, 47, 280, 46, 47, 45, 48]
    profiler3 = DataProfiler(latency_data, "response_latency_ms")
    stats3 = profiler3.stats
    skew3 = profiler3.detect_skew()
    outliers3 = profiler3.detect_outliers_iqr()
    
    # Update metrics
    last_metrics['scenario1'] = {
        'mean': round(stats1.mean(), 2),
        'median': round(stats1.median(), 2),
        'mode': stats1.mode() if stats1.mode() else '-',
        'skew_type': skew1['skew_type'],
        'outliers': outliers1['outlier_count'],
        'count': stats1.n
    }
    
    last_metrics['scenario2'] = {
        'mean': round(stats2.mean(), 2),
        'median': round(stats2.median(), 2),
        'mode': stats2.mode() if stats2.mode() else '-',
        'skew_type': skew2['skew_type'],
        'outliers': outliers2['outlier_count'],
        'count': stats2.n
    }
    
    last_metrics['scenario3'] = {
        'mean': round(stats3.mean(), 2),
        'median': round(stats3.median(), 2),
        'mode': stats3.mode() if stats3.mode() else '-',
        'skew_type': skew3['skew_type'],
        'outliers': outliers3['outlier_count'],
        'count': stats3.n
    }
    
    # Compute totals
    last_metrics['total_datapoints'] = stats1.n + stats2.n + stats3.n
    last_metrics['total_outliers'] = outliers1['outlier_count'] + outliers2['outlier_count'] + outliers3['outlier_count']
    last_metrics['total_skewed'] = sum(1 for s in [skew1, skew2, skew3] if s['skew_type'] != 'symmetric')
    
    last_metrics['last_run'] = json.dumps({'timestamp': 'now'})
    
    return last_metrics

@app.route('/')
def index():
    """Serve dashboard HTML"""
    return send_from_directory('.', 'dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """Return current metrics"""
    # Recompute to ensure fresh data
    compute_metrics()
    return jsonify(last_metrics)

@app.route('/api/run-demo', methods=['POST'])
def run_demo():
    """Execute demo and update metrics"""
    try:
        # Run the demo (prints to console)
        demo_real_world_scenarios()
        
        # Compute and update metrics
        compute_metrics()
        last_metrics['demo_status'] = 'Running'
        
        return jsonify({
            'status': 'success',
            'message': 'Demo executed successfully',
            'metrics': last_metrics
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Day 26 Dashboard Server...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("📈 API endpoint: http://localhost:5000/api/metrics")
    print("\n⚠️  Note: Run 'python lesson_code.py' first to ensure all imports work")
    print("🔄 Press Ctrl+C to stop the server\n")
    
    # Compute initial metrics
    try:
        compute_metrics()
        print("✅ Initial metrics computed successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not compute initial metrics: {e}")
        print("   Run 'python lesson_code.py' to fix this")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
EOF

# Create startup script
cat > start_dashboard.sh << 'EOF'
#!/bin/bash

# Day 26 Dashboard Startup Script
# This script checks for duplicate services and starts the dashboard server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "🔍 Checking for duplicate services..."

# Check if Flask app is already running
if pgrep -f "python.*app.py" > /dev/null; then
    echo "⚠️  Warning: Dashboard server appears to be running already"
    echo "   PIDs: $(pgrep -f 'python.*app.py' | tr '\n' ' ')"
    read -p "   Kill existing processes and start new? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "python.*app.py"
        sleep 2
        echo "✅ Killed existing processes"
    else
        echo "❌ Aborted. Dashboard may already be running on http://localhost:5000"
        exit 1
    fi
fi

# Check if port 5000 is in use
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Warning: Port 5000 is already in use"
    echo "   Another service may be using this port"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./env_setup.sh first"
    exit 1
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Activated virtual environment"
else
    echo "❌ Virtual environment activation script not found"
    exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Run setup.sh first"
    exit 1
fi

# Check if dashboard.html exists
if [ ! -f "dashboard.html" ]; then
    echo "❌ dashboard.html not found. Run setup.sh first"
    exit 1
fi

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask not found. Installing..."
    pip install flask flask-cors
fi

echo ""
echo "🚀 Starting dashboard server..."
echo "📊 Dashboard will be available at: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop"
echo ""

# Start the server
python app.py
EOF

chmod +x start_dashboard.sh

echo ""
echo "✅ All files generated successfully!"
echo ""
echo "📁 Files created:"
echo "   - env_setup.sh (environment setup script)"
echo "   - requirements.txt (Python dependencies)"
echo "   - lesson_code.py (main lesson implementation)"
echo "   - test_lesson.py (test suite)"
echo "   - README.md (quick start guide)"
echo "   - descriptive_statistics.ipynb (Jupyter notebook)"
echo "   - dashboard.html (metrics dashboard)"
echo "   - app.py (dashboard server)"
echo "   - start_dashboard.sh (startup script)"
echo ""
echo "🚀 Next steps:"
echo "   1. chmod +x env_setup.sh && ./env_setup.sh"
echo "   2. source venv/bin/activate"
echo "   3. python lesson_code.py"
echo "   4. python test_lesson.py"
echo "   5. python app.py (or ./start_dashboard.sh) to start dashboard"
echo ""
echo "Happy learning! 🎓"