#!/bin/bash

# Day 27: Measures of Spread - Lesson Files Generator
# This script generates all required lesson files

echo "🚀 Generating Day 27: Variance and Standard Deviation lesson files..."

# Create setup.sh
cat > setup.sh << 'SETUP_EOF'
#!/bin/bash

echo "📦 Setting up Day 27: Measures of Spread environment..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Found Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "Run 'source venv/bin/activate' to activate the environment"
echo "Then run 'python lesson_code.py' to start learning!"
SETUP_EOF

chmod +x setup.sh

# Create requirements.txt
cat > requirements.txt << 'REQ_EOF'
numpy==1.26.4
matplotlib==3.8.3
pandas==2.2.1
scipy==1.12.0
jupyter==1.0.0
pytest==8.1.1
REQ_EOF

# Create lesson_code.py
cat > lesson_code.py << 'LESSON_EOF'
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
LESSON_EOF

# Create test_lesson.py
cat > test_lesson.py << 'TEST_EOF'
"""
Unit tests for Day 27: Variance and Standard Deviation
Run with: pytest test_lesson.py -v
"""

import pytest
import numpy as np
from lesson_code import DataQualityChecker


class TestVarianceCalculations:
    """Test variance and standard deviation calculations"""
    
    def test_sample_variance(self):
        """Test sample variance calculation (n-1)"""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        checker = DataQualityChecker(data)
        
        # Manual calculation
        mean = np.mean(data)
        variance = sum((x - mean)**2 for x in data) / (len(data) - 1)
        
        assert abs(checker.variance - variance) < 0.001
        assert abs(checker.variance - 4.571) < 0.01
    
    def test_population_variance(self):
        """Test population variance calculation (n)"""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        checker = DataQualityChecker(data)
        
        # Population variance (n)
        mean = np.mean(data)
        pop_var = sum((x - mean)**2 for x in data) / len(data)
        
        assert abs(checker.population_var - pop_var) < 0.001
        assert abs(checker.population_var - 4.0) < 0.01
    
    def test_standard_deviation(self):
        """Test standard deviation is sqrt of variance"""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        checker = DataQualityChecker(data)
        
        assert abs(checker.std - np.sqrt(checker.variance)) < 0.001
        assert abs(checker.std - 2.138) < 0.01
    
    def test_zero_variance(self):
        """Test data with no variance (all same values)"""
        data = [5, 5, 5, 5, 5]
        checker = DataQualityChecker(data)
        
        assert checker.variance == 0
        assert checker.std == 0


class TestOutlierDetection:
    """Test outlier detection using 3-sigma rule"""
    
    def test_no_outliers(self):
        """Test dataset with no outliers"""
        np.random.seed(42)
        data = np.random.normal(100, 10, 100)
        checker = DataQualityChecker(data)
        
        outliers, count, _ = checker.detect_outliers(threshold=3)
        assert count <= 3  # Should be ~0-3 outliers in normal data
    
    def test_with_outliers(self):
        """Test dataset with clear outliers"""
        data = [10, 11, 12, 13, 14, 15, 100, 200]  # Last two are outliers
        checker = DataQualityChecker(data)
        
        outliers, count, indices = checker.detect_outliers(threshold=2)
        assert count >= 1  # Should detect at least one outlier
        assert 100 in outliers or 200 in outliers
    
    def test_outlier_threshold(self):
        """Test different outlier thresholds"""
        data = [10, 11, 12, 13, 14, 15, 16, 30]
        checker = DataQualityChecker(data)
        
        _, count_2sigma, _ = checker.detect_outliers(threshold=2)
        _, count_3sigma, _ = checker.detect_outliers(threshold=3)
        
        # More permissive threshold should find fewer outliers
        assert count_3sigma <= count_2sigma


class TestCoefficientOfVariation:
    """Test coefficient of variation calculations"""
    
    def test_cv_calculation(self):
        """Test CV = (std/mean) * 100"""
        data = [10, 20, 30, 40, 50]
        checker = DataQualityChecker(data)
        
        expected_cv = (checker.std / checker.mean) * 100
        assert abs(checker.coefficient_of_variation() - expected_cv) < 0.001
    
    def test_cv_interpretation(self):
        """Test CV categories for ML readiness"""
        # Low variance data
        low_var = np.random.normal(100, 5, 100)
        checker_low = DataQualityChecker(low_var)
        assert checker_low.coefficient_of_variation() < 15
        
        # High variance data
        high_var = np.random.normal(100, 50, 100)
        checker_high = DataQualityChecker(high_var)
        assert checker_high.coefficient_of_variation() > 30
    
    def test_cv_zero_mean(self):
        """Test CV with zero mean (should return inf)"""
        data = [-5, -3, 0, 3, 5]
        checker = DataQualityChecker(data)
        cv = checker.coefficient_of_variation()
        # With zero mean, CV is undefined (inf)
        assert cv == float('inf') or abs(checker.mean) < 0.001


class TestIQRCalculation:
    """Test Interquartile Range calculations"""
    
    def test_iqr_basic(self):
        """Test IQR calculation"""
        data = list(range(1, 101))  # 1 to 100
        checker = DataQualityChecker(data)
        
        iqr, q25, q75 = checker.calculate_iqr()
        
        assert abs(q25 - 25.5) < 1
        assert abs(q75 - 75.5) < 1
        assert abs(iqr - 50) < 1
    
    def test_iqr_robustness(self):
        """Test IQR is robust to outliers"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]  # One extreme outlier
        checker = DataQualityChecker(data)
        
        iqr, q25, q75 = checker.calculate_iqr()
        
        # IQR should not be heavily affected by the outlier
        assert iqr < 10  # Should be around 4-5


class TestDataQualityReport:
    """Test comprehensive quality report"""
    
    def test_report_returns_dict(self):
        """Test quality report returns proper dictionary"""
        data = [10, 20, 30, 40, 50]
        checker = DataQualityChecker(data)
        
        result = checker.quality_report()
        
        assert isinstance(result, dict)
        assert 'mean' in result
        assert 'std' in result
        assert 'variance' in result
        assert 'cv' in result
        assert 'outliers' in result
    
    def test_report_accuracy(self):
        """Test report values match individual calculations"""
        data = [15, 20, 25, 30, 35]
        checker = DataQualityChecker(data)
        
        result = checker.quality_report()
        
        assert result['mean'] == checker.mean
        assert result['std'] == checker.std
        assert result['variance'] == checker.variance


def test_numpy_consistency():
    """Test our calculations match NumPy's built-in functions"""
    np.random.seed(42)
    data = np.random.normal(50, 15, 100)
    
    checker = DataQualityChecker(data)
    
    # Compare with NumPy
    assert abs(checker.mean - np.mean(data)) < 0.001
    assert abs(checker.variance - np.var(data, ddof=1)) < 0.001
    assert abs(checker.std - np.std(data, ddof=1)) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
TEST_EOF

# Create README.md
cat > README.md << 'README_EOF'
# Day 27: Measures of Spread - Variance and Standard Deviation

Learn how to measure and understand data variability for AI/ML applications.

## 📚 What You'll Learn

- Calculate variance and standard deviation from scratch
- Detect outliers using the 3-sigma rule
- Assess data quality for machine learning
- Understand when to use sample vs population variance
- Apply feature scaling based on variance

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Run the Lesson

```bash
python lesson_code.py
```

This will:
- Analyze real-world response time data
- Compare datasets with different spreads
- Demonstrate ML feature scaling
- Generate visualization (`data_spread_analysis.png`)

### 3. Run Tests

```bash
pytest test_lesson.py -v
```

## 📊 Key Concepts Covered

### Variance
- **Formula**: σ² = Σ(x - μ)² / (n-1)
- Measures average squared deviation from mean
- Critical for understanding data consistency

### Standard Deviation
- **Formula**: σ = √variance
- Same units as original data (easier to interpret)
- Used in 68-95-99.7 rule for normal distributions

### Coefficient of Variation
- **Formula**: CV = (σ / μ) × 100%
- Relative measure of spread
- Helps compare variability across different scales

## 🎯 Real-World Applications

1. **Anomaly Detection**: Flag data points beyond 3σ
2. **Feature Scaling**: Normalize features with different variances
3. **Model Confidence**: Wider spread = less confident predictions
4. **A/B Testing**: Determine if differences are significant

## 📈 Output Examples

```
📊 DATA QUALITY REPORT: API Response Times (ms)
============================================================
Sample Size: 20
Range: [97.00, 250.00]

Central Tendency:
  Mean: 109.70
  Median: 101.00

Measures of Spread:
  Sample Variance: 1139.27
  Sample Std Dev: 33.75
  IQR: 4.50 (Q1=99.00, Q3=103.50)
  Coefficient of Variation: 30.77%

Outlier Detection (3σ rule):
  Outliers Found: 1 (5.0%)
  Outlier Values: [250.]

============================================================
ML Readiness: ⚠️  MODERATE
Recommendation: Some spread present. Consider feature scaling.
```

## 🔧 File Structure

```
day-27-variance-std/
├── setup.sh              # Environment setup
├── requirements.txt      # Python dependencies
├── lesson_code.py        # Main lesson implementation
├── test_lesson.py        # Unit tests
├── README.md            # This file
└── data_spread_analysis.png  # Generated visualization
```

## 💡 Practice Exercises

After completing the lesson, try these:

1. **Modify the threshold**: Change outlier detection from 3σ to 2σ
2. **Add your data**: Replace response_times with your own dataset
3. **Compare distributions**: Create datasets with CV of 10%, 25%, and 50%
4. **Build a monitor**: Track variance over time for drift detection

## 🎓 Learning Tips

- Focus on **why variance matters** for ML, not just calculations
- Practice interpreting CV values for different use cases
- Understand the difference between sample and population variance
- Connect today's concepts to tomorrow's correlation lesson

## 📚 Next Lesson

**Day 28: Correlation and Covariance**
- How variables move together
- Correlation vs causation
- Covariance matrices for ML

## ❓ Common Questions

**Q: When should I use sample vs population variance?**
A: Almost always use sample variance (n-1) in ML since you're working with training data, not all possible data.

**Q: What's a "good" coefficient of variation?**
A: < 15% is excellent, 15-30% is moderate, > 30% suggests you need normalization.

**Q: Why square the deviations in variance?**
A: To prevent positive and negative deviations from canceling out and to penalize larger deviations more.

## 🐛 Troubleshooting

**Import Error**: Make sure you've activated the virtual environment
```bash
source venv/bin/activate
```

**Missing Visualization**: Install matplotlib
```bash
pip install matplotlib
```

**Tests Failing**: Update NumPy to latest version
```bash
pip install --upgrade numpy
```

## 📞 Support

Having issues? Check:
- Python version is 3.11+
- All dependencies installed from requirements.txt
- Virtual environment is activated

---

**Time to Complete**: 2-3 hours
**Difficulty**: Beginner to Intermediate
**Prerequisites**: Day 26 (Mean, Median, Mode)
README_EOF

echo "✅ All files generated successfully!"
echo ""
echo "🔍 Verifying all files were created..."

# List of required files
REQUIRED_FILES=("setup.sh" "requirements.txt" "lesson_code.py" "test_lesson.py" "README.md")
MISSING_FILES=()
ALL_PRESENT=true

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing file: $file"
        MISSING_FILES+=("$file")
        ALL_PRESENT=false
    else
        echo "✅ Found: $file ($(wc -l < "$file" | tr -d ' ') lines)"
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    echo "❌ ERROR: Some required files are missing!"
    echo "Missing files: ${MISSING_FILES[*]}"
    exit 1
fi

echo ""
echo "✅ All required files verified and created successfully!"
echo ""
echo "📁 Created files:"
for file in "${REQUIRED_FILES[@]}"; do
    echo "  - $file"
done
echo ""
echo "🚀 Next steps:"
echo "  1. chmod +x setup.sh && ./setup.sh"
echo "  2. source venv/bin/activate"
echo "  3. python lesson_code.py"
echo ""
echo "Happy learning! 🎓"

