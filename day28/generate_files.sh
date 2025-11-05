#!/bin/bash

# Day 28: Correlation and Covariance - Lesson Files Generator
# This script generates all required lesson files

echo "🚀 Generating Day 28: Correlation and Covariance lesson files..."

# Create setup.sh
cat > setup.sh << 'SETUP_EOF'
#!/bin/bash

echo "📦 Setting up Day 28: Correlation and Covariance environment..."

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

# Create requirements.txt with Flask for dashboard
cat > requirements.txt << 'REQ_EOF'
numpy==1.26.3
matplotlib==3.8.2
pandas==2.2.0
seaborn==0.13.1
scipy==1.12.0
jupyter==1.0.0
pytest==7.4.4
flask==3.0.0
REQ_EOF

# Create lesson_code.py (using the content from the original setup.sh)
cat > lesson_code.py << 'LESSON_EOF'
"""
Day 28: Correlation and Covariance
Feature Relationship Analyzer for AI/ML Systems
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple, List, Dict

class FeatureAnalyzer:
    """Analyzes relationships between features in datasets for ML preparation."""
    
    def __init__(self, data: np.ndarray, feature_names: List[str] = None):
        """
        Initialize the analyzer with data.
        
        Args:
            data: 2D numpy array where rows are samples, columns are features
            feature_names: Optional list of feature names
        """
        self.data = data
        self.n_samples, self.n_features = data.shape
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(self.n_features)]
        
        # Calculate matrices
        self.covariance_matrix = None
        self.correlation_matrix = None
        
    def calculate_covariance_manual(self) -> np.ndarray:
        """
        Calculate covariance matrix manually to understand the math.
        
        Covariance measures how two variables change together:
        Cov(X,Y) = Σ[(Xi - X̄)(Yi - Ȳ)] / (n-1)
        """
        # Center the data (subtract mean from each feature)
        centered_data = self.data - np.mean(self.data, axis=0)
        
        # Calculate covariance matrix
        # This is the dot product of centered data with itself, divided by (n-1)
        covariance = np.dot(centered_data.T, centered_data) / (self.n_samples - 1)
        
        self.covariance_matrix = covariance
        return covariance
    
    def calculate_correlation_manual(self) -> np.ndarray:
        """
        Calculate correlation matrix manually.
        
        Correlation is standardized covariance:
        Corr(X,Y) = Cov(X,Y) / (σX × σY)
        
        This always gives values between -1 and +1.
        """
        if self.covariance_matrix is None:
            self.calculate_covariance_manual()
        
        # Get standard deviations (square root of diagonal elements)
        std_devs = np.sqrt(np.diag(self.covariance_matrix))
        
        # Create matrix of standard deviation products
        std_matrix = np.outer(std_devs, std_devs)
        
        # Correlation = Covariance / (std_dev_x * std_dev_y)
        correlation = self.covariance_matrix / std_matrix
        
        self.correlation_matrix = correlation
        return correlation
    
    def find_highly_correlated_features(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Find pairs of features with high correlation.
        
        In ML, highly correlated features (>0.8 or >0.9) often provide
        redundant information and can be removed.
        
        Args:
            threshold: Correlation threshold (default 0.8)
            
        Returns:
            List of tuples: (feature1, feature2, correlation_value)
        """
        if self.correlation_matrix is None:
            self.calculate_correlation_manual()
        
        high_corr_pairs = []
        
        # Iterate through upper triangle only (avoid duplicates and diagonal)
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                corr_value = abs(self.correlation_matrix[i, j])
                if corr_value >= threshold:
                    high_corr_pairs.append((
                        self.feature_names[i],
                        self.feature_names[j],
                        self.correlation_matrix[i, j]
                    ))
        
        # Sort by absolute correlation value (descending)
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_corr_pairs
    
    def suggest_features_to_remove(self, threshold: float = 0.9) -> Dict[str, List[str]]:
        """
        Suggest which features to remove based on high correlation.
        
        Strategy: For each highly correlated pair, keep the feature
        that has lower average correlation with all other features.
        
        Args:
            threshold: Correlation threshold for redundancy (default 0.9)
            
        Returns:
            Dictionary with 'keep' and 'remove' lists
        """
        high_corr_pairs = self.find_highly_correlated_features(threshold)
        
        if not high_corr_pairs:
            return {'keep': self.feature_names, 'remove': []}
        
        # Calculate average correlation for each feature
        avg_correlations = {}
        for i, feature in enumerate(self.feature_names):
            # Average of absolute correlations with other features
            other_corrs = np.abs(self.correlation_matrix[i, :])
            other_corrs = np.delete(other_corrs, i)  # Remove self-correlation
            avg_correlations[feature] = np.mean(other_corrs)
        
        # Decide which features to remove
        features_to_remove = set()
        
        for feat1, feat2, corr in high_corr_pairs:
            if feat1 not in features_to_remove and feat2 not in features_to_remove:
                # Remove the one with higher average correlation
                if avg_correlations[feat1] > avg_correlations[feat2]:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
        
        features_to_keep = [f for f in self.feature_names if f not in features_to_remove]
        
        return {
            'keep': features_to_keep,
            'remove': list(features_to_remove)
        }
    
    def visualize_correlation_matrix(self, save_path: str = None):
        """
        Create a heatmap visualization of the correlation matrix.
        
        This is standard practice in ML data exploration.
        Red = strong positive correlation
        Blue = strong negative correlation
        White = no correlation
        """
        if self.correlation_matrix is None:
            self.calculate_correlation_manual()
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            self.correlation_matrix,
            annot=True,  # Show correlation values
            fmt='.2f',   # Format to 2 decimal places
            cmap='RdBu_r',  # Red-Blue colormap (reversed)
            center=0,    # Center colormap at 0
            square=True,  # Make cells square
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        plt.title('Feature Correlation Matrix\n(Red = Positive, Blue = Negative)', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation heatmap to {save_path}")
        
        plt.close()  # Close to avoid display issues in headless mode
    
    def visualize_scatter_matrix(self, max_features: int = 5, save_path: str = None):
        """
        Create scatter plots for feature pairs to visualize relationships.
        
        Args:
            max_features: Maximum number of features to include (for readability)
            save_path: Optional path to save the figure
        """
        # Limit features for readability
        n_plot = min(max_features, self.n_features)
        plot_data = self.data[:, :n_plot]
        plot_names = self.feature_names[:n_plot]
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(plot_data, columns=plot_names)
        
        # Create scatter matrix
        fig, axes = plt.subplots(n_plot, n_plot, figsize=(12, 12))
        
        for i in range(n_plot):
            for j in range(n_plot):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: show histogram
                    ax.hist(df.iloc[:, i], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_ylabel('Frequency')
                else:
                    # Off-diagonal: show scatter plot
                    ax.scatter(df.iloc[:, j], df.iloc[:, i], alpha=0.5, s=20)
                    
                    # Add correlation value
                    corr = self.correlation_matrix[i, j]
                    ax.text(0.05, 0.95, f'r={corr:.2f}', 
                           transform=ax.transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Labels only on edges
                if i == n_plot - 1:
                    ax.set_xlabel(plot_names[j])
                if j == 0:
                    ax.set_ylabel(plot_names[i])
        
        plt.suptitle('Feature Relationship Scatter Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved scatter matrix to {save_path}")
        
        plt.close()  # Close to avoid display issues in headless mode
    
    def generate_report(self) -> str:
        """
        Generate a text report summarizing the analysis.
        
        This is what you'd include in model documentation.
        """
        self.calculate_correlation_manual()
        
        report = []
        report.append("=" * 60)
        report.append("FEATURE RELATIONSHIP ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nDataset Shape: {self.n_samples} samples × {self.n_features} features")
        report.append(f"Features: {', '.join(self.feature_names)}\n")
        
        # Summary statistics
        report.append("CORRELATION SUMMARY:")
        report.append("-" * 40)
        
        # Get upper triangle correlations (excluding diagonal)
        upper_triangle = []
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                upper_triangle.append(abs(self.correlation_matrix[i, j]))
        
        report.append(f"Average correlation: {np.mean(upper_triangle):.3f}")
        report.append(f"Maximum correlation: {np.max(upper_triangle):.3f}")
        report.append(f"Minimum correlation: {np.min(upper_triangle):.3f}\n")
        
        # High correlation pairs
        high_corr = self.find_highly_correlated_features(threshold=0.7)
        
        if high_corr:
            report.append("HIGH CORRELATION PAIRS (|r| >= 0.7):")
            report.append("-" * 40)
            for feat1, feat2, corr in high_corr:
                report.append(f"  {feat1} ↔ {feat2}: {corr:+.3f}")
        else:
            report.append("No highly correlated feature pairs found (threshold: 0.7)")
        
        # Feature removal suggestions
        report.append("\n" + "=" * 60)
        suggestions = self.suggest_features_to_remove(threshold=0.9)
        
        if suggestions['remove']:
            report.append("FEATURE REDUCTION SUGGESTIONS:")
            report.append("-" * 40)
            report.append(f"Recommended to KEEP: {', '.join(suggestions['keep'])}")
            report.append(f"Recommended to REMOVE: {', '.join(suggestions['remove'])}")
            report.append(f"\nRemoving {len(suggestions['remove'])} features would reduce")
            report.append(f"redundancy while preserving unique information.")
        else:
            report.append("All features provide unique information (no removal needed).")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def generate_sample_data() -> Tuple[np.ndarray, List[str]]:
    """
    Generate realistic sample data simulating user engagement metrics
    for a content recommendation system.
    
    This simulates data you'd collect from a production AI agent.
    """
    np.random.seed(42)
    n_samples = 200
    
    # Feature 1: Time spent on page (seconds)
    time_spent = np.random.normal(120, 40, n_samples)
    
    # Feature 2: Scroll depth (%) - correlated with time spent
    scroll_depth = 0.6 * time_spent + np.random.normal(0, 10, n_samples)
    scroll_depth = np.clip(scroll_depth, 0, 100)
    
    # Feature 3: Number of clicks - somewhat correlated with time
    num_clicks = 0.03 * time_spent + np.random.normal(0, 1, n_samples)
    num_clicks = np.clip(num_clicks, 0, 15)
    
    # Feature 4: Return visits - independent
    return_visits = np.random.poisson(3, n_samples)
    
    # Feature 5: Social shares - weakly correlated with engagement
    social_shares = 0.01 * time_spent + 0.2 * num_clicks + np.random.exponential(0.5, n_samples)
    
    # Combine into dataset
    data = np.column_stack([
        time_spent,
        scroll_depth,
        num_clicks,
        return_visits,
        social_shares
    ])
    
    feature_names = [
        'Time_Spent',
        'Scroll_Depth',
        'Num_Clicks',
        'Return_Visits',
        'Social_Shares'
    ]
    
    return data, feature_names


def main():
    """Main execution function."""
    
    print("\n" + "=" * 60)
    print("Day 28: Correlation and Covariance")
    print("Feature Relationship Analyzer")
    print("=" * 60 + "\n")
    
    # Generate sample data
    print("Generating sample user engagement data...")
    data, feature_names = generate_sample_data()
    print(f"Created dataset: {data.shape[0]} samples × {data.shape[1]} features\n")
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer(data, feature_names)
    
    # Calculate matrices
    print("Calculating covariance matrix...")
    cov_matrix = analyzer.calculate_covariance_manual()
    
    print("Calculating correlation matrix...")
    corr_matrix = analyzer.calculate_correlation_manual()
    
    # Verify against NumPy (quality check)
    print("\nVerifying calculations against NumPy...")
    np_cov = np.cov(data, rowvar=False)
    np_corr = np.corrcoef(data, rowvar=False)
    
    cov_match = np.allclose(cov_matrix, np_cov)
    corr_match = np.allclose(corr_matrix, np_corr)
    
    print(f"  Covariance matches NumPy: {'✓' if cov_match else '✗'}")
    print(f"  Correlation matches NumPy: {'✓' if corr_match else '✗'}")
    
    # Generate and print report
    print("\n")
    report = analyzer.generate_report()
    print(report)
    
    # Create visualizations
    print("\nGenerating visualizations...\n")
    
    print("1. Creating correlation heatmap...")
    analyzer.visualize_correlation_matrix(save_path='correlation_heatmap.png')
    
    print("2. Creating scatter matrix...")
    analyzer.visualize_scatter_matrix(max_features=4, save_path='scatter_matrix.png')
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  • Correlation ranges from -1 to +1")
    print("  • Values near ±1 indicate strong relationships")
    print("  • Values near 0 indicate independence")
    print("  • High correlation (>0.9) suggests feature redundancy")
    print("  • Use correlation analysis before training ML models")
    print("\nNext: Day 29 - The Central Limit Theorem")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
LESSON_EOF

# Create test_lesson.py (using the content from original)
cat > test_lesson.py << 'TEST_EOF'
"""
Tests for Day 28: Correlation and Covariance
Run with: pytest test_lesson.py
"""

import pytest
import numpy as np
from lesson_code import FeatureAnalyzer, generate_sample_data


class TestFeatureAnalyzer:
    """Test suite for FeatureAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create simple test data."""
        np.random.seed(42)
        # Create 3 features with known relationships
        x = np.random.normal(0, 1, 100)
        y = 2 * x + np.random.normal(0, 0.1, 100)  # Strong positive correlation
        z = np.random.normal(0, 1, 100)  # Independent
        
        data = np.column_stack([x, y, z])
        feature_names = ['X', 'Y', 'Z']
        
        return data, feature_names
    
    def test_initialization(self, sample_data):
        """Test analyzer initialization."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        
        assert analyzer.n_samples == 100
        assert analyzer.n_features == 3
        assert analyzer.feature_names == names
    
    def test_covariance_calculation(self, sample_data):
        """Test covariance matrix calculation."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        
        # Calculate using our method
        cov_manual = analyzer.calculate_covariance_manual()
        
        # Compare with NumPy
        cov_numpy = np.cov(data, rowvar=False)
        
        assert np.allclose(cov_manual, cov_numpy), "Covariance calculation doesn't match NumPy"
        
        # Check shape
        assert cov_manual.shape == (3, 3)
        
        # Check symmetry
        assert np.allclose(cov_manual, cov_manual.T), "Covariance matrix should be symmetric"
    
    def test_correlation_calculation(self, sample_data):
        """Test correlation matrix calculation."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        
        # Calculate using our method
        corr_manual = analyzer.calculate_correlation_manual()
        
        # Compare with NumPy
        corr_numpy = np.corrcoef(data, rowvar=False)
        
        assert np.allclose(corr_manual, corr_numpy), "Correlation calculation doesn't match NumPy"
        
        # Check diagonal is all 1s (feature correlates perfectly with itself)
        assert np.allclose(np.diag(corr_manual), 1.0), "Diagonal should be 1.0"
        
        # Check all values are between -1 and 1
        assert np.all(corr_manual >= -1.0) and np.all(corr_manual <= 1.0), \
            "Correlation values should be between -1 and 1"
    
    def test_high_correlation_detection(self, sample_data):
        """Test finding highly correlated features."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        analyzer.calculate_correlation_manual()
        
        # X and Y should be highly correlated (we created them that way)
        high_corr = analyzer.find_highly_correlated_features(threshold=0.8)
        
        # Should find at least one pair
        assert len(high_corr) > 0, "Should detect high correlation between X and Y"
        
        # Check format
        for feat1, feat2, corr in high_corr:
            assert isinstance(feat1, str)
            assert isinstance(feat2, str)
            assert abs(corr) >= 0.8
    
    def test_feature_removal_suggestions(self, sample_data):
        """Test feature removal suggestions."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        analyzer.calculate_correlation_manual()
        
        suggestions = analyzer.suggest_features_to_remove(threshold=0.95)
        
        # Check return format
        assert 'keep' in suggestions
        assert 'remove' in suggestions
        assert isinstance(suggestions['keep'], list)
        assert isinstance(suggestions['remove'], list)
        
        # Total should equal original features
        total_features = len(suggestions['keep']) + len(suggestions['remove'])
        assert total_features == 3
    
    def test_report_generation(self, sample_data):
        """Test report generation."""
        data, names = sample_data
        analyzer = FeatureAnalyzer(data, names)
        
        report = analyzer.generate_report()
        
        # Check report is non-empty string
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Check key sections are present
        assert "FEATURE RELATIONSHIP ANALYSIS REPORT" in report
        assert "CORRELATION SUMMARY" in report


class TestSampleDataGeneration:
    """Test sample data generation."""
    
    def test_generate_sample_data(self):
        """Test that sample data generation works."""
        data, feature_names = generate_sample_data()
        
        # Check shape
        assert data.shape == (200, 5), "Should have 200 samples and 5 features"
        
        # Check feature names
        assert len(feature_names) == 5
        assert all(isinstance(name, str) for name in feature_names)
        
        # Check data is numeric
        assert np.isfinite(data).all(), "All data should be finite"
        
        # Check reasonable value ranges
        time_spent = data[:, 0]
        assert time_spent.min() > 0, "Time spent should be positive"
        
        scroll_depth = data[:, 1]
        assert 0 <= scroll_depth.min() and scroll_depth.max() <= 100, \
            "Scroll depth should be between 0 and 100"


class TestCorrelationProperties:
    """Test mathematical properties of correlation."""
    
    def test_perfect_correlation(self):
        """Test perfect positive and negative correlation."""
        # Perfect positive correlation
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x  # Perfect linear relationship
        data = np.column_stack([x, y])
        
        analyzer = FeatureAnalyzer(data, ['X', 'Y'])
        corr = analyzer.calculate_correlation_manual()
        
        assert np.isclose(corr[0, 1], 1.0, atol=1e-10), \
            "Perfect positive relationship should have correlation = 1.0"
        
        # Perfect negative correlation
        z = -2 * x
        data = np.column_stack([x, z])
        
        analyzer = FeatureAnalyzer(data, ['X', 'Z'])
        corr = analyzer.calculate_correlation_manual()
        
        assert np.isclose(corr[0, 1], -1.0, atol=1e-10), \
            "Perfect negative relationship should have correlation = -1.0"
    
    def test_zero_correlation(self):
        """Test uncorrelated data."""
        np.random.seed(42)
        
        # Create independent random variables
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 1, 1000)
        data = np.column_stack([x, y])
        
        analyzer = FeatureAnalyzer(data, ['X', 'Y'])
        corr = analyzer.calculate_correlation_manual()
        
        # With enough samples, correlation should be near zero
        assert abs(corr[0, 1]) < 0.1, \
            "Independent variables should have correlation near 0"


# Run tests with verbose output
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
TEST_EOF

echo "✅ All files generated successfully!"
echo ""
echo "🔍 Verifying all files were created..."

# List of required files
REQUIRED_FILES=("setup.sh" "requirements.txt" "lesson_code.py" "test_lesson.py")
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

