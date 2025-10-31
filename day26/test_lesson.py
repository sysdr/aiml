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
        # Left-skewed: mean < median (most values high, few low outliers)
        data = [1, 90, 95, 98, 99, 100]
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
