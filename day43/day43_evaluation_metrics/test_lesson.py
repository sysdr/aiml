"""
Test Suite for Day 43: Model Evaluation Metrics
Comprehensive tests ensuring correct metric calculations
"""

import pytest
import numpy as np
from lesson_code import MetricsCalculator


class TestMetricsCalculator:
    """Test suite for metrics calculator"""
    
    def test_perfect_predictions(self):
        """Test metrics with 100% correct predictions"""
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        
        calc = MetricsCalculator(y_true, y_pred)
        
        assert calc.accuracy() == 1.0
        assert calc.precision() == 1.0
        assert calc.recall() == 1.0
        assert calc.f1_score() == 1.0
        assert calc.tp == 4
        assert calc.tn == 4
        assert calc.fp == 0
        assert calc.fn == 0
    
    def test_all_wrong_predictions(self):
        """Test metrics with 100% incorrect predictions"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        
        calc = MetricsCalculator(y_true, y_pred)
        
        assert calc.accuracy() == 0.0
        assert calc.precision() == 0.0  # No true positives
        assert calc.recall() == 0.0     # No true positives
        assert calc.tp == 0
        assert calc.tn == 0
        assert calc.fp == 2
        assert calc.fn == 2
    
    def test_high_precision_low_recall(self):
        """
        Test scenario: Conservative model
        Predicts positive only when very confident
        """
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # Only one positive prediction
        
        calc = MetricsCalculator(y_true, y_pred)
        
        # Should have perfect precision (the one positive prediction is correct)
        assert calc.precision() == 1.0
        # But terrible recall (only caught 1 out of 4 actual positives)
        assert calc.recall() == 0.25
        assert calc.tp == 1
        assert calc.fp == 0
        assert calc.fn == 3
    
    def test_high_recall_low_precision(self):
        """
        Test scenario: Aggressive model
        Predicts positive very liberally
        """
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0])  # Many positive predictions
        
        calc = MetricsCalculator(y_true, y_pred)
        
        # Should have perfect recall (caught both actual positives)
        assert calc.recall() == 1.0
        # But poor precision (only 2 out of 5 positive predictions are correct)
        assert calc.precision() == 0.4
        assert calc.tp == 2
        assert calc.fp == 3
        assert calc.fn == 0
    
    def test_balanced_metrics(self):
        """Test scenario with balanced precision and recall"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])
        
        calc = MetricsCalculator(y_true, y_pred)
        
        # TP=3, FP=1, FN=1, TN=3
        assert calc.precision() == 0.75  # 3/(3+1)
        assert calc.recall() == 0.75     # 3/(3+1)
        assert calc.accuracy() == 0.75   # (3+3)/8
        # F1 should equal precision and recall when they're equal
        assert calc.f1_score() == 0.75
    
    def test_imbalanced_data_scenario(self):
        """
        Test realistic imbalanced scenario (fraud detection)
        99% legitimate, 1% fraud
        """
        # 100 transactions: 99 legitimate, 1 fraud
        y_true = np.array([0] * 99 + [1] * 1)
        
        # Naive model: predict all legitimate (achieves 99% accuracy!)
        y_pred_naive = np.array([0] * 100)
        
        calc = MetricsCalculator(y_true, y_pred_naive)
        
        # High accuracy but completely useless
        assert calc.accuracy() == 0.99
        # But zero precision and recall (catches no fraud)
        assert calc.recall() == 0.0
        assert calc.tp == 0
        assert calc.fn == 1
    
    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape and values"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        
        calc = MetricsCalculator(y_true, y_pred)
        cm = calc.confusion_matrix()
        
        assert cm.shape == (2, 2)
        assert cm[0, 0] == calc.tn  # Top-left: True Negatives
        assert cm[0, 1] == calc.fp  # Top-right: False Positives
        assert cm[1, 0] == calc.fn  # Bottom-left: False Negatives
        assert cm[1, 1] == calc.tp  # Bottom-right: True Positives
    
    def test_f1_score_calculation(self):
        """Test F1 score as harmonic mean"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 0])
        
        calc = MetricsCalculator(y_true, y_pred)
        
        precision = calc.precision()  # 2/3
        recall = calc.recall()        # 2/4 = 0.5
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        assert abs(calc.f1_score() - expected_f1) < 1e-10
    
    def test_edge_case_no_positives(self):
        """Test when there are no positive predictions"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        
        calc = MetricsCalculator(y_true, y_pred)
        
        # Precision undefined (0/0), should return 0
        assert calc.precision() == 0.0
        # Recall is 0 (caught 0 out of 2)
        assert calc.recall() == 0.0
    
    def test_edge_case_no_actual_positives(self):
        """Test when there are no actual positives in ground truth"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        
        calc = MetricsCalculator(y_true, y_pred)
        
        # Recall undefined (dividing by 0), should return 0
        assert calc.recall() == 0.0
        # Precision is 0 (all positive predictions are false)
        assert calc.precision() == 0.0
    
    def test_medical_scenario_realistic(self):
        """
        Test medical diagnosis scenario with realistic numbers
        Similar to cancer screening test
        """
        # 1000 patients: 50 with disease (5%)
        y_true = np.array([1] * 50 + [0] * 950)
        
        # Model catches 45 out of 50 (90% recall)
        # But has 100 false positives (from the 950 healthy patients)
        y_pred = np.array([1] * 45 + [0] * 5 + [1] * 100 + [0] * 850)
        
        calc = MetricsCalculator(y_true, y_pred)
        
        # Recall should be high (catching disease is critical)
        assert calc.recall() >= 0.85
        # Precision will be lower (many false alarms)
        assert calc.precision() < 0.5
        # But this is acceptable in medical context!
        assert calc.fn <= 5  # Missing at most 5 patients
    
    def test_spam_filter_realistic(self):
        """
        Test spam filter scenario
        Gmail-style: prioritize precision
        """
        # 1000 emails: 150 spam (15%)
        y_true = np.array([1] * 150 + [0] * 850)
        
        # Model catches 130 spam
        # Only 10 false positives (good emails marked as spam)
        y_pred = np.array([1] * 130 + [0] * 20 + [1] * 10 + [0] * 840)
        
        calc = MetricsCalculator(y_true, y_pred)
        
        # Precision should be high (trust the spam folder)
        assert calc.precision() >= 0.90
        # Recall can be lower (some spam in inbox is tolerable)
        assert calc.recall() >= 0.80
        # Very few false positives
        assert calc.fp <= 10
    
    def test_metrics_dictionary_completeness(self):
        """Test that get_all_metrics returns all expected metrics"""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        
        calc = MetricsCalculator(y_true, y_pred)
        metrics = calc.get_all_metrics()
        
        expected_keys = [
            'Accuracy', 'Precision', 'Recall', 'F1 Score',
            'True Positives', 'True Negatives', 
            'False Positives', 'False Negatives'
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        # Check types
        assert isinstance(metrics['Accuracy'], float)
        assert isinstance(metrics['True Positives'], int)


def test_integration_all_scenarios():
    """
    Integration test: Run all scenarios and verify they complete
    """
    from lesson_code import ScenarioSimulator
    
    # Should not raise any exceptions
    ScenarioSimulator.medical_diagnosis_scenario()
    ScenarioSimulator.spam_filter_scenario()
    ScenarioSimulator.fraud_detection_scenario()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
