"""
Tests for Day 37: Introduction to AI, ML, and Deep Learning
"""

import pytest
import numpy as np
from sklearn.metrics import r2_score
from lesson_code import AIMLIntroduction


class TestAIMLIntroduction:
    """Test suite for AI/ML Introduction"""
    
    @pytest.fixture
    def intro(self):
        """Create AIMLIntroduction instance"""
        return AIMLIntroduction()
    
    def test_initialization(self, intro):
        """Test initialization"""
        assert intro.models == {}
        assert intro.metrics == {}
    
    def test_traditional_vs_ml(self, intro):
        """Test traditional programming vs ML comparison"""
        results = intro.traditional_programming_vs_ml()
        
        assert 'traditional' in results
        assert 'ml_r2_score' in results
        assert 'ml_prediction' in results
        assert 0 <= results['ml_r2_score'] <= 1
    
    def test_ai_ml_dl_relationship(self, intro):
        """Test AI, ML, DL relationship explanation"""
        results = intro.ai_ml_dl_relationship()
        
        assert 'ai' in results
        assert 'ml' in results
        assert 'dl' in results
    
    def test_supervised_learning_demo(self, intro):
        """Test supervised learning demonstration"""
        results = intro.supervised_learning_demo()
        
        assert 'model' in results
        assert 'mse' in results
        assert 'r2' in results
        assert 'predictions' in results
        assert results['r2'] > 0.5  # Should have reasonable performance
        assert results['mse'] > 0
    
    def test_learning_types_overview(self, intro):
        """Test learning types overview"""
        results = intro.learning_types_overview()
        
        assert 'Supervised Learning' in results
        assert 'Unsupervised Learning' in results
        assert 'Reinforcement Learning' in results
        
        for ml_type in results:
            assert 'description' in results[ml_type]
            assert 'examples' in results[ml_type]
    
    def test_complete_introduction(self, intro):
        """Test complete introduction workflow"""
        results = intro.run_complete_introduction()
        
        assert 'traditional_vs_ml' in results
        assert 'ai_ml_dl' in results
        assert 'supervised_learning' in results
        assert 'learning_types' in results


def test_model_predictions():
    """Test that model makes reasonable predictions"""
    intro = AIMLIntroduction()
    results = intro.supervised_learning_demo()
    
    model = results['model']
    
    # Test predictions are positive (house prices should be positive)
    predictions = model.predict([[1000], [2000], [3000]])
    assert all(p > 0 for p in predictions)
    
    # Test larger houses cost more (monotonic relationship)
    assert predictions[2] > predictions[1] > predictions[0]


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
