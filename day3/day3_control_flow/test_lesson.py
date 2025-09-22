"""
Tests for Day 3: Control Flow for AI Systems
Verify that all control flow concepts work correctly
"""

import sys
import pytest
from lesson_code import DataValidator, SimpleSentimentAI

def test_data_validator_basic():
    """Test basic data validation functionality"""
    validator = DataValidator()
    
    # Test valid data
    is_valid, message = validator.validate_single_item("Valid text data")
    assert is_valid == True
    assert "Valid data" in message
    
    # Test invalid data - None
    is_valid, message = validator.validate_single_item(None)
    assert is_valid == False
    assert "Missing data" in message
    
    # Test invalid data - too short
    is_valid, message = validator.validate_single_item("Hi")
    assert is_valid == False
    assert "Too short" in message

def test_data_validator_dataset():
    """Test dataset processing with mixed valid/invalid data"""
    validator = DataValidator()
    
    test_dataset = [
        "Valid text",
        None,
        "Another valid text",
        "X",  # Too short
        12345  # Valid number
    ]
    
    results = validator.process_dataset(test_dataset)
    
    # Should have 3 valid items and 2 invalid
    assert len(results['valid_data']) == 3
    assert len(results['invalid_data']) == 2
    assert results['stats']['total_checked'] == 5

def test_sentiment_analyzer_positive():
    """Test sentiment analysis for positive text"""
    analyzer = SimpleSentimentAI()
    
    result = analyzer.analyze_sentiment("This is amazing and wonderful!")
    
    assert result['sentiment'] == 'positive'
    assert result['confidence'] > 0.5
    assert 'amazing' in str(result['matching_words']).lower()

def test_sentiment_analyzer_negative():
    """Test sentiment analysis for negative text"""
    analyzer = SimpleSentimentAI()
    
    result = analyzer.analyze_sentiment("This is terrible and awful!")
    
    assert result['sentiment'] == 'negative'
    assert result['confidence'] > 0.5
    assert 'terrible' in str(result['matching_words']).lower()

def test_sentiment_analyzer_neutral():
    """Test sentiment analysis for neutral text"""
    analyzer = SimpleSentimentAI()
    
    result = analyzer.analyze_sentiment("This is a normal statement.")
    
    assert result['sentiment'] == 'neutral'
    assert result['confidence'] == 0.5

def test_sentiment_batch_processing():
    """Test batch sentiment analysis"""
    analyzer = SimpleSentimentAI()
    
    test_texts = [
        "Great product!",
        "Terrible experience",
        "It's okay"
    ]
    
    results = analyzer.batch_analyze(test_texts)
    
    # Should have results in different categories
    assert len(results['positive']) >= 1
    assert len(results['negative']) >= 1
    assert (len(results['neutral']) + len(results['mixed'])) >= 1

def test_data_types_handling():
    """Test handling of different data types"""
    validator = DataValidator()
    
    # Test string
    is_valid, _ = validator.validate_single_item("Test string")
    assert is_valid == True
    
    # Test integer
    is_valid, _ = validator.validate_single_item(12345)
    assert is_valid == True
    
    # Test float
    is_valid, _ = validator.validate_single_item(123.45)
    assert is_valid == True
    
    # Test invalid type (list)
    is_valid, _ = validator.validate_single_item([1, 2, 3])
    assert is_valid == False

def run_all_tests():
    """Run all tests and display results"""
    print("🧪 Running Day 3 Control Flow Tests")
    print("===================================")
    
    test_functions = [
        test_data_validator_basic,
        test_data_validator_dataset,
        test_sentiment_analyzer_positive,
        test_sentiment_analyzer_negative,
        test_sentiment_analyzer_neutral,
        test_sentiment_batch_processing,
        test_data_types_handling
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {str(e)}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Control flow concepts are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check your implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
