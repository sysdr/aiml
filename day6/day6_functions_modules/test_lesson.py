"""
Test suite for Day 6: Functions, Modules, and Libraries
This teaches students how to verify their AI code works correctly
"""

import json
from lesson_code import (
    clean_text, 
    extract_word_features, 
    analyze_text_sentiment, 
    calculate_ai_readiness,
    generate_analysis_report
)


def test_clean_text():
    """Test text cleaning function"""
    print("🧪 Testing clean_text function...")
    
    # Test cases
    test_cases = [
        ("Hello World!", "hello world"),
        ("  Extra   Spaces  ", "extra spaces"),
        ("UPPERCASE text", "uppercase text"),
        ("Punctuation! @#$%", "punctuation"),
        ("", ""),
        ("123 numbers", "123 numbers")
    ]
    
    passed = 0
    for input_text, expected in test_cases:
        result = clean_text(input_text)
        if result == expected:
            print(f"   ✅ '{input_text}' → '{result}'")
            passed += 1
        else:
            print(f"   ❌ '{input_text}' → '{result}' (expected '{expected}')")
    
    print(f"   📊 Passed: {passed}/{len(test_cases)} tests")
    return passed == len(test_cases)


def test_extract_word_features():
    """Test feature extraction function"""
    print("\n🧪 Testing extract_word_features function...")
    
    # Test with known text
    text = "hello world test"
    features = extract_word_features(text)
    
    expected = {
        'word_count': 3,
        'avg_word_length': 4.67,
        'unique_words': 3,
        'char_count': 16
    }
    
    passed = True
    for key, expected_value in expected.items():
        actual_value = features[key]
        if key == 'avg_word_length':
            # Allow small floating point differences
            if abs(actual_value - expected_value) < 0.1:
                print(f"   ✅ {key}: {actual_value}")
            else:
                print(f"   ❌ {key}: {actual_value} (expected ~{expected_value})")
                passed = False
        else:
            if actual_value == expected_value:
                print(f"   ✅ {key}: {actual_value}")
            else:
                print(f"   ❌ {key}: {actual_value} (expected {expected_value})")
                passed = False
    
    # Test empty text
    empty_features = extract_word_features("")
    if empty_features['word_count'] == 0:
        print("   ✅ Empty text handling works")
    else:
        print("   ❌ Empty text handling failed")
        passed = False
    
    return passed


def test_analyze_text_sentiment():
    """Test sentiment analysis function"""
    print("\n🧪 Testing analyze_text_sentiment function...")
    
    test_cases = [
        ("I love this amazing course", "positive"),
        ("This is terrible and awful", "negative"), 
        ("The weather is okay today", "neutral"),
        ("", "neutral")
    ]
    
    passed = 0
    for text, expected_sentiment in test_cases:
        result = analyze_text_sentiment(text)
        actual_sentiment = result['sentiment']
        
        if actual_sentiment == expected_sentiment:
            print(f"   ✅ '{text}' → {actual_sentiment}")
            passed += 1
        else:
            print(f"   ❌ '{text}' → {actual_sentiment} (expected {expected_sentiment})")
    
    print(f"   📊 Passed: {passed}/{len(test_cases)} tests")
    return passed == len(test_cases)


def test_calculate_ai_readiness():
    """Test AI readiness calculation"""
    print("\n🧪 Testing calculate_ai_readiness function...")
    
    # Test with good features
    good_features = {
        'word_count': 15,
        'avg_word_length': 5.2,
        'unique_words': 12,
        'char_count': 80
    }
    
    result = calculate_ai_readiness(good_features)
    
    if result['score'] > 70:
        print(f"   ✅ Good text gets high score: {result['score']}")
    else:
        print(f"   ❌ Good text got low score: {result['score']}")
        return False
    
    # Test with poor features
    poor_features = {
        'word_count': 2,
        'avg_word_length': 2.5,
        'unique_words': 1,
        'char_count': 5
    }
    
    result2 = calculate_ai_readiness(poor_features)
    
    if result2['score'] < 50:
        print(f"   ✅ Poor text gets low score: {result2['score']}")
    else:
        print(f"   ❌ Poor text got high score: {result2['score']}")
        return False
    
    return True


def test_integration():
    """Test the complete analysis pipeline"""
    print("\n🧪 Testing complete analysis pipeline...")
    
    test_text = "I absolutely love learning about AI and machine learning!"
    
    try:
        report = generate_analysis_report(test_text)
        
        # Check that report has all required sections
        required_keys = ['timestamp', 'original_text', 'features', 'sentiment', 'ai_readiness']
        
        for key in required_keys:
            if key in report:
                print(f"   ✅ Report contains {key}")
            else:
                print(f"   ❌ Report missing {key}")
                return False
        
        # Check that we can serialize to JSON (important for AI systems)
        json_str = json.dumps(report, indent=2)
        print("   ✅ Report serializes to JSON successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 RUNNING DAY 6 FUNCTION TESTS")
    print("="*50)
    
    tests = [
        ("Text Cleaning", test_clean_text),
        ("Feature Extraction", test_extract_word_features),
        ("Sentiment Analysis", test_analyze_text_sentiment),
        ("AI Readiness", test_calculate_ai_readiness),
        ("Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
            print(f"   🎉 {test_name} PASSED")
        else:
            print(f"   💥 {test_name} FAILED")
    
    print("\n" + "="*50)
    print(f"📊 FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your functions are working correctly!")
        print("🚀 You're ready to build AI systems!")
    else:
        print("🔧 Some tests failed. Review the code and try again.")
        print("💡 This is normal - debugging is part of AI development!")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()
