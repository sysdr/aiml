"""
Day 4: Tests for Lists and Tuples AI Implementation
Run with: python test_lesson.py
"""

import unittest
import sys
from lesson_code import AIDataProcessor


class TestAIDataStructures(unittest.TestCase):
    """Test cases to verify understanding of lists and tuples in AI context"""
    
    def setUp(self):
        """Set up test data processor for each test"""
        self.ai_processor = AIDataProcessor()
    
    def test_initialization(self):
        """Test that AI processor initializes correctly"""
        self.assertEqual(len(self.ai_processor.training_data), 0)
        self.assertEqual(len(self.ai_processor.predictions), 0)
        self.assertIsInstance(self.ai_processor.model_config, tuple)
        self.assertIsInstance(self.ai_processor.input_shape, tuple)
        
        # Verify tuple immutability concept
        with self.assertRaises(TypeError):
            self.ai_processor.model_config[0] = "new_type"
    
    def test_add_training_sample(self):
        """Test adding training samples with lists and tuples"""
        features = [0.1, 0.2, 0.3, 0.4]
        label = "test_class"
        metadata = ("source", 123456, 0.9)
        
        self.ai_processor.add_training_sample(features, label, metadata)
        
        self.assertEqual(len(self.ai_processor.training_data), 1)
        
        # Verify data structure
        data_point = self.ai_processor.training_data[0]
        self.assertIsInstance(data_point, tuple)  # Immutable record
        self.assertIsInstance(data_point[0], list)  # Features as list
        self.assertEqual(data_point[1], label)
        self.assertEqual(data_point[2], metadata)
    
    def test_feature_normalization(self):
        """Test feature normalization (common AI preprocessing)"""
        features = [10, 20, 30, 40, 50]
        normalized = self.ai_processor.normalize_features(features)
        
        # Check normalized range [0, 1]
        self.assertTrue(all(0 <= x <= 1 for x in normalized))
        self.assertEqual(min(normalized), 0.0)
        self.assertEqual(max(normalized), 1.0)
        
        # Test edge cases
        same_values = [5, 5, 5, 5]
        normalized_same = self.ai_processor.normalize_features(same_values)
        self.assertTrue(all(x == 0.5 for x in normalized_same))
    
    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        # Add some training data first
        self.ai_processor.add_training_sample([0.8, 0.9], "high")
        self.ai_processor.add_training_sample([0.1, 0.2], "low")
        
        # Test batch prediction
        test_features = [
            [0.7, 0.8],  # Should predict high
            [0.2, 0.1],  # Should predict low
            [0.5, 0.6],  # Should predict medium
        ]
        
        predictions = self.ai_processor.predict_batch(test_features)
        
        self.assertEqual(len(predictions), 3)
        
        # Verify prediction format (tuple with prediction and confidence)
        for prediction in predictions:
            self.assertIsInstance(prediction, tuple)
            self.assertEqual(len(prediction), 2)
            
            pred_class, confidence = prediction
            self.assertIsInstance(pred_class, str)
            self.assertIsInstance(confidence, (int, float))
            self.assertTrue(0 <= confidence <= 1)
    
    def test_performance_analysis(self):
        """Test dataset analysis functionality"""
        # Add diverse training data
        self.ai_processor.add_training_sample([0.1, 0.2], "cat")
        self.ai_processor.add_training_sample([0.3, 0.4], "dog")
        self.ai_processor.add_training_sample([0.5, 0.6], "cat")
        self.ai_processor.add_training_sample([0.7, 0.8], "bird")
        
        stats = self.ai_processor.analyze_performance()
        
        self.assertEqual(stats["total_training_samples"], 4)
        self.assertEqual(len(stats["unique_labels"]), 3)  # cat, dog, bird
        self.assertEqual(stats["label_distribution"]["cat"], 2)
        self.assertEqual(stats["label_distribution"]["dog"], 1)
        self.assertEqual(stats["label_distribution"]["bird"], 1)
    
    def test_confidence_filtering(self):
        """Test filtering predictions by confidence"""
        # Make some predictions first
        test_features = [[0.9, 0.9], [0.1, 0.1], [0.5, 0.5]]
        predictions = self.ai_processor.predict_batch(test_features)
        
        # Filter high confidence
        high_conf = self.ai_processor.filter_by_confidence(0.7)
        
        # Verify filtering worked
        for pred_class, confidence in high_conf:
            self.assertGreaterEqual(confidence, 0.7)
    
    def test_feature_statistics(self):
        """Test feature statistics calculation"""
        # Add training data with known features
        self.ai_processor.add_training_sample([1.0, 2.0, 3.0], "test1")
        self.ai_processor.add_training_sample([4.0, 5.0, 6.0], "test2")
        
        stats = self.ai_processor.get_feature_statistics()
        
        self.assertEqual(stats["total_features"], 6)  # 3 + 3
        self.assertEqual(stats["min_value"], 0.0)  # After normalization
        self.assertEqual(stats["max_value"], 1.0)  # After normalization
        self.assertEqual(stats["feature_dimensions"], 3)
    
    def test_list_vs_tuple_behavior(self):
        """Test understanding of mutable lists vs immutable tuples"""
        # Test list mutability
        features = [1, 2, 3]
        original_features = features.copy()
        features.append(4)
        self.assertNotEqual(features, original_features)
        
        # Test tuple immutability
        config = (1, 2, 3)
        with self.assertRaises(AttributeError):
            config.append(4)
        
        with self.assertRaises(TypeError):
            config[0] = 5
    
    def test_list_comprehensions(self):
        """Test AI-style data processing with list comprehensions"""
        data = [1, 2, 3, 4, 5]
        
        # Filter (common in AI preprocessing)
        filtered = [x for x in data if x > 3]
        self.assertEqual(filtered, [4, 5])
        
        # Transform (feature engineering)
        squared = [x**2 for x in data]
        self.assertEqual(squared, [1, 4, 9, 16, 25])
        
        # Combine filter and transform
        filtered_squared = [x**2 for x in data if x % 2 == 0]
        self.assertEqual(filtered_squared, [4, 16])


def run_interactive_tests():
    """Interactive tests that show results to students"""
    print("🧪 Running Interactive Tests - Learn by Seeing!")
    print("=" * 50)
    
    # Test 1: Data Structure Creation
    print("\n1. Testing Data Structure Creation...")
    ai_processor = AIDataProcessor()
    print(f"   ✅ Created AI processor with empty lists: {len(ai_processor.training_data)} samples")
    print(f"   ✅ Model config tuple: {ai_processor.model_config}")
    print(f"   ✅ Input shape tuple: {ai_processor.input_shape}")
    
    # Test 2: Adding Data
    print("\n2. Testing Data Addition...")
    ai_processor.add_training_sample([0.5, 0.8, 0.3], "positive")
    ai_processor.add_training_sample([0.2, 0.1, 0.9], "negative")
    print(f"   ✅ Added 2 samples, total: {len(ai_processor.training_data)}")
    
    # Show data structure
    sample = ai_processor.training_data[0]
    print(f"   📊 Sample structure: {type(sample).__name__} with {len(sample)} elements")
    print(f"   📊 Features type: {type(sample[0]).__name__} (mutable)")
    print(f"   📊 Label: {sample[1]} (immutable in tuple)")
    
    # Test 3: Predictions
    print("\n3. Testing Predictions...")
    test_data = [[0.7, 0.6, 0.4], [0.1, 0.2, 0.8]]
    predictions = ai_processor.predict_batch(test_data)
    print(f"   ✅ Made {len(predictions)} predictions")
    for i, (pred, conf) in enumerate(predictions):
        print(f"   🎯 Sample {i+1}: {pred} (confidence: {conf})")
    
    # Test 4: List vs Tuple behavior
    print("\n4. Testing List vs Tuple Behavior...")
    
    # Lists are mutable
    features_list = [1, 2, 3]
    print(f"   📝 Original list: {features_list}")
    features_list.append(4)
    print(f"   📝 After append: {features_list} ✅ Lists can change!")
    
    # Tuples are immutable
    config_tuple = (1, 2, 3)
    print(f"   🔒 Original tuple: {config_tuple}")
    try:
        config_tuple.append(4)
    except AttributeError as e:
        print(f"   🔒 Cannot append to tuple: {type(e).__name__} ✅ Tuples are immutable!")
    
    # Test 5: AI-style data processing
    print("\n5. Testing AI Data Processing Patterns...")
    data = [0.1, 0.5, 0.8, 0.3, 0.9, 0.2]
    
    # Filter high values (like filtering confident predictions)
    high_confidence = [x for x in data if x > 0.6]
    print(f"   🎯 High confidence values: {high_confidence}")
    
    # Transform data (like feature scaling)
    scaled = [x * 100 for x in data]
    print(f"   🔧 Scaled to percentage: {scaled}")
    
    # Combine operations (common AI pipeline)
    processed = [round(x * 100, 1) for x in data if x > 0.4]
    print(f"   ⚙️  Filtered + scaled: {processed}")
    
    print("\n✅ All interactive tests passed!")
    print("🎓 You understand how AI systems use lists and tuples!")


if __name__ == "__main__":
    print("Day 4: Lists and Tuples for AI - Test Suite")
    print("=" * 50)
    
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=1)
    
    # Run interactive tests
    run_interactive_tests()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("💡 Key learning: Lists for changing data, tuples for fixed records")
    print("🚀 Ready for Day 5: Dictionaries and Sets!")
