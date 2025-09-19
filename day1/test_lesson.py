#!/usr/bin/env python3
"""
Test Suite for Day 1: Python Basics for AI
==========================================

This module tests the core concepts learned in Day 1.
It helps students verify their understanding and ensures the code works correctly.

Run with: python test_lesson.py
"""

import unittest
import sys
import os

# Add current directory to path to import lesson_code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lesson_code import SmartResponder, demonstrate_ai_data_structures


class TestDay1Concepts(unittest.TestCase):
    """Test cases for Day 1 Python concepts used in AI."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.ai_assistant = SmartResponder("Test Assistant")
    
    def test_smart_responder_initialization(self):
        """Test that SmartResponder initializes correctly (like testing AI model loading)."""
        # Test basic initialization
        self.assertEqual(self.ai_assistant.name, "Test Assistant")
        self.assertEqual(len(self.ai_assistant.conversation_memory), 0)
        self.assertEqual(self.ai_assistant.user_profile["interactions"], 0)
        
        # Test that response patterns are loaded (like AI model weights)
        self.assertGreater(len(self.ai_assistant.response_patterns), 0)
        self.assertIn("greeting", self.ai_assistant.response_patterns)
        
        print("✅ AI Assistant initialized correctly!")
    
    def test_input_analysis(self):
        """Test input analysis function (like testing AI text processing)."""
        # Test greeting detection
        analysis = self.ai_assistant.analyze_input("Hello there!")
        self.assertIn("greeting", analysis["detected_patterns"])
        self.assertEqual(analysis["sentiment"], "positive")
        
        # Test question detection
        analysis = self.ai_assistant.analyze_input("What is AI?")
        self.assertIn("question", analysis["detected_patterns"])
        
        # Test AI topic detection
        analysis = self.ai_assistant.analyze_input("I love machine learning")
        self.assertIn("ai_related", analysis["detected_patterns"])
        
        # Test basic properties
        self.assertIn("word_count", analysis)
        self.assertIn("timestamp", analysis)
        
        print("✅ Input analysis works correctly!")
    
    def test_response_generation(self):
        """Test response generation (like testing AI output quality)."""
        # Test with greeting
        analysis = self.ai_assistant.analyze_input("Hello!")
        response = self.ai_assistant.generate_response(analysis)
        
        # Response should be a non-empty string
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # User profile should be updated
        self.assertEqual(self.ai_assistant.user_profile["interactions"], 1)
        
        print("✅ Response generation works correctly!")
    
    def test_conversation_memory(self):
        """Test conversation storage (like testing AI memory systems)."""
        user_input = "Hello AI!"
        analysis = self.ai_assistant.analyze_input(user_input)
        response = self.ai_assistant.generate_response(analysis)
        
        # Save conversation
        self.ai_assistant.save_conversation(user_input, analysis, response)
        
        # Check memory storage
        self.assertEqual(len(self.ai_assistant.conversation_memory), 1)
        
        saved_entry = self.ai_assistant.conversation_memory[0]
        self.assertEqual(saved_entry["user_input"], user_input)
        self.assertEqual(saved_entry["ai_response"], response)
        
        print("✅ Conversation memory works correctly!")
    
    def test_conversation_stats(self):
        """Test conversation statistics (like testing AI analytics)."""
        # Have a short conversation
        inputs = ["Hello!", "How are you?", "This is amazing!"]
        
        for user_input in inputs:
            analysis = self.ai_assistant.analyze_input(user_input)
            response = self.ai_assistant.generate_response(analysis)
            self.ai_assistant.save_conversation(user_input, analysis, response)
        
        # Check stats
        stats = self.ai_assistant.get_conversation_stats()
        
        self.assertEqual(stats["total_interactions"], 3)
        self.assertGreater(stats["average_message_length"], 0)
        self.assertIn("sentiment_distribution", stats)
        
        print("✅ Conversation statistics work correctly!")
    
    def test_data_structures(self):
        """Test AI data structure demonstration."""
        try:
            model_config = demonstrate_ai_data_structures()
            
            # Test that the returned config has expected structure
            self.assertIsInstance(model_config, dict)
            self.assertIn("model_name", model_config)
            self.assertIn("parameters", model_config)
            self.assertIn("training_data", model_config)
            self.assertIn("performance_metrics", model_config)
            
            # Test nested data structures
            self.assertIsInstance(model_config["training_data"], list)
            self.assertIsInstance(model_config["parameters"], dict)
            
            print("✅ AI data structures demonstration works!")
            
        except Exception as e:
            self.fail(f"Data structures demo failed: {e}")


class TestPythonBasics(unittest.TestCase):
    """Test basic Python concepts that are essential for AI."""
    
    def test_variables_and_data_types(self):
        """Test understanding of variables and data types used in AI."""
        # Test different data types used in AI
        model_name = "GPT-4"  # String for model identification
        model_version = 4.0   # Float for version numbers
        is_trained = True     # Boolean for status flags
        layer_count = 96      # Integer for architecture parameters
        
        # Test type checking (important for AI data validation)
        self.assertIsInstance(model_name, str)
        self.assertIsInstance(model_version, float)
        self.assertIsInstance(is_trained, bool)
        self.assertIsInstance(layer_count, int)
        
        print("✅ Variables and data types work correctly!")
    
    def test_lists_for_ai(self):
        """Test list operations commonly used in AI."""
        # AI training data example
        training_data = ["sample1", "sample2", "sample3"]
        
        # Test list operations
        training_data.append("sample4")
        self.assertEqual(len(training_data), 4)
        
        # Test list comprehension (common in AI preprocessing)
        processed_data = [sample.upper() for sample in training_data]
        self.assertEqual(len(processed_data), 4)
        self.assertTrue(all(sample.isupper() for sample in processed_data))
        
        print("✅ List operations for AI work correctly!")
    
    def test_dictionaries_for_ai(self):
        """Test dictionary operations used in AI configuration."""
        # AI model configuration
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        }
        
        # Test dictionary operations
        config["dropout_rate"] = 0.1
        self.assertIn("dropout_rate", config)
        
        # Test getting values with defaults (common in AI config)
        optimizer = config.get("optimizer", "adam")
        self.assertEqual(optimizer, "adam")
        
        print("✅ Dictionary operations for AI work correctly!")
    
    def test_functions_for_ai(self):
        """Test function concepts used in AI systems."""
        
        def preprocess_text(text):
            """Simple text preprocessing function like those used in AI."""
            return text.lower().strip()
        
        def calculate_accuracy(correct, total):
            """Calculate accuracy metric used in AI evaluation."""
            if total == 0:
                return 0
            return correct / total
        
        # Test functions
        processed = preprocess_text("  HELLO WORLD  ")
        self.assertEqual(processed, "hello world")
        
        accuracy = calculate_accuracy(85, 100)
        self.assertEqual(accuracy, 0.85)
        
        print("✅ Functions for AI work correctly!")


def run_interactive_tests():
    """Run interactive tests to verify student understanding."""
    print("\n🧪 Interactive Learning Verification")
    print("=" * 40)
    
    # Test 1: Variable Understanding
    print("\n📝 Test 1: Create a variable for storing AI model accuracy")
    try:
        user_accuracy = input("Enter a decimal value (e.g., 0.85): ")
        accuracy = float(user_accuracy)
        if 0 <= accuracy <= 1:
            print(f"✅ Great! You stored accuracy: {accuracy}")
        else:
            print("⚠️  Accuracy should be between 0 and 1, but good try!")
    except ValueError:
        print("⚠️  That's not a valid number, but you're learning!")
    
    # Test 2: List Understanding
    print("\n📝 Test 2: Understanding Lists in AI")
    favorite_topics = input("Enter 3 AI topics separated by commas: ").split(",")
    favorite_topics = [topic.strip() for topic in favorite_topics]
    print(f"✅ Your AI interest list: {favorite_topics}")
    print(f"   List length: {len(favorite_topics)} (AI systems track data size)")
    
    # Test 3: Function Understanding
    print("\n📝 Test 3: Function Logic")
    print("If an AI model gets 90 correct answers out of 100 questions,")
    user_answer = input("what's its accuracy? (enter as decimal): ")
    try:
        user_accuracy = float(user_answer)
        if abs(user_accuracy - 0.9) < 0.01:
            print("✅ Perfect! You understand AI accuracy calculation!")
        else:
            print(f"💡 Close! The answer is 0.9 (90/100). You said {user_accuracy}")
    except ValueError:
        print("💡 The answer is 0.9 (90 correct ÷ 100 total = 0.9)")
    
    print("\n🎉 Interactive tests complete! You're ready for Day 2!")


def main():
    """Run all tests and provide feedback to students."""
    print("🔬 Day 1 Test Suite: Python Basics for AI")
    print("=" * 45)
    print("This will verify you've mastered today's concepts!\n")
    
    # Run automated tests
    unittest.main(argv=[''], exit=False, verbosity=0)
    
    # Run interactive tests
    try:
        run_interactive_tests()
    except KeyboardInterrupt:
        print("\n\nTests interrupted - that's okay! Keep practicing! 🚀")
    
    print("\n" + "=" * 45)
    print("🏆 Day 1 Complete! Key Takeaways:")
    print("• Variables store AI data and configurations")
    print("• Lists and dictionaries organize AI information") 
    print("• Functions break complex AI tasks into manageable pieces")
    print("• Input/output enables AI systems to communicate")
    print("• Pattern matching is fundamental to AI understanding")
    print("\n🚀 Tomorrow: Variables, Data Types, and Operators for AI!")


if __name__ == "__main__":
    main()