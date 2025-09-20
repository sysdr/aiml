#!/usr/bin/env python3
"""
Test Suite for Day 2: Variables, Data Types, and Operators
180-Day AI and Machine Learning Course

Simple tests to verify understanding of Python fundamentals for AI.
"""

import pytest
from lesson_code import SimpleAIAgent

class TestDataTypes:
    """Test understanding of Python data types in AI context."""
    
    def test_string_operations(self):
        """Test string operations used in AI systems."""
        prompt = "What is machine learning?"
        
        # Test string properties
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert len(prompt.split()) == 4  # 4 words
        
        # Test string methods used in AI preprocessing
        assert prompt.lower() == "what is machine learning?"
        assert "machine" in prompt.lower()
        assert prompt.replace("?", "") == "What is machine learning"
    
    def test_numeric_types(self):
        """Test numeric operations for AI calculations."""
        # Integer operations
        batch_size = 32
        total_samples = 1000
        num_batches = total_samples // batch_size
        
        assert isinstance(batch_size, int)
        assert isinstance(num_batches, int)
        assert num_batches == 31
        
        # Float operations
        confidence = 0.87
        learning_rate = 0.001
        threshold = 0.8
        
        assert isinstance(confidence, float)
        assert confidence > threshold
        assert learning_rate < 1.0
    
    def test_boolean_logic(self):
        """Test boolean operations for AI decision making."""
        model_ready = True
        data_available = True
        user_authenticated = False
        
        # Test logical operators
        can_proceed = model_ready and data_available
        needs_auth = not user_authenticated
        system_ok = model_ready or data_available
        
        assert can_proceed == True
        assert needs_auth == True
        assert system_ok == True
    
    def test_list_operations(self):
        """Test list operations for AI data handling."""
        conversation = ["Hello", "How are you?", "I'm fine"]
        scores = [0.9, 0.85, 0.92]
        
        # Test list properties
        assert len(conversation) == 3
        assert len(scores) == 3
        
        # Test list operations used in AI
        conversation.append("Great!")
        assert len(conversation) == 4
        assert conversation[-1] == "Great!"
        
        # Test statistical operations
        avg_score = sum(scores) / len(scores)
        assert 0.8 < avg_score < 1.0
        assert max(scores) == 0.92
        assert min(scores) == 0.85

class TestAIAgent:
    """Test the SimpleAIAgent class functionality."""
    
    def test_agent_initialization(self):
        """Test agent creation and initial state."""
        agent = SimpleAIAgent("TestBot")
        
        # Test initial values
        assert agent.name == "TestBot"
        assert agent.is_active == True
        assert agent.total_interactions == 0
        assert len(agent.conversation_history) == 0
        assert len(agent.confidence_scores) == 0
    
    def test_agent_input_processing(self):
        """Test agent's ability to process user input."""
        agent = SimpleAIAgent("TestBot")
        
        # Test valid input
        response = agent.process_input("Hello!")
        assert isinstance(response, str)
        assert len(response) > 0
        assert agent.total_interactions == 1
        assert len(agent.conversation_history) == 1
        assert len(agent.confidence_scores) == 1
        
        # Test confidence score range
        confidence = agent.confidence_scores[0]
        assert 0.0 <= confidence <= 1.0
    
    def test_agent_status_tracking(self):
        """Test agent's status tracking capabilities."""
        agent = SimpleAIAgent("TestBot")
        
        # Process some interactions
        agent.process_input("Hello")
        agent.process_input("How are you?")
        
        status = agent.get_agent_status()
        
        # Test status structure
        assert isinstance(status, dict)
        assert "agent_name" in status
        assert "total_interactions" in status
        assert "is_active" in status
        assert "recent_conversations" in status
        
        # Test status values
        assert status["agent_name"] == "TestBot"
        assert status["total_interactions"] == 2
        assert status["is_active"] == True
        assert len(status["recent_conversations"]) == 2
    
    def test_invalid_input_handling(self):
        """Test agent's handling of invalid input."""
        agent = SimpleAIAgent("TestBot")
        
        # Test empty input
        response = agent.process_input("")
        assert "Error" in response or "Invalid" in response
        
        # Test None input handling (should not crash)
        try:
            response = agent.process_input(None)
            # Should either handle gracefully or raise appropriate error
            assert True
        except (TypeError, AttributeError):
            # Expected for None input
            assert True

class TestOperators:
    """Test understanding of operators in AI context."""
    
    def test_arithmetic_operators(self):
        """Test arithmetic operators used in AI calculations."""
        # Batch processing calculations
        total_data = 1000
        batch_size = 32
        
        num_batches = total_data // batch_size  # Floor division
        remainder = total_data % batch_size     # Modulo
        
        assert num_batches == 31
        assert remainder == 8
        assert num_batches * batch_size + remainder == total_data
    
    def test_comparison_operators(self):
        """Test comparison operators for AI thresholds."""
        accuracy = 0.87
        target = 0.90
        minimum = 0.75
        
        assert accuracy > minimum
        assert accuracy < target
        assert accuracy >= 0.87
        assert accuracy <= 1.0
        assert accuracy != target
    
    def test_logical_operators(self):
        """Test logical operators for AI decision making."""
        model_trained = True
        data_ready = True
        user_auth = False
        
        # AND operator
        can_predict = model_trained and data_ready
        assert can_predict == True
        
        # OR operator
        system_ready = model_trained or data_ready
        assert system_ready == True
        
        # NOT operator
        needs_auth = not user_auth
        assert needs_auth == True

def test_lesson_completeness():
    """Test that all required concepts are covered."""
    # This test ensures the lesson covers all required topics
    
    # Test that we can create and use all required data types
    name = "AI Agent"              # String
    confidence = 0.87              # Float
    interactions = 5               # Integer
    is_active = True              # Boolean
    messages = ["Hello", "Hi"]     # List
    
    # Test that we can perform AI-relevant operations
    avg_confidence = confidence * 0.9  # Arithmetic
    is_confident = confidence > 0.8    # Comparison
    ready = is_active and len(messages) > 0  # Logical
    
    # All should work without errors
    assert isinstance(name, str)
    assert isinstance(confidence, float)
    assert isinstance(interactions, int)
    assert isinstance(is_active, bool)
    assert isinstance(messages, list)
    assert ready == True

if __name__ == "__main__":
    # Run tests when script is executed directly
    print("🧪 Running Day 2 Tests...")
    print("=" * 30)
    
    # Simple test runner (without pytest)
    test_classes = [TestDataTypes(), TestAIAgent(), TestOperators()]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📋 Testing {class_name}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"  ✅ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
    
    # Run standalone test
    try:
        test_lesson_completeness()
        print(f"  ✅ test_lesson_completeness")
        total_tests += 1
        passed_tests += 1
    except Exception as e:
        print(f"  ❌ test_lesson_completeness: {e}")
        total_tests += 1
    
    # Results
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! You understand the fundamentals!")
    else:
        print("📚 Some tests failed. Review the lesson and try again.")
        
    print("\n🚀 Ready for Day 3: Control Flow!")
