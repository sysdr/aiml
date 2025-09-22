#!/usr/bin/env python3
"""
Test Suite for Day 5: Dictionaries and Sets
Verify student understanding with practical tests
"""

import pytest
import json
from lesson_code import AIDataManager


class TestAIDataManager:
    """Test the AIDataManager class functionality"""
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.ai_manager = AIDataManager()
    
    def test_add_valid_model_config(self):
        """Test adding valid model configuration"""
        config = {"temperature": 0.7, "max_tokens": 1000}
        self.ai_manager.add_model_config("test_model", config)
        
        assert "test_model" in self.ai_manager.model_configs
        assert self.ai_manager.model_configs["test_model"]["temperature"] == 0.7
    
    def test_invalid_temperature_config(self):
        """Test that invalid temperature raises error"""
        config = {"temperature": 1.5, "max_tokens": 1000}  # Invalid temperature
        
        with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
            self.ai_manager.add_model_config("invalid_model", config)
    
    def test_missing_required_config(self):
        """Test that missing required keys raise error"""
        config = {"temperature": 0.5}  # Missing max_tokens
        
        with pytest.raises(ValueError, match="Missing required config keys"):
            self.ai_manager.add_model_config("incomplete_model", config)
    
    def test_data_deduplication(self):
        """Test that data processing removes duplicates correctly"""
        raw_data = ["item1", "item2", "item1", "item3", "item2"]
        clean_data = self.ai_manager.process_training_data("test_dataset", raw_data)
        
        assert len(clean_data) == 3  # Should have 3 unique items
        assert clean_data == {"item1", "item2", "item3"}
    
    def test_empty_data_processing(self):
        """Test processing empty dataset"""
        clean_data = self.ai_manager.process_training_data("empty_dataset", [])
        
        assert len(clean_data) == 0
        assert isinstance(clean_data, set)
    
    def test_data_overlap_detection(self):
        """Test finding overlap between datasets"""
        data1 = ["a", "b", "c", "d"]
        data2 = ["c", "d", "e", "f"]
        
        self.ai_manager.process_training_data("dataset1", data1)
        self.ai_manager.process_training_data("dataset2", data2)
        
        overlap = self.ai_manager.find_data_overlap("dataset1", "dataset2")
        assert overlap == {"c", "d"}
    
    def test_get_model_config(self):
        """Test retrieving model configurations"""
        config = {"temperature": 0.3, "max_tokens": 500}
        self.ai_manager.add_model_config("test_model", config)
        
        # Test getting full config
        retrieved_config = self.ai_manager.get_model_config("test_model")
        assert retrieved_config == config
        
        # Test getting specific key
        temperature = self.ai_manager.get_model_config("test_model", "temperature")
        assert temperature == 0.3
    
    def test_nonexistent_model_config(self):
        """Test handling of non-existent model configuration"""
        config = self.ai_manager.get_model_config("nonexistent_model")
        
        # Should return default configuration
        assert "temperature" in config
        assert "max_tokens" in config
        assert config["temperature"] == 0.5


class TestDictionaryOperations:
    """Test core dictionary operations for AI"""
    
    def test_nested_dictionary_access(self):
        """Test accessing nested AI response data"""
        ai_response = {
            "choices": [
                {
                    "message": {
                        "content": "Hello, world!",
                        "confidence": 0.95
                    }
                }
            ],
            "usage": {
                "total_tokens": 10
            }
        }
        
        # Test deep access
        content = ai_response["choices"][0]["message"]["content"]
        confidence = ai_response["choices"][0]["message"]["confidence"]
        tokens = ai_response["usage"]["total_tokens"]
        
        assert content == "Hello, world!"
        assert confidence == 0.95
        assert tokens == 10
    
    def test_dictionary_update_operations(self):
        """Test updating model configurations"""
        config = {"temperature": 0.5, "max_tokens": 1000}
        
        # Update existing key
        config["temperature"] = 0.7
        assert config["temperature"] == 0.7
        
        # Add new key
        config["top_p"] = 0.9
        assert "top_p" in config
        assert config["top_p"] == 0.9


class TestSetOperations:
    """Test core set operations for AI"""
    
    def test_set_intersection(self):
        """Test finding common elements (skill overlap)"""
        ml_skills = {"python", "statistics", "pandas"}
        ai_skills = {"python", "tensorflow", "statistics"}
        
        common_skills = ml_skills & ai_skills
        assert common_skills == {"python", "statistics"}
    
    def test_set_union(self):
        """Test combining skill sets"""
        basic_skills = {"python", "git"}
        advanced_skills = {"machine_learning", "deep_learning"}
        
        all_skills = basic_skills | advanced_skills
        assert len(all_skills) == 4
        assert "python" in all_skills
        assert "machine_learning" in all_skills
    
    def test_set_difference(self):
        """Test finding unique skills"""
        data_science_skills = {"python", "sql", "statistics", "visualization"}
        programming_skills = {"python", "javascript", "git"}
        
        data_only = data_science_skills - programming_skills
        assert data_only == {"sql", "statistics", "visualization"}
    
    def test_membership_testing(self):
        """Test fast membership checking"""
        required_skills = {"python", "machine_learning", "statistics"}
        
        assert "python" in required_skills
        assert "javascript" not in required_skills
    
    def test_automatic_deduplication(self):
        """Test that sets automatically remove duplicates"""
        skills_with_duplicates = ["python", "sql", "python", "statistics", "sql"]
        unique_skills = set(skills_with_duplicates)
        
        assert len(unique_skills) == 3
        assert unique_skills == {"python", "sql", "statistics"}


def test_lesson_integration():
    """Integration test for the complete lesson workflow"""
    ai_manager = AIDataManager()
    
    # Add configurations
    ai_manager.add_model_config("chatbot", {
        "temperature": 0.3,
        "max_tokens": 500
    })
    
    # Process data
    raw_data = ["query1", "query2", "query1", "query3"]
    clean_data = ai_manager.process_training_data("test_data", raw_data)
    
    # Verify complete workflow
    assert len(ai_manager.model_configs) == 1
    assert len(clean_data) == 3
    assert "test_data" in ai_manager.processed_data
    
    # Generate summary
    summary = ai_manager.generate_ai_summary()
    assert summary["models_configured"] == 1
    assert summary["datasets_processed"] == 1


def run_interactive_tests():
    """Run tests with output for student verification"""
    print("🧪 Running Day 5 Tests...")
    print("=" * 40)
    
    # Test 1: Dictionary operations
    print("\n📝 Test 1: AI Configuration Management")
    ai_manager = AIDataManager()
    
    try:
        ai_manager.add_model_config("test_model", {
            "temperature": 0.7,
            "max_tokens": 1000
        })
        print("✅ Configuration test passed!")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    # Test 2: Set operations
    print("\n📝 Test 2: Data Deduplication")
    raw_data = ["item1", "item2", "item1", "item3", "item2", "item1"]
    clean_data = ai_manager.process_training_data("test_dataset", raw_data)
    
    expected_size = 3
    if len(clean_data) == expected_size:
        print("✅ Deduplication test passed!")
    else:
        print(f"❌ Deduplication test failed: expected {expected_size}, got {len(clean_data)}")
    
    # Test 3: Data analysis
    print("\n📝 Test 3: Data Overlap Analysis")
    data1 = ["a", "b", "c"]
    data2 = ["b", "c", "d"]
    
    ai_manager.process_training_data("dataset1", data1)
    ai_manager.process_training_data("dataset2", data2)
    overlap = ai_manager.find_data_overlap("dataset1", "dataset2")
    
    if len(overlap) == 2 and overlap == {"b", "c"}:
        print("✅ Overlap analysis test passed!")
    else:
        print(f"❌ Overlap analysis test failed: expected {{'b', 'c'}}, got {overlap}")
    
    print("\n🎉 All interactive tests completed!")
    return True


if __name__ == "__main__":
    # Run interactive tests for immediate feedback
    run_interactive_tests()
    
    # Instructions for running pytest
    print("\n💡 To run full test suite:")
    print("   pip install pytest")
    print("   pytest test_lesson.py -v")
