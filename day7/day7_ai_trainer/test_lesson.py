"""
Test file for Day 7: AI Assistant Training Simulator
Demonstrates testing concepts important for AI development
"""

import unittest
import tempfile
import os
from lesson_code import AIAssistant, TrainingSimulator


class TestAIAssistant(unittest.TestCase):
    """Test the AIAssistant class functionality."""
    
    def setUp(self):
        """Create a fresh assistant for each test."""
        self.assistant = AIAssistant("TestBot")
    
    def test_initialization(self):
        """Test that assistant initializes correctly."""
        self.assertEqual(self.assistant.name, "TestBot")
        self.assertGreater(len(self.assistant.skills), 0)
        self.assertEqual(self.assistant.experience, 0)
        self.assertEqual(self.assistant.confidence, 0.5)
    
    def test_input_classification(self):
        """Test input classification logic."""
        # Test greeting classification
        self.assertEqual(self.assistant.classify_input("hello"), "greeting")
        self.assertEqual(self.assistant.classify_input("Hi there!"), "greeting")
        
        # Test farewell classification
        self.assertEqual(self.assistant.classify_input("goodbye"), "farewell")
        self.assertEqual(self.assistant.classify_input("see you later"), "farewell")
        
        # Test unknown classification
        self.assertEqual(self.assistant.classify_input("random text"), "unknown")
    
    def test_learning(self):
        """Test that assistant can learn new responses."""
        initial_experience = self.assistant.experience
        
        # Test successful learning
        success = self.assistant.learn("test_skill", "test response")
        self.assertTrue(success)
        self.assertEqual(self.assistant.experience, initial_experience + 1)
        self.assertIn("test_skill", self.assistant.skills)
        self.assertIn("test response", self.assistant.skills["test_skill"])
    
    def test_duplicate_learning(self):
        """Test that duplicates are not added."""
        self.assistant.learn("greeting", "Hello!")
        initial_count = len(self.assistant.skills["greeting"])
        
        # Try to add duplicate
        success = self.assistant.learn("greeting", "Hello!")
        self.assertFalse(success)
        self.assertEqual(len(self.assistant.skills["greeting"]), initial_count)
    
    def test_response_generation(self):
        """Test that assistant generates appropriate responses."""
        # Test known input
        response = self.assistant.respond("hello")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Test unknown input
        response = self.assistant.respond("askjdhaksjdh")
        self.assertIn("still learning", response.lower())
    
    def test_stats(self):
        """Test statistics generation."""
        stats = self.assistant.get_stats()
        required_keys = ["name", "experience", "confidence", "skills_count", "total_responses"]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats["name"], "TestBot")
    
    def test_save_and_load(self):
        """Test saving and loading assistant state."""
        # Train the assistant
        self.assistant.learn("test", "test response")
        original_experience = self.assistant.experience
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_assistant.json")
            saved_file = self.assistant.save_progress(filepath)
            self.assertTrue(os.path.exists(saved_file))
            
            # Load new assistant
            loaded_assistant = AIAssistant.load_progress(saved_file)
            self.assertEqual(loaded_assistant.name, self.assistant.name)
            self.assertEqual(loaded_assistant.experience, original_experience)
            self.assertIn("test", loaded_assistant.skills)


class TestTrainingSimulator(unittest.TestCase):
    """Test the TrainingSimulator class functionality."""
    
    def setUp(self):
        """Create a fresh simulator for each test."""
        self.simulator = TrainingSimulator()
    
    def test_initialization(self):
        """Test simulator initializes correctly."""
        self.assertEqual(len(self.simulator.assistants), 0)
        self.assertIsNone(self.simulator.current_assistant)
    
    def test_assistant_management(self):
        """Test creating and managing assistants."""
        # Create assistant
        assistant = AIAssistant("TestBot")
        self.simulator.assistants["TestBot"] = assistant
        self.simulator.current_assistant = assistant
        
        self.assertEqual(len(self.simulator.assistants), 1)
        self.assertIsNotNone(self.simulator.current_assistant)
        self.assertEqual(self.simulator.current_assistant.name, "TestBot")


def run_basic_validation():
    """Run basic validation tests without unittest framework."""
    print("🧪 Running basic validation tests...")
    
    # Test 1: Create assistant
    print("Test 1: Creating AI Assistant...")
    assistant = AIAssistant("Validator")
    assert assistant.name == "Validator"
    print("✅ Assistant creation successful")
    
    # Test 2: Learning
    print("Test 2: Teaching assistant...")
    success = assistant.learn("validation", "This is a test response")
    assert success == True
    assert "validation" in assistant.skills
    print("✅ Learning mechanism working")
    
    # Test 3: Response generation
    print("Test 3: Getting responses...")
    response = assistant.respond("hello")
    assert isinstance(response, str)
    assert len(response) > 0
    print("✅ Response generation working")
    
    # Test 4: Statistics
    print("Test 4: Checking statistics...")
    stats = assistant.get_stats()
    assert "experience" in stats
    assert stats["experience"] > 0
    print("✅ Statistics generation working")
    
    print("\n🎉 All basic validation tests passed!")
    print("Your AI Assistant Training Simulator is ready to use!")


if __name__ == "__main__":
    print("Day 7 Test Suite: AI Assistant Training Simulator")
    print("="*50)
    
    # Run basic validation first
    run_basic_validation()
    
    print("\n" + "="*50)
    print("Running comprehensive test suite...")
    
    # Run full test suite
    unittest.main(verbosity=2)
