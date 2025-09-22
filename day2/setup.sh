#!/bin/bash

# Generate Day 2 Lesson Files: Variables, Data Types, and Operators for AI
# Course: 180-Day AI and Machine Learning from Scratch

echo "🚀 Generating Day 2 lesson files..."

# Create project directory structure
mkdir -p day2_variables_datatypes
cd day2_variables_datatypes

# Generate setup.sh - Environment setup script
echo "📦 Creating setup.sh..."
cat > setup.sh << 'EOF'
#!/bin/bash

echo "🔧 Setting up Python environment for Day 2: Variables, Data Types, and Operators"

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv ai_course_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source ai_course_env/Scripts/activate
else
    source ai_course_env/bin/activate
fi

# Install requirements
echo "📥 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete! Run 'source ai_course_env/bin/activate' to start coding."
echo "🎯 Then run: python lesson_code.py"
EOF

chmod +x setup.sh

# Generate requirements.txt - Dependencies
echo "📋 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Day 2: Variables, Data Types, and Operators for AI
# Python 3.11+ required

# Core data science libraries
jupyter==1.0.0
ipython==8.17.2

# For future AI/ML lessons
numpy==1.25.2
pandas==2.1.4

# Testing
pytest==7.4.3

# Code formatting (optional but recommended)
black==23.11.0
EOF

# Generate lesson_code.py - Main lesson implementation
echo "🐍 Creating lesson_code.py..."
cat > lesson_code.py << 'EOF'
#!/usr/bin/env python3
"""
Day 2: Variables, Data Types, and Operators for AI
180-Day AI and Machine Learning Course

This lesson teaches Python data types specifically for AI/ML applications.
Focus: How variables and operators form the foundation of AI systems.
"""

import random
from typing import List, Dict, Union

def demonstrate_ai_data_types():
    """Demonstrate how different Python data types are used in AI systems."""
    
    print("🤖 AI Data Types Demonstration")
    print("=" * 40)
    
    # Strings - The language of AI communication
    print("\n📝 STRINGS - AI Communication")
    user_prompt = "What's the weather like today?"
    model_name = "ChatHelper-v1"
    system_message = "You are a helpful AI assistant"
    
    print(f"User Prompt: {user_prompt}")
    print(f"Model Name: {model_name}")
    print(f"System Message: {system_message}")
    
    # String operations common in AI
    prompt_length = len(user_prompt)
    words_in_prompt = len(user_prompt.split())
    print(f"Prompt Analysis: {prompt_length} characters, {words_in_prompt} words")
    
    # Numbers - The mathematical brain of AI
    print("\n🔢 NUMBERS - AI Mathematics")
    
    # Integers for counting and configuration
    max_tokens = 150
    temperature_setting = 7  # Will convert to 0.7
    training_epochs = 1000
    
    # Floats for AI calculations
    model_confidence = 0.87
    learning_rate = 0.001
    accuracy_score = 94.2
    temperature = temperature_setting / 10  # Convert to proper scale
    
    print(f"Model Settings:")
    print(f"  Max Tokens: {max_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Current Accuracy: {accuracy_score}%")
    print(f"  Confidence: {model_confidence}")
    
    # Boolean - Decision gates of AI
    print("\n🚦 BOOLEANS - AI Decision Making")
    is_training_mode = False
    user_authenticated = True
    use_cache = True
    model_ready = True
    
    print(f"System Status:")
    print(f"  Training Mode: {is_training_mode}")
    print(f"  User Authenticated: {user_authenticated}")
    print(f"  Cache Enabled: {use_cache}")
    print(f"  Model Ready: {model_ready}")
    
    # Lists - Data collections for AI
    print("\n📚 LISTS - AI Data Collections")
    conversation_history = [
        "Hello!",
        "How are you?",
        "I'm doing well, thanks!",
        "Can you help me with something?"
    ]
    
    confidence_scores = [0.92, 0.87, 0.94, 0.81]
    supported_languages = ["English", "Spanish", "French", "German", "Japanese"]
    
    print(f"Conversation History ({len(conversation_history)} messages):")
    for i, message in enumerate(conversation_history):
        print(f"  {i+1}: {message}")
    
    print(f"\nConfidence Scores: {confidence_scores}")
    print(f"Average Confidence: {sum(confidence_scores) / len(confidence_scores):.2f}")
    print(f"Supported Languages: {', '.join(supported_languages)}")

class SimpleAIAgent:
    """A simple AI agent demonstrating how data types work together."""
    
    def __init__(self, name: str):
        # Agent identity and state
        self.name = name
        self.is_active = True
        self.version = "1.0.0"
        
        # Data storage
        self.conversation_history: List[str] = []
        self.confidence_scores: List[float] = []
        self.user_preferences: Dict[str, Union[str, bool, int]] = {}
        
        # Performance tracking
        self.total_interactions = 0
        self.successful_responses = 0
        self.average_confidence = 0.0
        
        print(f"🤖 AI Agent '{self.name}' initialized (v{self.version})")
    
    def process_input(self, user_input: str) -> str:
        """Process user input and generate response with confidence tracking."""
        
        # Input validation
        if not isinstance(user_input, str) or len(user_input.strip()) == 0:
            return "Error: Invalid input provided"
        
        # Store conversation
        self.conversation_history.append(user_input)
        self.total_interactions += 1
        
        # Simulate AI processing with confidence calculation
        # In real AI: this would be actual model inference
        word_count = len(user_input.split())
        base_confidence = random.uniform(0.7, 0.95)
        
        # Adjust confidence based on input complexity
        if word_count <= 3:
            confidence = min(base_confidence + 0.05, 0.98)
        elif word_count > 10:
            confidence = max(base_confidence - 0.1, 0.6)
        else:
            confidence = base_confidence
        
        confidence = round(confidence, 3)
        self.confidence_scores.append(confidence)
        
        # Update statistics using operators
        self.average_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
        
        # Decision making with boolean logic
        if confidence > 0.85:
            response_quality = "high"
            self.successful_responses += 1
        elif confidence > 0.75:
            response_quality = "good"
            self.successful_responses += 1
        else:
            response_quality = "moderate"
        
        # Generate response based on input analysis
        if "hello" in user_input.lower() or "hi" in user_input.lower():
            response = f"Hello! I'm {self.name}, ready to help you."
        elif "?" in user_input:
            response = f"That's an interesting question. Let me think about that..."
        elif "weather" in user_input.lower():
            response = "I'd need to connect to a weather service to get current conditions."
        else:
            response = f"I understand you said: '{user_input[:50]}...' Let me help with that."
        
        print(f"🎯 Processed: {confidence:.1%} confidence ({response_quality} quality)")
        return response
    
    def get_agent_status(self) -> Dict:
        """Return comprehensive agent status using all data types."""
        
        # Calculate success rate
        success_rate = (self.successful_responses / max(self.total_interactions, 1)) * 100
        
        return {
            # String data
            "agent_name": self.name,
            "version": self.version,
            "status": "active" if self.is_active else "inactive",
            
            # Numeric data
            "total_interactions": self.total_interactions,
            "successful_responses": self.successful_responses,
            "success_rate": round(success_rate, 1),
            "average_confidence": round(self.average_confidence, 3),
            
            # Boolean data
            "is_active": self.is_active,
            "has_conversation_history": len(self.conversation_history) > 0,
            
            # List data
            "recent_conversations": self.conversation_history[-3:],  # Last 3
            "recent_confidence_scores": self.confidence_scores[-3:],
            
            # Performance metrics
            "min_confidence": min(self.confidence_scores) if self.confidence_scores else 0,
            "max_confidence": max(self.confidence_scores) if self.confidence_scores else 0
        }
    
    def demonstrate_operators(self):
        """Show how operators are used in AI systems."""
        
        print(f"\n🔧 OPERATORS in AI Systems")
        print("=" * 30)
        
        # Arithmetic operators in AI
        print("➕ Arithmetic Operators:")
        batch_size = 32
        total_samples = 1000
        num_batches = total_samples // batch_size  # Floor division
        remainder_samples = total_samples % batch_size  # Modulo
        
        print(f"  Training Data: {total_samples} samples")
        print(f"  Batch Size: {batch_size}")
        print(f"  Number of Batches: {num_batches}")
        print(f"  Remaining Samples: {remainder_samples}")
        
        # Comparison operators for thresholds
        print("\n🎯 Comparison Operators (AI Thresholds):")
        current_accuracy = 0.87
        target_accuracy = 0.90
        minimum_confidence = 0.75
        
        print(f"  Current Accuracy: {current_accuracy:.1%}")
        print(f"  Target Reached: {current_accuracy >= target_accuracy}")
        print(f"  Above Minimum: {current_accuracy > minimum_confidence}")
        print(f"  Needs Improvement: {current_accuracy < target_accuracy}")
        
        # Logical operators for decision making
        print("\n🧠 Logical Operators (AI Decision Logic):")
        model_trained = True
        data_validated = True
        user_authorized = True
        system_healthy = True
        
        can_make_prediction = model_trained and data_validated
        can_serve_user = user_authorized and system_healthy
        system_ready = can_make_prediction and can_serve_user
        
        print(f"  Model Trained: {model_trained}")
        print(f"  Data Validated: {data_validated}")
        print(f"  Can Make Prediction: {can_make_prediction}")
        print(f"  Can Serve User: {can_serve_user}")
        print(f"  System Ready: {system_ready}")

def main():
    """Main lesson demonstration."""
    
    print("🎓 Day 2: Variables, Data Types, and Operators for AI")
    print("=" * 55)
    
    # Part 1: Basic data types in AI context
    demonstrate_ai_data_types()
    
    print("\n" + "=" * 55)
    
    # Part 2: Interactive AI agent demonstration
    print("🤖 INTERACTIVE AI AGENT DEMO")
    
    # Create an AI agent
    agent = SimpleAIAgent("StudyBuddy")
    
    # Simulate some interactions
    test_inputs = [
        "Hello there!",
        "What's the weather like today?",
        "Can you help me learn Python?",
        "How do neural networks work?",
        "Thanks for your help!"
    ]
    
    print(f"\n📝 Processing {len(test_inputs)} test interactions...")
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n[Interaction {i}]")
        print(f"User: {user_input}")
        response = agent.process_input(user_input)
        print(f"Agent: {response}")
    
    # Show agent status
    print(f"\n📊 AGENT STATUS REPORT")
    print("=" * 25)
    status = agent.get_agent_status()
    
    for key, value in status.items():
        if isinstance(value, list):
            print(f"{key}: {value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Part 3: Operators demonstration
    agent.demonstrate_operators()
    
    print(f"\n🎉 Lesson Complete!")
    print("You've learned how Python's fundamental data types")
    print("form the building blocks of AI systems!")
    
    print(f"\n🚀 Next: Day 3 - Control Flow (Making AI Decisions)")

if __name__ == "__main__":
    main()
EOF

chmod +x lesson_code.py

# Generate test_lesson.py - Simple tests
echo "🧪 Creating test_lesson.py..."
cat > test_lesson.py << 'EOF'
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
EOF

chmod +x test_lesson.py

# Generate README.md - Quick start guide
echo "📖 Creating README.md..."
cat > README.md << 'EOF'
# Day 2: Variables, Data Types, and Operators for AI

Welcome to Day 2 of your 180-day AI/ML journey! Today we learn how Python's fundamental data types form the building blocks of AI systems.

## 🎯 Learning Objectives

By the end of this lesson, you'll be able to:
- Use Python variables to store AI-relevant data
- Understand how strings, numbers, booleans, and lists power AI systems
- Build a simple "AI agent memory system"
- Apply operators for AI calculations and decision-making

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh

# Activate environment (if not auto-activated)
source ai_course_env/bin/activate  # Linux/Mac
# OR
ai_course_env\Scripts\activate     # Windows
```

### 2. Run the Main Lesson
```bash
python lesson_code.py
```

### 3. Verify Your Learning
```bash
python test_lesson.py
```

## 📋 What You'll Build

- **AI Agent Memory System**: A simple class that demonstrates how Python data types work together in AI applications
- **Data Type Demonstrations**: Interactive examples showing how strings, numbers, booleans, and lists are used in real AI systems
- **Operator Showcase**: See how arithmetic, comparison, and logical operators power AI decision-making

## 🔧 Key Concepts Covered

### Data Types for AI
- **Strings**: AI communication and text processing
- **Integers**: Counting, batch sizes, epochs
- **Floats**: Confidence scores, learning rates, probabilities
- **Booleans**: System states and decision gates
- **Lists**: Datasets, conversation history, predictions

### Operators in AI
- **Arithmetic**: Batch calculations, statistical operations
- **Comparison**: Threshold checking, performance evaluation
- **Logical**: Decision making, system state validation

## 📁 Project Structure

```
day2_variables_datatypes/
├── setup.sh              # Environment setup
├── lesson_code.py         # Main lesson implementation
├── test_lesson.py         # Verification tests
├── requirements.txt       # Python dependencies
├── README.md             # This guide
└── ai_course_env/        # Virtual environment (created by setup)
```

## 🎮 Interactive Elements

1. **AI Agent Demo**: Create and interact with a simple AI agent
2. **Data Type Explorer**: See how each Python type applies to AI
3. **Operator Workshop**: Practice calculations used in real AI systems

## 🧪 Testing Your Knowledge

Run `python test_lesson.py` to verify you understand:
- String operations for text processing
- Numeric calculations for AI metrics
- Boolean logic for decision making
- List operations for data management
- The SimpleAIAgent class functionality

## 🔗 Connection to AI/ML

Everything you learn today directly applies to:
- **ChatGPT-style systems**: Conversation history (lists), confidence scores (floats)
- **Image Recognition**: Probability arrays (lists of floats), classification decisions (booleans)
- **Training Systems**: Batch processing (integers), learning rates (floats), model states (booleans)

## 🎯 Success Criteria

You're ready for Day 3 when you can:
- ✅ Create variables for different types of AI data
- ✅ Use operators to perform AI-relevant calculations
- ✅ Understand how simple Python constructs build complex AI systems
- ✅ Pass all tests in `test_lesson.py`

## 🚀 What's Next?

**Day 3: Control Flow** - Learn if/else statements and loops that let your AI agent make intelligent decisions based on the data you learned to store today.

## 💡 Tips for Success

1. **Think AI-First**: Every example relates to real AI applications
2. **Practice Interactively**: Modify the code and see what happens
3. **Connect the Dots**: See how simple variables become complex AI systems
4. **Test Often**: Run the tests to verify your understanding

## 🆘 Need Help?

If you encounter issues:
1. Check that Python 3.11+ is installed: `python3 --version`
2. Ensure virtual environment is activated (prompt should show `(ai_course_env)`)
3. Run tests to identify specific knowledge gaps
4. Review the lesson article for conceptual understanding

---

**Remember**: Today's simple variables and operators are the foundation of every AI system you'll ever build. Master these fundamentals, and you're on your way to creating amazing AI applications! 🤖✨
EOF

# Make all scripts executable
chmod +x *.sh *.py

echo "✅ All lesson files generated successfully!"
echo ""
echo "📁 Generated files:"
echo "   - setup.sh (Environment setup)"
echo "   - lesson_code.py (Main lesson)"
echo "   - test_lesson.py (Knowledge verification)"
echo "   - requirements.txt (Dependencies)"
echo "   - README.md (Quick start guide)"
echo ""
echo "🚀 To start learning:"
echo "   1. chmod +x setup.sh && ./setup.sh"
echo "   2. python lesson_code.py"
echo "   3. python test_lesson.py"
echo ""
echo "🎓 Happy learning!"