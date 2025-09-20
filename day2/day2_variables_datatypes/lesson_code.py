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
