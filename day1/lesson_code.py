#!/usr/bin/env python3
"""
Day 1: Introduction to Programming and Python Basics for AI
=========================================================

This module demonstrates core Python concepts used in AI/ML applications.
Focus: Variables, data structures, functions, and basic AI-like processing.

Author: AI/ML Course
Date: Day 1 of 180-day journey
"""

import json
import random
from datetime import datetime
from typing import List, Dict, Any


class SmartResponder:
    """
    A simple AI-like system that demonstrates core Python concepts
    used in real AI applications.
    
    This class shows how AI systems:
    1. Store information (variables, data structures)
    2. Process input (functions)
    3. Learn patterns (basic pattern matching)
    4. Generate responses (rule-based generation)
    """
    
    def __init__(self, name: str = "AI Assistant"):
        """Initialize the Smart Responder with basic AI components."""
        self.name = name
        self.conversation_memory: List[Dict[str, Any]] = []
        self.user_profile: Dict[str, Any] = {
            "interactions": 0,
            "topics_discussed": [],
            "sentiment_history": [],
            "preferred_response_style": "friendly"
        }
        
        # AI Training Data: Response patterns (like a mini AI model)
        self.response_patterns = {
            "greeting": {
                "keywords": ["hello", "hi", "hey", "good morning", "good afternoon", "hey there"],
                "responses": [
                    "Hello! Ready to explore AI together? 🤖",
                    "Hi there! I'm excited to learn with you today!",
                    "Hey! Let's dive into some AI magic! ✨"
                ]
            },
            "question": {
                "keywords": ["what", "how", "why", "when", "where", "which", "?"],
                "responses": [
                    "Great question! That's exactly the kind of curiosity that drives AI innovation.",
                    "I love questions! AI systems learn by asking and answering questions too.",
                    "Interesting question! Let me process that like an AI would..."
                ]
            },
            "emotion_positive": {
                "keywords": ["happy", "excited", "great", "awesome", "love", "amazing", "fantastic"],
                "responses": [
                    "I can detect positive sentiment! AI emotion recognition is fascinating.",
                    "Your enthusiasm is contagious! This is how AI systems learn to recognize emotions.",
                    "Positive vibes detected! 😊 AI is getting better at understanding human emotions."
                ]
            },
            "emotion_negative": {
                "keywords": ["sad", "frustrated", "confused", "difficult", "hard", "worried"],
                "responses": [
                    "I sense some challenge in your message. Remember, every AI expert was once a beginner!",
                    "It's okay to find this challenging - AI systems learn from mistakes too.",
                    "Detected some difficulty - that's normal! Even AI models need lots of practice."
                ]
            },
            "ai_related": {
                "keywords": ["ai", "artificial intelligence", "machine learning", "ml", "algorithm", "neural", "model"],
                "responses": [
                    "Now we're talking AI! You're already thinking like a future AI engineer.",
                    "AI topic detected! This is exactly what we're learning to build.",
                    "Perfect! You're connecting the dots between Python and AI systems."
                ]
            }
        }
    
    def analyze_input(self, user_input: str) -> Dict[str, Any]:
        """
        Analyze user input to determine response type and extract information.
        This simulates how AI systems process and classify text.
        
        Args:
            user_input: Raw text from user
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "text": user_input,
            "length": len(user_input),
            "word_count": len(user_input.split()),
            "timestamp": datetime.now().isoformat(),
            "detected_patterns": [],
            "sentiment": "neutral"
        }
        
        text_lower = user_input.lower()
        
        # Pattern detection (like AI classification)
        for pattern_type, pattern_data in self.response_patterns.items():
            for keyword in pattern_data["keywords"]:
                if keyword in text_lower:
                    analysis["detected_patterns"].append(pattern_type)
                    break
        
        # Simple sentiment analysis
        if any(pattern in analysis["detected_patterns"] for pattern in ["emotion_positive", "greeting"]):
            analysis["sentiment"] = "positive"
        elif "emotion_negative" in analysis["detected_patterns"]:
            analysis["sentiment"] = "negative"
        
        return analysis
    
    def generate_response(self, analysis: Dict[str, Any]) -> str:
        """
        Generate contextual response based on input analysis.
        This demonstrates how AI systems create relevant outputs.
        
        Args:
            analysis: Results from analyze_input()
            
        Returns:
            Generated response string
        """
        # Update user profile (like AI learning)
        self.user_profile["interactions"] += 1
        self.user_profile["sentiment_history"].append(analysis["sentiment"])
        
        # Extract topics for user profile
        for pattern in analysis["detected_patterns"]:
            if pattern not in self.user_profile["topics_discussed"]:
                self.user_profile["topics_discussed"].append(pattern)
        
        # Response generation logic
        if analysis["detected_patterns"]:
            # Pick the first detected pattern for response
            primary_pattern = analysis["detected_patterns"][0]
            possible_responses = self.response_patterns[primary_pattern]["responses"]
            base_response = random.choice(possible_responses)
        else:
            # Default response for unrecognized input
            base_response = f"I'm processing your message of {analysis['word_count']} words. Keep talking - AI learns from conversation!"
        
        # Add contextual information (like AI personalization)
        if self.user_profile["interactions"] > 1:
            interaction_context = f" (This is our {self.user_profile['interactions']} interaction!)"
            base_response += interaction_context
        
        return base_response
    
    def save_conversation(self, user_input: str, analysis: Dict[str, Any], response: str):
        """
        Save conversation to memory (like AI systems store training data).
        
        Args:
            user_input: Original user message
            analysis: Analysis results
            response: Generated response
        """
        conversation_entry = {
            "timestamp": analysis["timestamp"],
            "user_input": user_input,
            "analysis": analysis,
            "ai_response": response,
            "interaction_number": self.user_profile["interactions"]
        }
        
        self.conversation_memory.append(conversation_entry)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the conversation (like AI system monitoring).
        
        Returns:
            Dictionary with conversation statistics
        """
        return {
            "total_interactions": len(self.conversation_memory),
            "topics_discussed": self.user_profile["topics_discussed"],
            "sentiment_distribution": {
                "positive": self.user_profile["sentiment_history"].count("positive"),
                "neutral": self.user_profile["sentiment_history"].count("neutral"),
                "negative": self.user_profile["sentiment_history"].count("negative")
            },
            "average_message_length": sum(
                len(entry["user_input"]) for entry in self.conversation_memory
            ) / len(self.conversation_memory) if self.conversation_memory else 0
        }
    
    def run_interactive_session(self):
        """
        Run an interactive conversation session.
        This is like a simple chatbot interface.
        """
        print(f"🤖 {self.name} v1.0 - Your First AI Program!")
        print("=" * 50)
        print("This program demonstrates core Python concepts used in AI:")
        print("• Variables store information (like AI memory)")
        print("• Functions process data (like AI algorithms)")
        print("• Data structures organize knowledge (like AI databases)")
        print("• Pattern matching classifies input (like AI understanding)")
        print("\nType 'stats' to see conversation analytics")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print(f"\n🤖 Thanks for exploring AI basics with me!")
                    print("Final conversation stats:")
                    stats = self.get_conversation_stats()
                    print(json.dumps(stats, indent=2))
                    print("\nKeep learning - tomorrow we'll explore more AI concepts! 🚀")
                    break
                
                elif user_input.lower() == 'stats':
                    print("\n📊 Conversation Analytics (like AI monitoring):")
                    stats = self.get_conversation_stats()
                    print(json.dumps(stats, indent=2))
                    continue
                
                elif not user_input:
                    print("🤖 I'm listening... try saying something!")
                    continue
                
                # Core AI pipeline: Analyze → Generate → Store
                analysis = self.analyze_input(user_input)
                response = self.generate_response(analysis)
                self.save_conversation(user_input, analysis, response)
                
                print(f"🤖 {response}")
                
            except KeyboardInterrupt:
                print("\n\n🤖 Session interrupted. Thanks for learning!")
                break
            except Exception as e:
                print(f"🤖 Oops! AI error: {e}")
                print("This is how we debug AI systems - by catching and fixing errors!")


def demonstrate_ai_data_structures():
    """
    Demonstrate how AI systems use Python data structures.
    This function shows practical examples of AI data handling.
    """
    print("\n🔍 AI Data Structures Demo")
    print("=" * 30)
    
    # Lists: Sequential data (like training examples)
    training_examples = [
        "Hello, how are you?",
        "What's the weather like?",
        "Can you help me code?",
        "I love learning AI!"
    ]
    print(f"Training Examples (List): {len(training_examples)} samples")
    
    # Dictionaries: Structured data (like user profiles)
    user_profile = {
        "name": "AI Student",
        "skill_level": "beginner",
        "interests": ["python", "ai", "machine learning"],
        "progress": {
            "day": 1,
            "completed_lessons": [],
            "total_score": 0
        }
    }
    print(f"User Profile (Dict): {user_profile['name']} - {user_profile['skill_level']}")
    
    # Nested structures: Complex AI data (like model configurations)
    ai_model_config = {
        "model_name": "SmartResponder",
        "version": "1.0",
        "parameters": {
            "response_patterns": 5,
            "memory_size": 100,
            "learning_rate": "adaptive"
        },
        "training_data": training_examples,
        "performance_metrics": {
            "accuracy": 0.85,
            "response_time": "0.1s",
            "user_satisfaction": "high"
        }
    }
    
    print(f"AI Model Config: {ai_model_config['model_name']} v{ai_model_config['version']}")
    print(f"Performance: {ai_model_config['performance_metrics']['accuracy']} accuracy")
    
    return ai_model_config


def main():
    """
    Main function to run the Day 1 lesson.
    This demonstrates the complete AI development workflow.
    """
    print("🚀 Welcome to Day 1: Python Basics for AI!")
    print("Today you'll learn the building blocks of AI systems.\n")
    
    # Demonstrate AI data structures
    model_config = demonstrate_ai_data_structures()
    
    # Create and run the Smart Responder
    print("\n🤖 Starting Interactive AI Session...")
    ai_assistant = SmartResponder("Day 1 AI Assistant")
    
    # Run the interactive session
    ai_assistant.run_interactive_session()


if __name__ == "__main__":
    main()