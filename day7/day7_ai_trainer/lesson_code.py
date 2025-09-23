"""
Day 7: AI Assistant Training Simulator
A command-line game that teaches Python concepts for AI development
"""

import random
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class AIAssistant:
    """
    Represents an AI assistant that can learn and respond to user input.
    
    This class demonstrates core AI concepts:
    - State management (skills, experience, confidence)
    - Pattern matching (input classification)
    - Learning from examples (training function)
    """
    
    def __init__(self, name: str):
        self.name = name
        self.skills: Dict[str, List[str]] = {
            "greeting": ["Hello!", "Hi there!", "Good day!"],
            "farewell": ["Goodbye!", "See you later!", "Take care!"],
            "thanks": ["You're welcome!", "Happy to help!", "No problem!"]
        }
        self.experience = 0
        self.confidence = 0.5
        self.conversation_history: List[Dict] = []
    
    def classify_input(self, user_input: str) -> str:
        """
        Classify user input into skill categories.
        In real AI: This would be a classifier model or NLP pipeline.
        """
        input_lower = user_input.lower()
        
        # Simple keyword matching (in production: use ML models)
        greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon"]
        farewell_words = ["bye", "goodbye", "see you", "farewell", "take care"]
        thanks_words = ["thank", "thanks", "appreciate", "grateful"]
        
        if any(word in input_lower for word in greeting_words):
            return "greeting"
        elif any(word in input_lower for word in farewell_words):
            return "farewell"
        elif any(word in input_lower for word in thanks_words):
            return "thanks"
        else:
            return "unknown"
    
    def respond(self, user_input: str) -> str:
        """
        Generate a response based on learned skills.
        Demonstrates: Decision making, confidence tracking, memory updates.
        """
        skill_category = self.classify_input(user_input)
        
        # Log conversation for analysis
        self.conversation_history.append({
            "input": user_input,
            "category": skill_category,
            "timestamp": datetime.now().isoformat()
        })
        
        if skill_category in self.skills and self.skills[skill_category]:
            # Choose response and boost confidence
            response = random.choice(self.skills[skill_category])
            self.confidence = min(1.0, self.confidence + 0.05)
            return response
        else:
            # Handle unknown input
            self.confidence = max(0.1, self.confidence - 0.02)
            return f"I'm still learning about that. Could you teach me what to say when someone says '{user_input}'?"
    
    def learn(self, skill_category: str, example_response: str) -> bool:
        """
        Add new training examples to the assistant.
        Demonstrates: Knowledge base updates, learning from examples.
        """
        if not skill_category or not example_response:
            return False
            
        if skill_category not in self.skills:
            self.skills[skill_category] = []
        
        # Avoid duplicates
        if example_response not in self.skills[skill_category]:
            self.skills[skill_category].append(example_response)
            self.experience += 1
            self.confidence = min(1.0, self.confidence + 0.1)
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Return current assistant statistics."""
        return {
            "name": self.name,
            "experience": self.experience,
            "confidence": round(self.confidence, 2),
            "skills_count": len(self.skills),
            "total_responses": sum(len(responses) for responses in self.skills.values()),
            "conversations": len(self.conversation_history)
        }
    
    def save_progress(self, filename: Optional[str] = None) -> str:
        """Save assistant state to JSON file (like model checkpoints)."""
        if filename is None:
            filename = f"{self.name.lower().replace(' ', '_')}_assistant.json"
        
        data = {
            "name": self.name,
            "skills": self.skills,
            "experience": self.experience,
            "confidence": self.confidence,
            "conversation_history": self.conversation_history,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename
    
    @classmethod
    def load_progress(cls, filename: str) -> 'AIAssistant':
        """Load assistant from saved file (like loading a trained model)."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        assistant = cls(data["name"])
        assistant.skills = data["skills"]
        assistant.experience = data["experience"]
        assistant.confidence = data["confidence"]
        assistant.conversation_history = data.get("conversation_history", [])
        
        return assistant


class TrainingSimulator:
    """
    Main game interface for training AI assistants.
    Demonstrates: User interaction, system orchestration, data persistence.
    """
    
    def __init__(self):
        self.assistants: Dict[str, AIAssistant] = {}
        self.current_assistant: Optional[AIAssistant] = None
    
    def create_assistant(self) -> AIAssistant:
        """Create a new AI assistant."""
        print("\n🤖 Creating your AI Assistant...")
        name = input("Enter assistant name: ").strip()
        
        if not name:
            name = f"Assistant_{random.randint(1000, 9999)}"
            print(f"Using default name: {name}")
        
        assistant = AIAssistant(name)
        self.assistants[name] = assistant
        self.current_assistant = assistant
        
        print(f"✅ Created {name}! Ready for training.")
        return assistant
    
    def load_assistant(self) -> bool:
        """Load an existing assistant from file."""
        files = [f for f in os.listdir('.') if f.endswith('_assistant.json')]
        
        if not files:
            print("No saved assistants found.")
            return False
        
        print("\nSaved assistants:")
        for i, file in enumerate(files, 1):
            print(f"  {i}. {file}")
        
        try:
            choice = int(input("Select assistant (number): ")) - 1
            if 0 <= choice < len(files):
                assistant = AIAssistant.load_progress(files[choice])
                self.assistants[assistant.name] = assistant
                self.current_assistant = assistant
                print(f"✅ Loaded {assistant.name}!")
                return True
        except (ValueError, IndexError):
            pass
        
        print("Invalid selection.")
        return False
    
    def chat_mode(self):
        """Interactive chat with the current assistant."""
        if not self.current_assistant:
            print("No assistant selected!")
            return
        
        print(f"\n💬 Chatting with {self.current_assistant.name}")
        print("Type 'back' to return to main menu")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'back':
                break
            
            if user_input:
                response = self.current_assistant.respond(user_input)
                print(f"{self.current_assistant.name}: {response}")
    
    def training_mode(self):
        """Train the current assistant with new examples."""
        if not self.current_assistant:
            print("No assistant selected!")
            return
        
        print(f"\n🎓 Training {self.current_assistant.name}")
        print("Teach your assistant new responses!")
        print("Type 'back' to return to main menu")
        
        while True:
            print("\nCurrent skills:", list(self.current_assistant.skills.keys()))
            
            skill = input("Skill category (or 'back'): ").strip()
            if skill.lower() == 'back':
                break
            
            example = input("Example response: ").strip()
            
            if skill and example:
                if self.current_assistant.learn(skill, example):
                    print(f"✅ {self.current_assistant.name} learned: '{example}' for '{skill}'")
                else:
                    print("❌ Could not add this example (might be duplicate)")
    
    def show_stats(self):
        """Display current assistant statistics."""
        if not self.current_assistant:
            print("No assistant selected!")
            return
        
        stats = self.current_assistant.get_stats()
        
        print(f"\n📊 {stats['name']} Statistics:")
        print(f"  Experience Points: {stats['experience']}")
        print(f"  Confidence Level: {stats['confidence']}")
        print(f"  Skill Categories: {stats['skills_count']}")
        print(f"  Total Responses: {stats['total_responses']}")
        print(f"  Conversations: {stats['conversations']}")
        
        print(f"\n🧠 Knowledge Base:")
        for skill, responses in self.current_assistant.skills.items():
            print(f"  {skill}: {len(responses)} responses")
    
    def save_assistant(self):
        """Save current assistant to file."""
        if not self.current_assistant:
            print("No assistant selected!")
            return
        
        filename = self.current_assistant.save_progress()
        print(f"✅ Saved {self.current_assistant.name} to {filename}")
    
    def run(self):
        """Main game loop."""
        print("🤖 Welcome to AI Assistant Training Simulator!")
        print("Learn Python concepts while building AI systems")
        
        while True:
            print("\n" + "="*50)
            if self.current_assistant:
                print(f"Current Assistant: {self.current_assistant.name}")
            else:
                print("No assistant selected")
            
            print("\nOptions:")
            print("  1. Create new assistant")
            print("  2. Load existing assistant")
            if self.current_assistant:
                print("  3. Chat with assistant")
                print("  4. Train assistant")
                print("  5. View statistics")
                print("  6. Save assistant")
            print("  0. Quit")
            
            choice = input("\nChoose option: ").strip()
            
            if choice == "1":
                self.create_assistant()
            elif choice == "2":
                self.load_assistant()
            elif choice == "3" and self.current_assistant:
                self.chat_mode()
            elif choice == "4" and self.current_assistant:
                self.training_mode()
            elif choice == "5" and self.current_assistant:
                self.show_stats()
            elif choice == "6" and self.current_assistant:
                self.save_assistant()
            elif choice == "0":
                if self.current_assistant:
                    save_choice = input("Save current assistant before quitting? (y/n): ").lower()
                    if save_choice == 'y':
                        self.save_assistant()
                
                print("Thanks for using AI Assistant Training Simulator!")
                print("You've learned key concepts for building AI systems! 🚀")
                break
            else:
                print("Invalid option or no assistant selected.")


def main():
    """Entry point for the AI Assistant Training Simulator."""
    try:
        simulator = TrainingSimulator()
        simulator.run()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted. Goodbye! 👋")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("This is a great debugging opportunity - check your code!")


if __name__ == "__main__":
    main()
