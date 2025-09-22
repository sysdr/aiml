#!/usr/bin/env python3
"""
Day 5: Data Structures for AI - Dictionaries and Sets
Interactive lesson on using dictionaries and sets for AI applications
"""

import json
import time
from typing import Dict, Set, List, Union
from collections import defaultdict


class AIDataManager:
    """
    Manages AI model configurations and processes training data.
    Demonstrates real-world usage of dictionaries and sets in AI systems.
    """
    
    def __init__(self):
        self.model_configs: Dict[str, Dict] = {}
        self.processed_data: Dict[str, Set[str]] = {}
        self.performance_stats: Dict[str, Dict] = defaultdict(dict)
    
    def add_model_config(self, model_name: str, config: Dict) -> None:
        """Store AI model configuration with validation"""
        required_keys = {"temperature", "max_tokens"}
        if not required_keys.issubset(config.keys()):
            missing = required_keys - config.keys()
            raise ValueError(f"Missing required config keys: {missing}")
        
        # Validate temperature range (typical AI model constraint)
        if not 0 <= config["temperature"] <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        
        self.model_configs[model_name] = config
        print(f"✅ Added config for {model_name}")
        self._display_config(model_name, config)
    
    def _display_config(self, model_name: str, config: Dict) -> None:
        """Helper to display configuration in user-friendly format"""
        print(f"   📊 Configuration:")
        for key, value in config.items():
            if key == "temperature":
                creativity = "Low (Factual)" if value < 0.3 else "Medium" if value < 0.7 else "High (Creative)"
                print(f"     🎨 {key}: {value} ({creativity})")
            else:
                print(f"     ⚙️  {key}: {value}")
    
    def process_training_data(self, dataset_name: str, raw_data: List[str]) -> Set[str]:
        """
        Clean and deduplicate training data using sets.
        Returns cleaned data and stores performance metrics.
        """
        start_time = time.time()
        
        # Convert to set for automatic deduplication
        clean_data = set(data.strip().lower() for data in raw_data if data.strip())
        
        # Store cleaned data
        self.processed_data[dataset_name] = clean_data
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        duplicate_count = len(raw_data) - len(clean_data)
        duplicate_percentage = (duplicate_count / len(raw_data)) * 100 if raw_data else 0
        
        # Store metrics for analysis
        self.performance_stats[dataset_name] = {
            "original_size": len(raw_data),
            "cleaned_size": len(clean_data),
            "duplicates_removed": duplicate_count,
            "duplicate_percentage": round(duplicate_percentage, 2),
            "processing_time": round(processing_time, 4)
        }
        
        self._display_processing_results(dataset_name)
        return clean_data
    
    def _display_processing_results(self, dataset_name: str) -> None:
        """Display data processing results"""
        stats = self.performance_stats[dataset_name]
        print(f"\n📊 Dataset '{dataset_name}' Processing Results:")
        print(f"   📥 Raw entries: {stats['original_size']}")
        print(f"   ✨ Clean entries: {stats['cleaned_size']}")
        print(f"   🗑️  Duplicates removed: {stats['duplicates_removed']} ({stats['duplicate_percentage']}%)")
        print(f"   ⏱️  Processing time: {stats['processing_time']}s")
    
    def get_model_config(self, model_name: str, key: str = None) -> Union[Dict, any]:
        """
        Get model configuration or specific setting.
        Handles missing configurations gracefully.
        """
        if model_name not in self.model_configs:
            print(f"⚠️  Model '{model_name}' not configured. Using defaults.")
            return {"temperature": 0.5, "max_tokens": 1000}
        
        config = self.model_configs[model_name]
        return config.get(key) if key else config
    
    def find_data_overlap(self, dataset1: str, dataset2: str) -> Set[str]:
        """
        Find common elements between datasets.
        Useful for AI model validation and data analysis.
        """
        if dataset1 not in self.processed_data:
            print(f"❌ Dataset '{dataset1}' not found")
            return set()
        
        if dataset2 not in self.processed_data:
            print(f"❌ Dataset '{dataset2}' not found")
            return set()
        
        overlap = self.processed_data[dataset1] & self.processed_data[dataset2]
        overlap_percentage = (len(overlap) / max(len(self.processed_data[dataset1]), 
                                               len(self.processed_data[dataset2]))) * 100
        
        print(f"\n🔍 Data Overlap Analysis:")
        print(f"   📊 Common elements: {len(overlap)}")
        print(f"   📈 Overlap percentage: {overlap_percentage:.1f}%")
        print(f"   🏷️  Common items: {sorted(list(overlap))[:5]}{'...' if len(overlap) > 5 else ''}")
        
        return overlap
    
    def generate_ai_summary(self) -> Dict:
        """Generate a summary report for AI system monitoring"""
        total_datasets = len(self.processed_data)
        total_models = len(self.model_configs)
        total_data_points = sum(len(data) for data in self.processed_data.values())
        
        return {
            "system_status": "operational",
            "models_configured": total_models,
            "datasets_processed": total_datasets,
            "total_clean_data_points": total_data_points,
            "average_model_temperature": sum(
                config.get("temperature", 0.5) for config in self.model_configs.values()
            ) / max(total_models, 1),
            "performance_stats": dict(self.performance_stats)
        }


def demonstrate_ai_workflow():
    """
    Demonstrate a complete AI data workflow using dictionaries and sets.
    This simulates real-world AI system operations.
    """
    print("🤖 AI Data Manager - Day 5 Demonstration")
    print("=" * 50)
    
    # Initialize our AI system
    ai_system = AIDataManager()
    
    # Step 1: Configure different AI models for different purposes
    print("\n🔧 Step 1: Configuring AI Models")
    print("-" * 30)
    
    try:
        # Customer service chatbot - factual and safe
        ai_system.add_model_config("customer_support", {
            "temperature": 0.2,
            "max_tokens": 500,
            "safety_level": "high",
            "response_style": "professional"
        })
        
        # Creative content generator - more flexible
        ai_system.add_model_config("content_creator", {
            "temperature": 0.8,
            "max_tokens": 2000,
            "safety_level": "medium",
            "response_style": "creative"
        })
        
        # Code assistant - balanced
        ai_system.add_model_config("code_helper", {
            "temperature": 0.4,
            "max_tokens": 1500,
            "safety_level": "medium",
            "response_style": "technical"
        })
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
    
    # Step 2: Process real-world training data with duplicates
    print("\n📊 Step 2: Processing Training Data")
    print("-" * 30)
    
    # Simulate real customer support data (with realistic duplicates)
    customer_queries = [
        "how to reset password", "billing question", "technical support",
        "How to reset password", "account locked", "BILLING QUESTION",
        "technical support", "feature request", "password reset help",
        "billing inquiry", "tech support needed", "account issues"
    ]
    
    # Simulate product feedback data
    product_reviews = [
        "excellent product", "poor quality", "excellent product",
        "good value", "Poor Quality", "amazing features", "good value",
        "EXCELLENT PRODUCT", "needs improvement", "great design",
        "poor quality", "value for money"
    ]
    
    # Process the datasets
    clean_queries = ai_system.process_training_data("customer_support", customer_queries)
    clean_reviews = ai_system.process_training_data("product_feedback", product_reviews)
    
    # Step 3: Analyze data relationships
    print("\n🔍 Step 3: Data Analysis")
    print("-" * 30)
    
    overlap = ai_system.find_data_overlap("customer_support", "product_feedback")
    
    # Step 4: Use configurations in practice
    print("\n⚙️ Step 4: Using Model Configurations")
    print("-" * 30)
    
    # Get specific model settings
    support_temp = ai_system.get_model_config("customer_support", "temperature")
    creative_temp = ai_system.get_model_config("content_creator", "temperature")
    
    print(f"🎯 Customer Support Model Creativity: {support_temp}")
    print(f"🎨 Content Creator Model Creativity: {creative_temp}")
    print(f"💡 Creativity Difference: {creative_temp - support_temp:.1f}")
    
    # Step 5: Generate system summary
    print("\n📈 Step 5: System Summary")
    print("-" * 30)
    
    summary = ai_system.generate_ai_summary()
    print(json.dumps(summary, indent=2))
    
    return ai_system


def interactive_dictionary_demo():
    """Interactive demonstration of dictionary operations for AI"""
    print("\n🎯 Interactive Dictionary Demo")
    print("=" * 40)
    
    # Real AI API response simulation
    ai_response = {
        "model": "gemini-pro",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Python is great for AI development!",
                    "confidence": 0.95
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        },
        "metadata": {
            "processing_time": 1.2,
            "model_version": "1.0.0"
        }
    }
    
    print("📡 Simulated AI API Response:")
    print(json.dumps(ai_response, indent=2))
    
    # Extract key information (common AI workflow)
    print("\n🔍 Extracting Key Information:")
    content = ai_response["choices"][0]["message"]["content"]
    confidence = ai_response["choices"][0]["message"]["confidence"]
    total_tokens = ai_response["usage"]["total_tokens"]
    
    print(f"💬 AI Response: {content}")
    print(f"🎯 Confidence: {confidence * 100}%")
    print(f"🔢 Tokens Used: {total_tokens}")
    
    return ai_response


def interactive_sets_demo():
    """Interactive demonstration of set operations for AI"""
    print("\n🎯 Interactive Sets Demo")
    print("=" * 40)
    
    # Simulate AI training data categories
    ml_skills = {"python", "statistics", "linear_algebra", "data_analysis", "pandas"}
    ai_skills = {"python", "neural_networks", "deep_learning", "tensorflow", "data_analysis"}
    data_skills = {"python", "sql", "statistics", "visualization", "pandas", "data_analysis"}
    
    print("👨‍💻 Skill Sets in AI:")
    print(f"🤖 ML Skills: {ml_skills}")
    print(f"🧠 AI Skills: {ai_skills}")
    print(f"📊 Data Skills: {data_skills}")
    
    # Find intersections (core skills needed)
    core_skills = ml_skills & ai_skills & data_skills
    print(f"\n⭐ Core Skills (needed for all): {core_skills}")
    
    # Find unique skills per area
    ml_unique = ml_skills - ai_skills - data_skills
    ai_unique = ai_skills - ml_skills - data_skills
    data_unique = data_skills - ml_skills - ai_skills
    
    print(f"\n🔍 Unique Skills:")
    print(f"   ML Only: {ml_unique}")
    print(f"   AI Only: {ai_unique}")
    print(f"   Data Only: {data_unique}")
    
    # Union (all skills combined)
    all_skills = ml_skills | ai_skills | data_skills
    print(f"\n🌟 All Skills Combined: {len(all_skills)} skills")
    print(f"   Complete Skill Set: {sorted(all_skills)}")
    
    return {
        "core_skills": core_skills,
        "all_skills": all_skills,
        "total_unique_skills": len(all_skills)
    }


def main():
    """Main function to run the complete Day 5 lesson"""
    print("🚀 Welcome to Day 5: Data Structures for AI")
    print("📚 Learning: Dictionaries and Sets for AI Applications")
    print("=" * 60)
    
    # Run demonstrations
    try:
        # Main AI workflow demonstration
        ai_system = demonstrate_ai_workflow()
        
        # Interactive demos
        interactive_dictionary_demo()
        sets_analysis = interactive_sets_demo()
        
        # Final summary
        print("\n🎉 Day 5 Complete!")
        print("=" * 30)
        print("✅ You've learned:")
        print("   • Dictionary operations for AI configurations")
        print("   • Set operations for data deduplication")
        print("   • Real-world AI data processing workflows")
        print("   • Performance optimization with proper data structures")
        
        print(f"\n📊 Today's Processing Summary:")
        summary = ai_system.generate_ai_summary()
        print(f"   🤖 Models Configured: {summary['models_configured']}")
        print(f"   📦 Datasets Processed: {summary['datasets_processed']}")
        print(f"   🔢 Clean Data Points: {summary['total_clean_data_points']}")
        
        print("\n🚀 Ready for Day 6: Functions, Modules, and Libraries!")
        
    except Exception as e:
        print(f"❌ Error during lesson: {e}")
        print("💡 Check your Python environment and try again.")


if __name__ == "__main__":
    main()
