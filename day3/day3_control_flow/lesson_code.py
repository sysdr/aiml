"""
Day 3: Control Flow for AI Systems
Teaching AI systems to make decisions through if-else statements and loops
"""

import random
from datetime import datetime
from typing import List, Dict, Tuple, Any

def display_welcome():
    """Display welcome message for Day 3"""
    print("🤖 Day 3: Control Flow for AI Systems")
    print("=====================================")
    print("Today we're building the decision-making logic of AI systems!")
    print()

class DataValidator:
    """AI Data Validation System using Control Flow"""
    
    def __init__(self):
        self.validation_rules = {
            'min_length': 3,
            'max_length': 1000,
            'allowed_types': (str, int, float)
        }
        self.stats = {
            'total_checked': 0,
            'valid_count': 0,
            'invalid_count': 0
        }
    
    def validate_single_item(self, data_point: Any) -> Tuple[bool, str]:
        """
        Validate a single data point for AI training
        This mimics real AI data preprocessing
        """
        # Rule 1: Check for None values
        if data_point is None:
            return False, "Missing data (None value)"
        
        # Rule 2: Check data type
        if not isinstance(data_point, self.validation_rules['allowed_types']):
            return False, f"Invalid type: {type(data_point).__name__}"
        
        # Rule 3: Check length constraints
        data_str = str(data_point)
        if len(data_str) < self.validation_rules['min_length']:
            return False, f"Too short: {len(data_str)} chars (min: {self.validation_rules['min_length']})"
        
        if len(data_str) > self.validation_rules['max_length']:
            return False, f"Too long: {len(data_str)} chars (max: {self.validation_rules['max_length']})"
        
        # All checks passed
        return True, "Valid data"
    
    def process_dataset(self, dataset: List[Any]) -> Dict[str, Any]:
        """
        Process entire dataset using loops - core AI workflow
        """
        print(f"🔄 Processing dataset of {len(dataset)} items...")
        
        valid_data = []
        invalid_data = []
        processing_log = []
        
        # Main processing loop - this is how AI systems handle big data
        for index, item in enumerate(dataset):
            self.stats['total_checked'] += 1
            
            is_valid, message = self.validate_single_item(item)
            
            if is_valid:
                # Data preprocessing for AI
                processed_item = self._preprocess_for_ai(item)
                valid_data.append(processed_item)
                self.stats['valid_count'] += 1
                processing_log.append(f"✅ Item {index}: {message}")
            else:
                invalid_data.append({'item': item, 'reason': message})
                self.stats['invalid_count'] += 1
                processing_log.append(f"❌ Item {index}: {message}")
        
        return {
            'valid_data': valid_data,
            'invalid_data': invalid_data,
            'processing_log': processing_log,
            'stats': self.stats.copy()
        }
    
    def _preprocess_for_ai(self, item: Any) -> str:
        """Simple preprocessing steps common in AI pipelines"""
        # Convert to string and normalize
        processed = str(item).strip().lower()
        
        # Remove extra whitespace (common AI preprocessing step)
        processed = ' '.join(processed.split())
        
        return processed

class SimpleSentimentAI:
    """Simple AI sentiment analyzer using control flow"""
    
    def __init__(self):
        self.sentiment_keywords = {
            'positive': ['great', 'amazing', 'love', 'excellent', 'fantastic', 'awesome', 'perfect', 'wonderful'],
            'negative': ['terrible', 'hate', 'awful', 'bad', 'worst', 'horrible', 'disappointing', 'poor']
        }
        self.analysis_count = 0
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using if-else decision logic
        This demonstrates how AI classification works
        """
        if not text or not isinstance(text, str):
            return {
                'sentiment': 'unknown',
                'confidence': 0.0,
                'reasoning': 'Invalid or empty text'
            }
        
        text_lower = text.lower()
        positive_score = 0
        negative_score = 0
        matching_words = []
        
        # Count positive keywords using loops
        for word in self.sentiment_keywords['positive']:
            if word in text_lower:
                positive_score += 1
                matching_words.append(f"+{word}")
        
        # Count negative keywords using loops
        for word in self.sentiment_keywords['negative']:
            if word in text_lower:
                negative_score += 1
                matching_words.append(f"-{word}")
        
        # Decision making logic - the heart of AI classification
        total_score = positive_score + negative_score
        
        if total_score == 0:
            sentiment = 'neutral'
            confidence = 0.5
            reasoning = 'No sentiment keywords found'
        elif positive_score > negative_score:
            sentiment = 'positive'
            confidence = min(0.95, 0.6 + (positive_score - negative_score) * 0.1)
            reasoning = f'Positive keywords: {positive_score}, Negative: {negative_score}'
        elif negative_score > positive_score:
            sentiment = 'negative'
            confidence = min(0.95, 0.6 + (negative_score - positive_score) * 0.1)
            reasoning = f'Negative keywords: {negative_score}, Positive: {positive_score}'
        else:
            sentiment = 'mixed'
            confidence = 0.5
            reasoning = f'Equal positive and negative signals: {positive_score} each'
        
        self.analysis_count += 1
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'reasoning': reasoning,
            'matching_words': matching_words,
            'scores': {'positive': positive_score, 'negative': negative_score}
        }
    
    def batch_analyze(self, texts: List[str]) -> Dict[str, List]:
        """Analyze multiple texts using loops - scalable AI processing"""
        results = {
            'positive': [],
            'negative': [],
            'neutral': [],
            'mixed': [],
            'unknown': []
        }
        
        print(f"🧠 Analyzing {len(texts)} texts for sentiment...")
        
        for i, text in enumerate(texts):
            analysis = self.analyze_sentiment(text)
            sentiment = analysis['sentiment']
            
            result_entry = {
                'index': i,
                'text': text,
                'analysis': analysis,
                'processed_at': datetime.now().strftime('%H:%M:%S')
            }
            
            results[sentiment].append(result_entry)
        
        return results

def demonstrate_ai_training_loop():
    """Simulate AI model training using while loops"""
    print("🎯 Simulating AI Model Training Loop")
    print("===================================")
    
    accuracy = 0.20  # Starting accuracy
    epoch = 0
    target_accuracy = 0.85
    max_epochs = 50
    
    print(f"Target accuracy: {target_accuracy:.2%}")
    print(f"Starting accuracy: {accuracy:.2%}")
    print()
    
    # Training loop - this is how real AI models learn
    while accuracy < target_accuracy and epoch < max_epochs:
        epoch += 1
        
        # Simulate learning (in real AI, this would be gradient descent)
        improvement = random.uniform(0.01, 0.05)
        accuracy += improvement
        
        # Add some randomness (real training has ups and downs)
        if random.random() < 0.2:  # 20% chance of temporary setback
            accuracy -= random.uniform(0.005, 0.015)
        
        # Keep accuracy realistic
        accuracy = min(accuracy, 0.95)
        
        print(f"Epoch {epoch:2d}: Accuracy = {accuracy:.2%} (+{improvement:.2%})")
        
        # Simulate learning rate adjustment (common AI technique)
        if epoch % 10 == 0:
            print("    🔧 Adjusting learning rate...")
        
        # Early stopping condition (prevents overfitting)
        if accuracy >= target_accuracy:
            print(f"    🎉 Target reached in {epoch} epochs!")
            break
    
    if accuracy < target_accuracy:
        print(f"    ⏰ Training stopped at max epochs ({max_epochs})")
    
    return epoch, accuracy

def main():
    """Main demonstration of AI control flow concepts"""
    display_welcome()
    
    print("Demo 1: AI Data Validation System")
    print("=================================")
    
    # Create sample dataset with various data quality issues
    sample_dataset = [
        "This is a great product review",  # Valid
        "Bad",  # Too short
        None,  # Missing data
        12345,  # Valid number
        "",  # Empty string
        "Excellent customer service experience",  # Valid
        ["not", "a", "string"],  # Wrong type
        "Terrible experience with poor quality",  # Valid
        42,  # Valid number
        "OK"  # Too short
    ]
    
    validator = DataValidator()
    results = validator.process_dataset(sample_dataset)
    
    print(f"\n📊 Validation Results:")
    print(f"Valid items: {len(results['valid_data'])}")
    print(f"Invalid items: {len(results['invalid_data'])}")
    print(f"Success rate: {(len(results['valid_data'])/len(sample_dataset)*100):.1f}%")
    
    print("\n" + "="*50)
    print("Demo 2: AI Sentiment Analysis")
    print("=================================")
    
    sentiment_ai = SimpleSentimentAI()
    
    sample_reviews = [
        "This product is absolutely amazing! I love it!",
        "Terrible quality, worst purchase ever",
        "It's okay, nothing special about it",
        "Fantastic customer service and great value",
        "Bad experience with poor support",
        "The product works as expected"
    ]
    
    sentiment_results = sentiment_ai.batch_analyze(sample_reviews)
    
    print(f"\n📊 Sentiment Analysis Results:")
    for sentiment, items in sentiment_results.items():
        if items:
            print(f"{sentiment.title()}: {len(items)} items")
            for item in items[:2]:  # Show first 2 examples
                print(f"  • \"{item['text'][:40]}...\" (confidence: {item['analysis']['confidence']:.2%})")
    
    print("\n" + "="*50)
    print("Demo 3: AI Training Simulation")
    print("=================================")
    
    final_epoch, final_accuracy = demonstrate_ai_training_loop()
    
    print(f"\n🏆 Training Complete!")
    print(f"Final accuracy: {final_accuracy:.2%} after {final_epoch} epochs")
    
    print("\n" + "="*50)
    print("🎓 Key Takeaways")
    print("===============")
    print("✅ If-else statements enable AI decision making")
    print("✅ Loops process large datasets efficiently") 
    print("✅ While loops control AI training iterations")
    print("✅ Control flow is the foundation of all AI systems")
    print("\nTomorrow: Data Structures (Lists & Tuples) for organizing AI data!")

if __name__ == "__main__":
    main()
