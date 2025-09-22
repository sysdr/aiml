"""
Day 6: Functions, Modules, and Libraries for AI
A comprehensive text analyzer demonstrating AI development patterns
"""

# Libraries - Standing on giants' shoulders
import json
import string
from collections import Counter
import random
from datetime import datetime


# Functions - AI Workhorses
def clean_text(text):
    """
    Clean text for AI processing - removes noise, standardizes format
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned text ready for AI processing
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase for consistency
    cleaned = text.lower().strip()
    
    # Remove extra whitespace (common in real-world data)
    cleaned = ' '.join(cleaned.split())
    
    # Remove basic punctuation that adds noise
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    
    # Remove any remaining extra whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned


def extract_word_features(text):
    """
    Extract basic features that AI models can use
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of text features
    """
    if not text:
        return {
            'word_count': 0,
            'avg_word_length': 0,
            'unique_words': 0,
            'char_count': 0
        }
    
    words = text.split()
    return {
        'word_count': len(words),
        'avg_word_length': round(sum(len(word) for word in words) / len(words), 2) if words else 0,
        'unique_words': len(set(words)),
        'char_count': len(text)
    }


def analyze_text_sentiment(text):
    """
    Basic sentiment analysis using word counting approach
    This demonstrates how AI starts with simple rules before machine learning
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Sentiment analysis results
    """
    # Simple word lists (in production, these would be much larger)
    positive_words = ['good', 'great', 'awesome', 'excellent', 'happy', 'love', 
                     'wonderful', 'amazing', 'fantastic', 'perfect', 'best']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry',
                     'horrible', 'worst', 'disgusting', 'annoying', 'stupid']
    
    # Clean and count words
    words = clean_text(text).split()
    word_counts = Counter(words)
    
    # Calculate sentiment scores
    positive_score = sum(word_counts[word] for word in positive_words if word in word_counts)
    negative_score = sum(word_counts[word] for word in negative_words if word in word_counts)
    
    # Determine overall sentiment
    if positive_score > negative_score:
        sentiment = 'positive'
        confidence = round(positive_score / (positive_score + negative_score + 1), 2)
    elif negative_score > positive_score:
        sentiment = 'negative'
        confidence = round(negative_score / (positive_score + negative_score + 1), 2)
    else:
        sentiment = 'neutral'
        confidence = 0.5
    
    return {
        'positive_score': positive_score,
        'negative_score': negative_score,
        'sentiment': sentiment,
        'confidence': confidence
    }


def calculate_ai_readiness(features):
    """
    Determine how suitable this text is for AI processing
    
    Args:
        features (dict): Text features from extract_word_features
        
    Returns:
        dict: AI readiness assessment
    """
    score = 0
    reasons = []
    
    # Check word count
    if features['word_count'] >= 10:
        score += 40
        reasons.append("Good word count for analysis")
    elif features['word_count'] >= 5:
        score += 20
        reasons.append("Adequate word count")
    else:
        reasons.append("Too few words for reliable analysis")
    
    # Check word diversity
    if features['unique_words'] > features['word_count'] * 0.7:
        score += 30
        reasons.append("High word diversity")
    elif features['unique_words'] > features['word_count'] * 0.5:
        score += 15
        reasons.append("Moderate word diversity")
    else:
        reasons.append("Low word diversity (many repeated words)")
    
    # Check average word length
    if features['avg_word_length'] > 4:
        score += 30
        reasons.append("Good vocabulary complexity")
    elif features['avg_word_length'] > 3:
        score += 15
        reasons.append("Adequate vocabulary complexity")
    else:
        reasons.append("Simple vocabulary")
    
    return {
        'score': min(score, 100),
        'grade': 'A' if score >= 80 else 'B' if score >= 60 else 'C' if score >= 40 else 'D',
        'reasons': reasons
    }


def generate_analysis_report(text):
    """
    Combine all analysis into a comprehensive report
    This demonstrates how AI systems combine multiple analyses
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Comprehensive analysis report
    """
    # Process text through our AI pipeline
    cleaned_text = clean_text(text)
    features = extract_word_features(cleaned_text)
    sentiment = analyze_text_sentiment(text)
    ai_readiness = calculate_ai_readiness(features)
    
    # Combine results
    report = {
        'timestamp': datetime.now().isoformat(),
        'original_text': text,
        'cleaned_text': cleaned_text,
        'features': features,
        'sentiment': sentiment,
        'ai_readiness': ai_readiness,
        'summary': f"Text contains {features['word_count']} words with {sentiment['sentiment']} sentiment"
    }
    
    return report


def process_user_input():
    """Get and validate text from user"""
    print("\n" + "="*50)
    print("Enter text to analyze (or 'quit' to exit):")
    text = input("> ").strip()
    
    if text.lower() in ['quit', 'exit', 'q']:
        return None
    
    if not text:
        print("❌ Please enter some text!")
        return process_user_input()
    
    return text


def display_report(report):
    """Display analysis report in a user-friendly format"""
    print("\n" + "🤖 AI TEXT ANALYSIS REPORT" + "\n" + "="*50)
    
    # Basic info
    print(f"📝 Original Text: {report['original_text']}")
    print(f"🧹 Cleaned Text: {report['cleaned_text']}")
    print(f"📊 Summary: {report['summary']}")
    
    # Features
    features = report['features']
    print(f"\n📈 TEXT FEATURES:")
    print(f"   • Words: {features['word_count']}")
    print(f"   • Unique words: {features['unique_words']}")
    print(f"   • Average word length: {features['avg_word_length']}")
    print(f"   • Characters: {features['char_count']}")
    
    # Sentiment
    sentiment = report['sentiment']
    print(f"\n😊 SENTIMENT ANALYSIS:")
    print(f"   • Overall sentiment: {sentiment['sentiment'].upper()}")
    print(f"   • Confidence: {sentiment['confidence']}")
    print(f"   • Positive words found: {sentiment['positive_score']}")
    print(f"   • Negative words found: {sentiment['negative_score']}")
    
    # AI Readiness
    ai_readiness = report['ai_readiness']
    print(f"\n🚀 AI READINESS ASSESSMENT:")
    print(f"   • Score: {ai_readiness['score']}/100 (Grade: {ai_readiness['grade']})")
    print(f"   • Analysis:")
    for reason in ai_readiness['reasons']:
        print(f"     - {reason}")


def demo_mode():
    """Run demonstration with sample texts"""
    sample_texts = [
        "I love this amazing course! It's helping me learn AI in such a clear way.",
        "This is confusing and I hate how complicated everything seems.",
        "The weather is nice today. It is sunny outside.",
        "AI will revolutionize technology and create incredible opportunities for innovation."
    ]
    
    print("🎯 DEMO MODE: Analyzing sample texts...")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n📋 Sample {i}:")
        report = generate_analysis_report(text)
        display_report(report)
        
        if i < len(sample_texts):
            print("\n" + "="*50)


def main():
    """Main application entry point"""
    print("🤖 AI TEXT ANALYZER v1.0")
    print("Day 6: Functions, Modules, and Libraries")
    print("="*50)
    print("This tool demonstrates AI development patterns:")
    print("• Functions for specific tasks")
    print("• Modules for organization") 
    print("• Libraries for powerful features")
    print("="*50)
    
    # Ask if user wants demo or interactive mode
    mode = input("Choose mode: (d)emo or (i)nteractive? ").lower()
    
    if mode.startswith('d'):
        demo_mode()
    
    print("\n🎮 INTERACTIVE MODE:")
    
    while True:
        text = process_user_input()
        if text is None:
            break
        
        try:
            # Generate and display report
            report = generate_analysis_report(text)
            display_report(report)
            
            # Option to save report
            save = input("\n💾 Save this report to file? (y/n): ").lower()
            if save.startswith('y'):
                filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"✅ Report saved to {filename}")
        
        except Exception as e:
            print(f"❌ Error analyzing text: {e}")
            print("Please try again with different text.")
    
    print("\n👋 Thanks for using AI Text Analyzer!")
    print("🎯 You've learned how functions, modules, and libraries work together!")


# Module organization - this pattern scales to large AI systems
if __name__ == "__main__":
    main()
