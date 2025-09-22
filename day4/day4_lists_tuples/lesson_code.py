"""
Day 4: Lists and Tuples for AI/ML
A practical implementation demonstrating how AI systems use Python data structures.
"""

class AIDataProcessor:
    """
    Simulates how AI systems organize and process data using lists and tuples.
    This mirrors real-world AI data pipelines.
    """
    
    def __init__(self):
        # Lists: Dynamic collections that grow (like AI learning)
        self.training_data = []
        self.predictions = []
        self.confidence_scores = []
        
        # Tuples: Immutable configurations (like model hyperparameters)
        self.model_config = ("neural_network", 3, 0.001)  # (type, layers, learning_rate)
        self.input_shape = (224, 224, 3)  # (width, height, channels) - common AI image format
    
    def add_training_sample(self, features, label, metadata=None):
        """
        Add a training sample - mimics feeding data to an AI model.
        
        Args:
            features: List of numeric features (like pixel values or sensor readings)
            label: String classification label
            metadata: Tuple of additional info (timestamp, source, confidence)
        """
        # Store features as list (mutable for preprocessing)
        processed_features = self.normalize_features(features)
        
        # Create data point as tuple (immutable record)
        if metadata:
            data_point = (processed_features, label, metadata)
        else:
            import time
            timestamp = int(time.time())
            data_point = (processed_features, label, ("auto", timestamp, 1.0))
        
        self.training_data.append(data_point)
        print(f"📊 Added sample: {label} with {len(features)} features")
    
    def normalize_features(self, features):
        """Normalize features like real AI preprocessing"""
        if not features:
            return []
        
        # Simple min-max normalization
        min_val = min(features)
        max_val = max(features)
        
        if max_val == min_val:
            return [0.5] * len(features)
        
        return [(x - min_val) / (max_val - min_val) for x in features]
    
    def predict_batch(self, feature_sets):
        """
        Simulate batch prediction - how AI models process multiple inputs.
        Returns list of (prediction, confidence) tuples.
        """
        predictions = []
        
        for features in feature_sets:
            # Simulate AI prediction logic
            normalized = self.normalize_features(features)
            
            # Simple classifier: based on feature average
            avg_feature = sum(normalized) / len(normalized) if normalized else 0
            
            if avg_feature > 0.7:
                prediction = "high_class"
                confidence = min(0.95, avg_feature + 0.1)
            elif avg_feature > 0.3:
                prediction = "medium_class"
                confidence = 0.7 + (avg_feature - 0.3) * 0.5
            else:
                prediction = "low_class"
                confidence = max(0.5, avg_feature + 0.2)
            
            # Store as tuple (immutable prediction record)
            result = (prediction, round(confidence, 3))
            predictions.append(result)
        
        self.predictions.extend(predictions)
        return predictions
    
    def analyze_performance(self):
        """Analyze model performance - like AI evaluation metrics"""
        if not self.training_data:
            return {"error": "No training data available"}
        
        # Extract labels from training data
        labels = [data_point[1] for data_point in self.training_data]
        unique_labels = list(set(labels))
        
        # Count label distribution
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Calculate basic statistics
        total_samples = len(self.training_data)
        prediction_count = len(self.predictions)
        
        return {
            "total_training_samples": total_samples,
            "unique_labels": unique_labels,
            "label_distribution": label_counts,
            "predictions_made": prediction_count,
            "model_config": self.model_config,
            "input_shape": self.input_shape
        }
    
    def filter_by_confidence(self, min_confidence=0.8):
        """Filter predictions by confidence - common AI post-processing"""
        high_confidence = [pred for pred in self.predictions 
                          if pred[1] >= min_confidence]
        
        print(f"🎯 High confidence predictions: {len(high_confidence)}/{len(self.predictions)}")
        return high_confidence
    
    def get_feature_statistics(self):
        """Calculate feature statistics across dataset"""
        if not self.training_data:
            return {}
        
        # Extract all feature vectors
        all_features = []
        for data_point in self.training_data:
            features = data_point[0]  # First element of tuple
            all_features.extend(features)
        
        if not all_features:
            return {}
        
        return {
            "total_features": len(all_features),
            "min_value": min(all_features),
            "max_value": max(all_features),
            "average": sum(all_features) / len(all_features),
            "feature_dimensions": len(self.training_data[0][0]) if self.training_data else 0
        }


def demonstrate_ai_data_structures():
    """
    Interactive demonstration of lists and tuples in AI context.
    """
    print("🤖 AI Data Structures Demo - Lists and Tuples in Action!")
    print("=" * 60)
    
    # Initialize our AI data processor
    ai_processor = AIDataProcessor()
    
    # Simulate adding training data (like training a computer vision model)
    print("\n📸 Adding Computer Vision Training Data:")
    
    # Image features: [brightness, contrast, edge_count, color_variance]
    ai_processor.add_training_sample([0.8, 0.6, 45, 0.3], "cat", ("camera_1", 1234567890, 0.9))
    ai_processor.add_training_sample([0.2, 0.9, 67, 0.8], "dog", ("camera_2", 1234567891, 0.95))
    ai_processor.add_training_sample([0.7, 0.4, 23, 0.2], "cat", ("camera_1", 1234567892, 0.85))
    ai_processor.add_training_sample([0.3, 0.8, 78, 0.9], "bird", ("camera_3", 1234567893, 0.92))
    
    # Analyze our training data
    print("\n📊 Training Data Analysis:")
    stats = ai_processor.analyze_performance()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demonstrate batch prediction
    print("\n🔮 Making Predictions on New Data:")
    new_samples = [
        [0.75, 0.5, 40, 0.25],  # Cat-like features
        [0.25, 0.85, 70, 0.85], # Dog-like features
        [0.9, 0.3, 15, 0.1],    # Unknown features
    ]
    
    predictions = ai_processor.predict_batch(new_samples)
    for i, (prediction, confidence) in enumerate(predictions):
        features = new_samples[i]
        print(f"  Sample {i+1}: {features} → {prediction} (confidence: {confidence})")
    
    # Filter high-confidence predictions
    print("\n🎯 High-Confidence Filtering:")
    high_conf = ai_processor.filter_by_confidence(0.8)
    for pred, conf in high_conf:
        print(f"  {pred}: {conf}")
    
    # Feature statistics
    print("\n📈 Feature Statistics:")
    feature_stats = ai_processor.get_feature_statistics()
    for key, value in feature_stats.items():
        print(f"  {key}: {value}")
    
    # Demonstrate list and tuple operations
    print("\n🔧 Core Data Structure Operations:")
    
    # List operations (mutable)
    sample_features = [0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"  Original features: {sample_features}")
    
    # Add new feature (list is mutable)
    sample_features.append(0.6)
    print(f"  After adding feature: {sample_features}")
    
    # Modify existing feature
    sample_features[0] = 0.15
    print(f"  After modifying first: {sample_features}")
    
    # Tuple operations (immutable)
    image_dimensions = (1920, 1080, 3)  # width, height, channels
    print(f"  Image dimensions: {image_dimensions}")
    print(f"  Width: {image_dimensions[0]}, Height: {image_dimensions[1]}")
    
    # Unpacking tuple
    width, height, channels = image_dimensions
    print(f"  Unpacked: {width}x{height} with {channels} channels")
    
    # List comprehension for AI data processing
    squared_features = [x**2 for x in sample_features]
    print(f"  Squared features: {squared_features}")
    
    # Filter features above threshold
    high_features = [x for x in sample_features if x > 0.3]
    print(f"  Features > 0.3: {high_features}")
    
    print("\n✅ Demo complete! You've seen how AI systems use lists and tuples.")
    print("🎓 Key insight: Lists for changing data, tuples for fixed records!")


def interactive_exercise():
    """
    Interactive coding exercise for students.
    """
    print("\n" + "="*60)
    print("🎯 INTERACTIVE EXERCISE: Build Your Own AI Dataset")
    print("="*60)
    
    ai_system = AIDataProcessor()
    
    print("\nTask: Create a simple sentiment analysis dataset")
    print("You'll add text features and sentiment labels\n")
    
    # Sample text features: [word_count, positive_words, negative_words, sentence_length]
    sample_data = [
        ([15, 3, 0, 67], "positive", "Great product, love it!"),
        ([8, 0, 2, 34], "negative", "Terrible quality, disappointed"),
        ([12, 2, 1, 56], "neutral", "Average product, nothing special"),
        ([20, 5, 0, 89], "positive", "Amazing experience, highly recommend!"),
        ([6, 0, 3, 28], "negative", "Worst purchase ever made"),
    ]
    
    print("Adding sample sentiment data...")
    for features, label, text in sample_data:
        # Add metadata as tuple: (text_sample, character_count, has_exclamation)
        metadata = (text, len(text), "!" in text)
        ai_system.add_training_sample(features, label, metadata)
    
    # Test the system
    print("\n🧪 Testing with new text samples:")
    test_samples = [
        [18, 4, 0, 78],  # Positive-like
        [10, 1, 2, 45],  # Negative-like
        [14, 2, 1, 62],  # Neutral-like
    ]
    
    predictions = ai_system.predict_batch(test_samples)
    test_texts = [
        "Fantastic service and great value for money!",
        "Poor quality, not worth the price at all",
        "Decent product, meets basic expectations"
    ]
    
    for i, ((pred, conf), text) in enumerate(zip(predictions, test_texts)):
        print(f"  Text: '{text[:40]}...'")
        print(f"  Prediction: {pred} (confidence: {conf})\n")
    
    # Show dataset statistics
    print("📊 Final Dataset Statistics:")
    stats = ai_system.analyze_performance()
    feature_stats = ai_system.get_feature_statistics()
    
    print(f"  Total samples: {stats['total_training_samples']}")
    print(f"  Sentiment classes: {stats['unique_labels']}")
    print(f"  Class distribution: {stats['label_distribution']}")
    print(f"  Feature dimensions: {feature_stats['feature_dimensions']}")
    print(f"  Average feature value: {feature_stats['average']:.3f}")
    
    print("\n🎉 Exercise complete! You've built a mini AI sentiment analyzer!")
    return ai_system


if __name__ == "__main__":
    # Run the main demonstration
    demonstrate_ai_data_structures()
    
    # Run interactive exercise
    interactive_exercise()
    
    print("\n" + "="*60)
    print("🎓 Day 4 Complete: Lists and Tuples for AI")
    print("🚀 Tomorrow: Dictionaries and Sets for lightning-fast AI lookups!")
    print("="*60)
