#!/usr/bin/env python3
"""
Day 10 Matrix Operations Demo Integration
Shows how matrix operations power AI systems and integrates with the dashboard
"""

import numpy as np
import requests
import json
import time

def demo_matrix_ai_integration():
    """Demonstrate matrix operations and their connection to AI"""
    print("🎯 Day 10: Matrix Operations - AI Integration Demo")
    print("=" * 60)
    
    # 1. Create matrices representing AI data
    print("📊 Creating AI Data Matrices...")
    
    # Image data matrix (like a small 8x8 grayscale image)
    image_data = np.random.randint(0, 256, (8, 8))
    print(f"🖼️  Image Matrix (8x8): Shape {image_data.shape}")
    print(f"   Average brightness: {np.mean(image_data):.1f}")
    
    # Neural network weights
    np.random.seed(42)
    weights = np.random.randn(64, 32) * 0.1
    print(f"🧠 Neural Network Weights: Shape {weights.shape}")
    print(f"   Weight statistics: Mean={np.mean(weights):.3f}, Std={np.std(weights):.3f}")
    
    # 2. Demonstrate matrix operations
    print("\n⚡ Matrix Operations for AI:")
    
    # Image preprocessing (brightness adjustment)
    bright_image = np.clip(image_data + 50, 0, 255)
    print(f"☀️  Brightness adjustment: {np.mean(image_data):.1f} → {np.mean(bright_image):.1f}")
    
    # Matrix statistics for AI analysis
    print(f"📈 Image Analysis:")
    print(f"   Min: {np.min(image_data)}, Max: {np.max(image_data)}")
    print(f"   Standard deviation: {np.std(image_data):.2f}")
    
    # 3. Connect to AI Chat Assistant
    print("\n🤖 Connecting to AI Chat Assistant...")
    
    try:
        # Test the AI system with matrix-related questions
        questions = [
            "What are matrices used for in AI?",
            "How do neural networks use matrices?",
            "Explain matrix operations in machine learning"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n💬 Question {i}: {question}")
            
            response = requests.post(
                "http://localhost:8000/api/v1/chat",
                json={"message": question},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    answer = data["response"]
                    print(f"🤖 AI Response: {answer[:100]}...")
                    
                    # Simulate processing time
                    time.sleep(1)
                else:
                    print(f"❌ Error: {data.get('error_message', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("💡 Make sure the AI Chat Assistant is running on http://localhost:8000")
    
    # 4. Show real-world AI matrix sizes
    print("\n🌍 Real-World AI Matrix Sizes:")
    ai_systems = {
        "Small CNN": (224, 224, 3),
        "Large CNN": (512, 512, 3),
        "Transformer Embedding": (50000, 512),
        "GPT-3 Weights": (175000000000, 1),  # Simplified
        "ImageNet Dataset": (1000000, 224*224*3)
    }
    
    for system, shape in ai_systems.items():
        if len(shape) == 3:
            total_elements = shape[0] * shape[1] * shape[2]
        else:
            total_elements = shape[0] * shape[1]
        print(f"   {system}: {shape} = {total_elements:,} elements")
    
    print("\n🎉 Matrix Operations Demo Complete!")
    print("✅ Key Insights:")
    print("   • Matrices are the foundation of all AI systems")
    print("   • Simple operations like addition/scaling power complex AI")
    print("   • Real AI systems use matrices with millions/billions of elements")
    print("   • Understanding matrices helps debug and improve AI systems")
    
    return {
        "image_data": image_data.tolist(),
        "weights_shape": weights.shape,
        "brightness_change": float(np.mean(bright_image) - np.mean(image_data)),
        "ai_questions_asked": len(questions)
    }

if __name__ == "__main__":
    demo_matrix_ai_integration()
