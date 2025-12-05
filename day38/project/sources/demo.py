"""
Interactive Demo for Day 38: Machine Learning Workflow
"""

import requests
import time
import sys
import os

# Add parent directory to path to import lesson_code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lesson_code import MLWorkflowPipeline, run_complete_workflow

def run_demo():
    """Run interactive demo"""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE ML WORKFLOW DEMO")
    print("="*60)
    
    pipeline = MLWorkflowPipeline()
    
    print("\n1️⃣  Stage 1: Problem Definition...")
    time.sleep(1)
    problem = pipeline.define_problem()
    print(f"   Task: {problem['task']}")
    print(f"   Target: {problem['target_variable']}")
    print(f"   Success Threshold: {problem['success_threshold']}")
    
    print("\n2️⃣  Stage 2: Data Collection...")
    time.sleep(1)
    df = pipeline.collect_data()
    print(f"   Collected {len(df)} reviews")
    print(f"   Positive: {df['sentiment'].sum()}, Negative: {(df['sentiment']==0).sum()}")
    
    print("\n3️⃣  Stage 3: Data Preparation...")
    time.sleep(1)
    X_train, X_test, y_train, y_test = pipeline.prepare_data(df)
    print(f"   Train samples: {len(y_train)}")
    print(f"   Test samples: {len(y_test)}")
    print(f"   Features: {X_train.shape[1]}")
    
    print("\n4️⃣  Stage 4: Model Training...")
    time.sleep(1)
    model = pipeline.train_model(X_train, y_train)
    print("   Model trained successfully!")
    
    print("\n5️⃣  Stage 5: Model Evaluation...")
    time.sleep(1)
    metrics = pipeline.evaluate_model(X_test, y_test)
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    
    print("\n6️⃣  Stage 6: Deployment...")
    time.sleep(1)
    model_path = pipeline.deploy_model()
    print(f"   Model saved to: {model_path}")
    
    print("\n7️⃣  Stage 7: Monitoring & Prediction...")
    time.sleep(1)
    new_reviews = [
        "This product is incredible! Love it so much.",
        "Terrible quality. Very disappointed with this purchase.",
        "Good value for money. Works as expected."
    ]
    results = pipeline.predict(new_reviews)
    
    print("\n   Predictions:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. \"{result['review']}\"")
        print(f"      → {result['sentiment']} (confidence: {result['confidence']:.1%})")
    
    # Try to update dashboard if running
    try:
        response = requests.post('http://localhost:5000/api/update-demo', timeout=1)
        if response.status_code == 200:
            print("\n✅ Demo metrics updated on dashboard")
    except:
        print("\nℹ️  Dashboard not running (this is okay)")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)
    print("\n📊 Key Results:")
    print(f"   - Accuracy: {metrics['accuracy']:.3f}")
    print(f"   - F1 Score: {metrics['f1_score']:.3f}")
    print(f"   - Precision: {metrics['precision']:.3f}")
    print(f"   - Recall: {metrics['recall']:.3f}")
    print(f"   - Predictions made: {len(results)}")
    print("\n🚀 Next: Check the dashboard at http://localhost:5000")
    
    return {
        'metrics': metrics,
        'predictions': results
    }

if __name__ == "__main__":
    run_demo()

