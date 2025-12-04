"""
Interactive Demo for Day 37: Introduction to AI, ML, and Deep Learning
"""

import requests
import time
from lesson_code import AIMLIntroduction

def run_demo():
    """Run interactive demo"""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE AI/ML DEMO")
    print("="*60)
    
    intro = AIMLIntroduction()
    
    print("\n1️⃣  Running Traditional Programming vs ML comparison...")
    time.sleep(1)
    intro.traditional_programming_vs_ml()
    
    print("\n2️⃣  Explaining AI, ML, and Deep Learning relationship...")
    time.sleep(1)
    intro.ai_ml_dl_relationship()
    
    print("\n3️⃣  Running Supervised Learning Demo...")
    time.sleep(1)
    results = intro.supervised_learning_demo()
    
    print("\n4️⃣  Overview of Learning Types...")
    time.sleep(1)
    intro.learning_types_overview()
    
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
    print(f"   - Model R² Score: {results['r2']:.3f}")
    print(f"   - Model MSE: ${results['mse']:,.2f}")
    print(f"   - Predictions made: 3")
    print("\n🚀 Next: Check the dashboard at http://localhost:5000")
    
    return results

if __name__ == "__main__":
    run_demo()
