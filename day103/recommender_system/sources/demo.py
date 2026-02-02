"""
Interactive Demo for Day 103: Recommender Systems Theory
"""

import sys
import os
import requests
import time

# Add parent directory to path to import lesson_code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lesson_code import (
    CollaborativeFilteringEngine,
    ContentBasedEngine,
    HybridRecommender,
    create_sample_dataset
)

def run_demo():
    """Run interactive demo"""
    print("\n" + "="*70)
    print("🎮 INTERACTIVE RECOMMENDER SYSTEMS DEMO")
    print("="*70)
    
    print("\n1️⃣  Generating sample dataset...")
    time.sleep(1)
    interactions, item_features = create_sample_dataset()
    print(f"   Created {len(interactions)} ratings from {interactions['user_id'].nunique()} users")
    print(f"   Item catalog: {interactions['item_id'].nunique()} items")
    
    test_user = 5
    
    print("\n2️⃣  Running Collaborative Filtering Demo...")
    time.sleep(1)
    cf_engine = CollaborativeFilteringEngine()
    cf_engine.fit(interactions)
    cf_results = cf_engine.recommend(test_user, top_n=5)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(cf_results.recommended_items, cf_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    print("\n3️⃣  Running Content-Based Filtering Demo...")
    time.sleep(1)
    cb_engine = ContentBasedEngine()
    cb_engine.fit(item_features, interactions)
    user_rated_items = set(interactions[interactions['user_id'] == test_user]['item_id'])
    cb_results = cb_engine.recommend(test_user, top_n=5, exclude_items=user_rated_items)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(cb_results.recommended_items, cb_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    print("\n4️⃣  Running Hybrid Recommender Demo...")
    time.sleep(1)
    hybrid = HybridRecommender(collaborative_weight=0.6, content_weight=0.4)
    hybrid.fit(interactions, item_features)
    hybrid_results = hybrid.recommend(test_user, top_n=5)
    print(f"   Recommendations for User {test_user}:")
    for item, score in zip(hybrid_results.recommended_items, hybrid_results.scores):
        print(f"      Item {item}: {score:.3f}")
    
    # Try to update dashboard if running
    try:
        response = requests.post('http://localhost:5000/api/update-demo', timeout=1)
        if response.status_code == 200:
            print("\n✅ Demo metrics updated on dashboard")
    except:
        print("\nℹ️  Dashboard not running (this is okay)")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\n📊 Key Results:")
    print(f"   - Collaborative Filtering: {len(cf_results.recommended_items)} recommendations")
    print(f"   - Content-Based Filtering: {len(cb_results.recommended_items)} recommendations")
    print(f"   - Hybrid Recommender: {len(hybrid_results.recommended_items)} recommendations")
    print("\n🚀 Next: Check the dashboard at http://localhost:5000")
    
    return {
        'cf': cf_results,
        'cb': cb_results,
        'hybrid': hybrid_results
    }

if __name__ == "__main__":
    run_demo()
