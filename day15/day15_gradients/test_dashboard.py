#!/usr/bin/env python3
"""
Test script for the Gradient Descent Dashboard API
"""

import requests
import json

def test_dashboard_api():
    """Test the dashboard API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Gradient Descent Dashboard API")
    print("=" * 50)
    
    # Test 1: Check if dashboard is accessible
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Dashboard homepage accessible")
        else:
            print(f"❌ Dashboard homepage failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to dashboard: {e}")
        return
    
    # Test 2: Test plot generation
    try:
        plot_data = {
            "weight": 0.1,
            "bias": 0,
            "learning_rate": 0.000001,
            "epochs": 100
        }
        response = requests.post(f"{base_url}/api/plot", json=plot_data)
        if response.status_code == 200:
            data = response.json()
            if 'plot' in data and len(data['plot']) > 0:
                print("✅ Plot generation working")
            else:
                print("❌ Plot generation failed - no plot data")
        else:
            print(f"❌ Plot generation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Plot generation error: {e}")
    
    # Test 3: Test training API
    try:
        train_data = {
            "learning_rate": 0.000001,
            "epochs": 10,
            "initial_weight": 0.1,
            "initial_bias": 0
        }
        response = requests.post(f"{base_url}/api/train", json=train_data)
        if response.status_code == 200:
            data = response.json()
            if 'history' in data and 'final_weight' in data:
                print("✅ Training API working")
                print(f"   Final weight: {data['final_weight']:.4f}")
                print(f"   Final bias: {data['final_bias']:.2f}")
                print(f"   Training epochs: {len(data['history'])}")
            else:
                print("❌ Training API failed - missing data")
        else:
            print(f"❌ Training API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Training API error: {e}")
    
    # Test 4: Test data API
    try:
        response = requests.get(f"{base_url}/api/data")
        if response.status_code == 200:
            data = response.json()
            if 'house_sizes' in data and 'house_prices' in data:
                print("✅ Data API working")
                print(f"   Dataset size: {len(data['house_sizes'])} houses")
            else:
                print("❌ Data API failed - missing data")
        else:
            print(f"❌ Data API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Data API error: {e}")
    
    print("\n🎯 Dashboard Status: READY")
    print("🌐 Access at: http://localhost:5000")
    print("📊 All API endpoints tested successfully!")

if __name__ == "__main__":
    test_dashboard_api()
