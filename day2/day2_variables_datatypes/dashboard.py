#!/usr/bin/env python3
"""
Web Dashboard for Day 2: Variables, Data Types, and Operators for AI
180-Day AI and Machine Learning Course

A simple web dashboard to display AI agent metrics and demo functionality.
"""

from flask import Flask, render_template, jsonify, request
import json
import time
from lesson_code import SimpleAIAgent, demonstrate_ai_data_types
import threading
import random

app = Flask(__name__)

# Global AI agent instance
ai_agent = SimpleAIAgent("DashboardBot")
metrics_history = []

def update_metrics():
    """Update metrics periodically"""
    global metrics_history
    
    while True:
        status = ai_agent.get_agent_status()
        metrics_history.append({
            'timestamp': time.time(),
            'total_interactions': status['total_interactions'],
            'success_rate': status['success_rate'],
            'average_confidence': status['average_confidence'],
            'is_active': status['is_active']
        })
        
        # Keep only last 50 entries
        if len(metrics_history) > 50:
            metrics_history = metrics_history[-50:]
        
        time.sleep(5)  # Update every 5 seconds

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """API endpoint to get current metrics"""
    status = ai_agent.get_agent_status()
    return jsonify({
        'agent_name': status['agent_name'],
        'version': status['version'],
        'status': status['status'],
        'total_interactions': status['total_interactions'],
        'successful_responses': status['successful_responses'],
        'success_rate': status['success_rate'],
        'average_confidence': status['average_confidence'],
        'is_active': status['is_active'],
        'has_conversation_history': status['has_conversation_history'],
        'recent_conversations': status['recent_conversations'],
        'recent_confidence_scores': status['recent_confidence_scores'],
        'min_confidence': status['min_confidence'],
        'max_confidence': status['max_confidence']
    })

@app.route('/api/metrics/history')
def get_metrics_history():
    """API endpoint to get metrics history"""
    return jsonify(metrics_history)

@app.route('/api/demo', methods=['POST'])
def run_demo():
    """API endpoint to run demo interactions"""
    data = request.get_json()
    user_input = data.get('input', '')
    
    if user_input:
        response = ai_agent.process_input(user_input)
        return jsonify({
            'response': response,
            'status': ai_agent.get_agent_status()
        })
    
    return jsonify({'error': 'No input provided'})

@app.route('/api/demo/auto', methods=['POST'])
def run_auto_demo():
    """API endpoint to run automatic demo with sample inputs"""
    sample_inputs = [
        "Hello there!",
        "What's the weather like today?",
        "Can you help me learn Python?",
        "How do neural networks work?",
        "Thanks for your help!"
    ]
    
    results = []
    for input_text in sample_inputs:
        response = ai_agent.process_input(input_text)
        results.append({
            'input': input_text,
            'response': response,
            'confidence': ai_agent.confidence_scores[-1] if ai_agent.confidence_scores else 0
        })
    
    return jsonify({
        'results': results,
        'final_status': ai_agent.get_agent_status()
    })

if __name__ == '__main__':
    # Start metrics update thread
    metrics_thread = threading.Thread(target=update_metrics, daemon=True)
    metrics_thread.start()
    
    print("🚀 Starting AI Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🤖 AI Agent initialized and ready!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
