"""
Real-time Probability Dashboard
Live visualization of probability simulations and spam classifier
"""

import json
import time
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from lesson_code import ProbabilityBasics, SpamClassifier, ProbabilityDistribution
import random
from collections import Counter
import plotly.graph_objs as go
import plotly.utils

app = Flask(__name__)
app.config['SECRET_KEY'] = 'probability_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for simulations
simulation_state = {
    'coin_flips': {'heads': 0, 'tails': 0, 'total': 0, 'running': False},
    'die_rolls': {'counts': {i: 0 for i in range(1, 7)}, 'total': 0, 'running': False},
    'spam_classifier': {'emails_classified': [], 'running': False}
}

# Initialize components
prob_basics = ProbabilityBasics()
spam_classifier = SpamClassifier()
prob_dist = ProbabilityDistribution()

# Training data for spam classifier
spam_emails = [
    "win free money now click here",
    "free prize winner claim now", 
    "congratulations you won money",
    "claim your free prize today",
    "winner winner free money"
]

ham_emails = [
    "meeting scheduled for tomorrow at three",
    "project deadline is next week",
    "lunch plans for today", 
    "can we schedule a call",
    "review the project proposal"
]

# Train the classifier
spam_classifier.train(spam_emails, ham_emails)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to Probability Dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_coin_simulation')
def start_coin_simulation():
    """Start coin flip simulation"""
    if simulation_state['coin_flips']['running']:
        return
    
    simulation_state['coin_flips']['running'] = True
    simulation_state['coin_flips']['heads'] = 0
    simulation_state['coin_flips']['tails'] = 0
    simulation_state['coin_flips']['total'] = 0
    
    def run_coin_simulation():
        while simulation_state['coin_flips']['running']:
            # Simulate a batch of flips
            batch_size = random.randint(10, 50)
            for _ in range(batch_size):
                if random.random() < 0.5:
                    simulation_state['coin_flips']['heads'] += 1
                else:
                    simulation_state['coin_flips']['tails'] += 1
                simulation_state['coin_flips']['total'] += 1
            
            # Send update
            heads_prob = simulation_state['coin_flips']['heads'] / simulation_state['coin_flips']['total']
            tails_prob = simulation_state['coin_flips']['tails'] / simulation_state['coin_flips']['total']
            
            socketio.emit('coin_update', {
                'heads': simulation_state['coin_flips']['heads'],
                'tails': simulation_state['coin_flips']['tails'],
                'total': simulation_state['coin_flips']['total'],
                'heads_probability': heads_prob,
                'tails_probability': tails_prob,
                'theoretical_probability': 0.5
            })
            
            time.sleep(0.5)  # Update every 500ms
    
    thread = threading.Thread(target=run_coin_simulation)
    thread.daemon = True
    thread.start()

@socketio.on('stop_coin_simulation')
def stop_coin_simulation():
    """Stop coin flip simulation"""
    simulation_state['coin_flips']['running'] = False

@socketio.on('start_die_simulation')
def start_die_simulation():
    """Start die roll simulation"""
    if simulation_state['die_rolls']['running']:
        return
    
    simulation_state['die_rolls']['running'] = True
    simulation_state['die_rolls']['counts'] = {i: 0 for i in range(1, 7)}
    simulation_state['die_rolls']['total'] = 0
    
    def run_die_simulation():
        while simulation_state['die_rolls']['running']:
            # Simulate a batch of rolls
            batch_size = random.randint(20, 100)
            for _ in range(batch_size):
                roll = random.randint(1, 6)
                simulation_state['die_rolls']['counts'][roll] += 1
                simulation_state['die_rolls']['total'] += 1
            
            # Calculate probabilities
            probabilities = {}
            for i in range(1, 7):
                probabilities[i] = simulation_state['die_rolls']['counts'][i] / simulation_state['die_rolls']['total']
            
            socketio.emit('die_update', {
                'counts': simulation_state['die_rolls']['counts'],
                'total': simulation_state['die_rolls']['total'],
                'probabilities': probabilities,
                'theoretical_probability': 1/6
            })
            
            time.sleep(0.8)  # Update every 800ms
    
    thread = threading.Thread(target=run_die_simulation)
    thread.daemon = True
    thread.start()

@socketio.on('stop_die_simulation')
def stop_die_simulation():
    """Stop die roll simulation"""
    simulation_state['die_rolls']['running'] = False

@socketio.on('classify_email')
def classify_email(data):
    """Classify an email and return results"""
    email_text = data.get('email', '')
    if not email_text:
        return
    
    result = spam_classifier.classify(email_text)
    
    # Add to history
    simulation_state['spam_classifier']['emails_classified'].append({
        'text': email_text,
        'classification': result['classification'],
        'spam_probability': result['spam_probability'],
        'ham_probability': result['ham_probability'],
        'confidence': result['confidence'],
        'timestamp': time.time()
    })
    
    # Keep only last 20 emails
    if len(simulation_state['spam_classifier']['emails_classified']) > 20:
        simulation_state['spam_classifier']['emails_classified'] = simulation_state['spam_classifier']['emails_classified'][-20:]
    
    emit('classification_result', result)

@socketio.on('get_classification_history')
def get_classification_history():
    """Send classification history to client"""
    emit('classification_history', simulation_state['spam_classifier']['emails_classified'])

@socketio.on('get_current_state')
def get_current_state():
    """Send current simulation state to client"""
    emit('current_state', simulation_state)

if __name__ == '__main__':
    print("🎲 Starting Probability Dashboard...")
    print("📊 Open your browser to: http://localhost:5000")
    print("🔄 Real-time simulations will update automatically")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
