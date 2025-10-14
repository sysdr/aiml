#!/usr/bin/env python3
"""
🎯 Gradient Descent Interactive Dashboard
A comprehensive web interface for exploring gradient descent concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

class GradientDescentDashboard:
    def __init__(self):
        self.house_sizes = np.array([1200, 1500, 1800, 2000, 2200, 2500, 2800])
        self.house_prices = np.array([200000, 250000, 300000, 350000, 380000, 450000, 520000])
        
    def generate_plot(self, weight, bias, learning_rate, epochs, show_gradient=False):
        """Generate a matplotlib plot and return as base64 string"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Data and Model
        ax1.scatter(self.house_sizes, self.house_prices, color='blue', s=100, alpha=0.7, label='Training Data')
        
        # Generate predictions
        predictions = weight * self.house_sizes + bias
        ax1.plot(self.house_sizes, predictions, color='red', linewidth=3, label=f'Model: ${weight:.2f} × size + ${bias:.0f}')
        
        ax1.set_xlabel('House Size (sq ft)', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title('House Price Prediction Model', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add prediction examples
        test_sizes = [1600, 2000, 2800]
        test_predictions = [weight * size + bias for size in test_sizes]
        for size, pred in zip(test_sizes, test_predictions):
            ax1.plot([size, size], [pred, pred], 'go', markersize=8)
            ax1.annotate(f'${pred:,.0f}', (size, pred), xytext=(5, 10), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Plot 2: Loss over time (simulated)
        if epochs > 0:
            # Simulate loss reduction
            initial_loss = np.mean((self.house_prices - (0.1 * self.house_sizes + 0))**2)
            final_loss = np.mean((self.house_prices - predictions)**2)
            
            epoch_range = np.linspace(0, epochs, 50)
            loss_values = initial_loss * np.exp(-epoch_range / (epochs/3)) + final_loss
            
            ax2.plot(epoch_range, loss_values, color='green', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss (Mean Squared Error)', fontsize=12)
            ax2.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str
    
    def run_gradient_descent(self, learning_rate, epochs, initial_weight, initial_bias):
        """Run gradient descent and return training history"""
        weight = initial_weight
        bias = initial_bias
        history = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = weight * self.house_sizes + bias
            error = self.house_prices - predictions
            
            # Clip error to prevent overflow
            error = np.clip(error, -1e6, 1e6)
            loss = np.mean(error ** 2)
            
            # Calculate gradients with clipping
            weight_gradient = (2/len(self.house_sizes)) * np.sum(error * self.house_sizes)
            bias_gradient = (2/len(self.house_sizes)) * np.sum(error)
            
            # Clip gradients to prevent overflow
            weight_gradient = np.clip(weight_gradient, -1e6, 1e6)
            bias_gradient = np.clip(bias_gradient, -1e6, 1e6)
            
            # Update parameters
            weight -= learning_rate * weight_gradient
            bias -= learning_rate * bias_gradient
            
            # Clip parameters to prevent overflow
            weight = np.clip(weight, -1e6, 1e6)
            bias = np.clip(bias, -1e6, 1e6)
            
            # Store history
            history.append({
                'epoch': epoch + 1,
                'loss': float(loss) if not np.isnan(loss) and not np.isinf(loss) else 0.0,
                'weight': float(weight) if not np.isnan(weight) and not np.isinf(weight) else 0.0,
                'bias': float(bias) if not np.isnan(bias) and not np.isinf(bias) else 0.0,
                'weight_gradient': float(weight_gradient) if not np.isnan(weight_gradient) and not np.isinf(weight_gradient) else 0.0,
                'bias_gradient': float(bias_gradient) if not np.isnan(bias_gradient) and not np.isinf(bias_gradient) else 0.0
            })
            
            # Stop if loss becomes too small or NaN
            if np.isnan(loss) or loss < 1e-10:
                break
        
        # Ensure final values are valid numbers
        final_weight = float(weight) if not np.isnan(weight) and not np.isinf(weight) else 0.0
        final_bias = float(bias) if not np.isnan(bias) and not np.isinf(bias) else 0.0
        
        return history, final_weight, final_bias

# Initialize dashboard
dashboard = GradientDescentDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/plot', methods=['POST'])
def generate_plot_api():
    """API endpoint to generate plots"""
    data = request.json
    weight = float(data.get('weight', 0.1))
    bias = float(data.get('bias', 0))
    learning_rate = float(data.get('learning_rate', 0.000001))
    epochs = int(data.get('epochs', 100))
    
    plot_data = dashboard.generate_plot(weight, bias, learning_rate, epochs)
    
    return jsonify({
        'plot': plot_data,
        'weight': weight,
        'bias': bias,
        'learning_rate': learning_rate,
        'epochs': epochs
    })

@app.route('/api/train', methods=['POST'])
def train_model():
    """API endpoint to run gradient descent training"""
    data = request.json
    learning_rate = float(data.get('learning_rate', 0.000001))
    epochs = int(data.get('epochs', 100))
    initial_weight = float(data.get('initial_weight', 0.1))
    initial_bias = float(data.get('initial_bias', 0))
    
    history, final_weight, final_bias = dashboard.run_gradient_descent(
        learning_rate, epochs, initial_weight, initial_bias
    )
    
    return jsonify({
        'history': history,
        'final_weight': final_weight,
        'final_bias': final_bias,
        'final_loss': history[-1]['loss'] if history else 0.0
    })

@app.route('/api/data')
def get_data():
    """API endpoint to get training data"""
    return jsonify({
        'house_sizes': dashboard.house_sizes.tolist(),
        'house_prices': dashboard.house_prices.tolist()
    })

if __name__ == '__main__':
    print("🚀 Starting Gradient Descent Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🎯 Features:")
    print("   • Interactive parameter adjustment")
    print("   • Real-time visualization")
    print("   • Training history tracking")
    print("   • Multiple learning rate experiments")
    app.run(debug=True, host='0.0.0.0', port=5000)
