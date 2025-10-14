#!/usr/bin/env python3
"""
Day 15: Gradients and Gradient Descent
Interactive implementation of gradient descent for beginners
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import time

class GradientDescentVisualizer:
    """Visual gradient descent implementation for learning"""
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize the gradient descent visualizer
        
        Args:
            learning_rate: How big steps to take (0.001 to 0.1 typically)
        """
        self.learning_rate = learning_rate
        self.weight = np.random.randn()  # Random starting point
        self.bias = np.random.randn()
        self.history = {'weights': [], 'biases': [], 'losses': []}
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using current parameters
        
        Linear model: y = weight * x + bias
        This is how AI systems make predictions!
        """
        return self.weight * X + self.bias
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate how wrong our predictions are
        Uses Mean Squared Error - standard in AI
        """
        predictions = self.predict(X)
        error = predictions - y
        return np.mean(error ** 2)
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Calculate gradients - the heart of AI learning!
        
        These partial derivatives tell us which direction 
        to adjust our parameters to reduce error
        """
        m = len(X)
        predictions = self.predict(X)
        error = predictions - y
        
        # Partial derivatives from calculus (Day 14 connection!)
        weight_gradient = (2/m) * np.sum(error * X)
        bias_gradient = (2/m) * np.sum(error)
        
        return weight_gradient, bias_gradient
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Take one step toward better predictions
        This happens millions of times in real AI training
        """
        # Calculate which direction to move
        weight_grad, bias_grad = self.compute_gradients(X, y)
        
        # Move in opposite direction (downhill toward lower error)
        self.weight -= self.learning_rate * weight_grad
        self.bias -= self.learning_rate * bias_grad
        
        # Record progress for visualization
        self.history['weights'].append(self.weight)
        self.history['biases'].append(self.bias)
        self.history['losses'].append(self.compute_loss(X, y))
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              verbose: bool = True) -> None:
        """
        Train the model using gradient descent
        
        Args:
            X: Input features (house sizes, etc.)
            y: Target values (house prices, etc.)
            epochs: Number of training iterations
            verbose: Print training progress
        """
        if verbose:
            print(f"🚀 Starting gradient descent training...")
            print(f"Initial: weight={self.weight:.3f}, bias={self.bias:.3f}")
            print(f"Initial loss: {self.compute_loss(X, y):.2f}")
            print()
        
        for epoch in range(epochs):
            self.train_step(X, y)
            
            if verbose and (epoch + 1) % 20 == 0:
                current_loss = self.history['losses'][-1]
                print(f"Epoch {epoch+1:3d}: Loss = {current_loss:8.2f}, "
                      f"Weight = {self.weight:6.3f}, Bias = {self.bias:6.3f}")
        
        if verbose:
            print(f"\n✅ Training complete!")
            print(f"Final: weight={self.weight:.3f}, bias={self.bias:.3f}")
            print(f"Final loss: {self.history['losses'][-1]:.2f}")
    
    def plot_training_progress(self) -> None:
        """Visualize how the model learned"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss over time
        ax1.plot(self.history['losses'], 'b-', linewidth=2)
        ax1.set_title('AI Learning Progress\n(Loss Decreasing = Getting Smarter)', 
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Error (Loss)')
        ax1.grid(True, alpha=0.3)
        
        # Plot parameter evolution
        ax2.plot(self.history['weights'], label='Weight', linewidth=2)
        ax2.plot(self.history['biases'], label='Bias', linewidth=2)
        ax2.set_title('Parameter Evolution\n(How AI Adjusts Its "Brain")', 
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Parameter Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, X: np.ndarray, y: np.ndarray) -> None:
        """Show how well our AI learned to predict"""
        predictions = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='red', s=100, label='Actual Data', zorder=5)
        plt.plot(X, predictions, 'b-', linewidth=3, label='AI Predictions')
        
        # Show prediction errors
        for i in range(len(X)):
            plt.plot([X[i], X[i]], [y[i], predictions[i]], 
                    'gray', linestyle='--', alpha=0.7)
        
        plt.title('AI Predictions vs Reality\n(How Close Did We Get?)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('House Size (sq ft)')
        plt.ylabel('House Price ($)')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()


def real_world_example():
    """
    Demonstrate gradient descent on a real AI problem:
    Predicting house prices (like Zillow's algorithm)
    """
    print("🏠 Real-World AI Example: House Price Prediction")
    print("=" * 55)
    
    # Realistic house data
    house_sizes = np.array([1200, 1500, 1800, 2100, 2400, 2700, 3000])
    house_prices = np.array([200000, 250000, 300000, 350000, 400000, 450000, 500000])
    
    print(f"Dataset: {len(house_sizes)} houses")
    print("Sample data:")
    for i in range(3):
        print(f"  {house_sizes[i]} sq ft → ${house_prices[i]:,}")
    print("  ...")
    print()
    
    # Train the AI model
    model = GradientDescentVisualizer(learning_rate=0.0001)
    model.train(house_sizes, house_prices, epochs=200, verbose=True)
    
    # Test predictions
    print(f"\n🔮 AI Predictions:")
    test_sizes = [1600, 2000, 2800]
    for size in test_sizes:
        price = model.predict(np.array([size]))[0]
        print(f"  {size} sq ft house: ${price:,.0f}")
    
    print(f"\n📊 Model learned: ${model.weight:.0f} per sq ft + ${model.bias:,.0f} base")
    
    # Visualize results
    model.plot_training_progress()
    model.plot_predictions(house_sizes, house_prices)


def interactive_exploration():
    """
    Let users experiment with different learning rates
    This shows why hyperparameter tuning matters in AI
    """
    print("\n🧪 Interactive Exploration: Learning Rate Effects")
    print("=" * 55)
    
    # Simple dataset for experimentation
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship
    
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    
    plt.figure(figsize=(15, 8))
    
    for i, lr in enumerate(learning_rates):
        plt.subplot(2, 2, i+1)
        
        model = GradientDescentVisualizer(learning_rate=lr)
        model.train(X, y, epochs=50, verbose=False)
        
        plt.plot(model.history['losses'], linewidth=2)
        plt.title(f'Learning Rate = {lr}', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Add interpretation
        final_loss = model.history['losses'][-1]
        if lr <= 0.01:
            plt.text(0.5, 0.8, 'Slow but Steady', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        elif lr <= 0.1:
            plt.text(0.5, 0.8, 'Good Balance', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        else:
            plt.text(0.5, 0.8, 'Too Fast!', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    plt.suptitle('Learning Rate Impact on AI Training', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\n💡 Key Insights:")
    print("   • Too low: Learns slowly but surely")
    print("   • Just right: Fast and stable learning")  
    print("   • Too high: Unstable, might not learn at all")
    print("   • This is why AI engineers spend time tuning hyperparameters!")


if __name__ == "__main__":
    print("🎓 Day 15: Gradients and Gradient Descent")
    print("Understanding How AI Systems Learn")
    print("=" * 50)
    print()
    
    # Run the main examples
    real_world_example()
    interactive_exploration()
    
    print("\n🌟 Congratulations!")
    print("You just implemented the core learning algorithm behind:")
    print("   • ChatGPT and language models")
    print("   • Computer vision systems") 
    print("   • Recommendation engines")
    print("   • And virtually every AI system!")
    print("\nNext: Review week to solidify these concepts! 🚀")
