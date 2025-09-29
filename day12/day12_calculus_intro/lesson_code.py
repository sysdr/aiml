#!/usr/bin/env python3
"""
Day 12: Introduction to Calculus for AI/ML
Complete implementation of calculus concepts for AI applications
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Callable, Tuple

class AICalculusToolkit:
    """A comprehensive toolkit for understanding calculus in AI/ML context"""
    
    def __init__(self):
        self.learning_history = []
        print("🧮 AI Calculus Toolkit initialized!")
    
    def simple_function(self, x: float) -> float:
        """Basic quadratic function: f(x) = x^2 + 2x + 1"""
        return x**2 + 2*x + 1
    
    def derivative_function(self, x: float) -> float:
        """Analytical derivative: f'(x) = 2x + 2"""
        return 2*x + 2
    
    def numerical_derivative(self, func: Callable, x: float, h: float = 1e-7) -> float:
        """
        Calculate derivative using numerical approximation.
        This mimics how AI frameworks compute gradients.
        """
        return (func(x + h) - func(x - h)) / (2 * h)
    
    def compare_derivatives(self, x: float) -> Dict[str, float]:
        """Compare analytical vs numerical derivative calculations"""
        analytical = self.derivative_function(x)
        numerical = self.numerical_derivative(self.simple_function, x)
        error = abs(analytical - numerical)
        
        return {
            'x': x,
            'analytical': analytical,
            'numerical': numerical,
            'error': error
        }
    
    def gradient_descent_1d(self, func: Callable, derivative_func: Callable, 
                           start_x: float, learning_rate: float = 0.1, 
                           steps: int = 50) -> Tuple[float, List[float]]:
        """
        Implement gradient descent - the core algorithm behind neural network training
        """
        x = start_x
        history = [x]
        
        print(f"🎯 Starting gradient descent from x = {start_x}")
        
        for step in range(steps):
            gradient = derivative_func(x)
            x = x - learning_rate * gradient
            history.append(x)
            
            if step % 10 == 0:
                print(f"   Step {step}: x = {x:.4f}, f(x) = {func(x):.4f}")
        
        print(f"✅ Converged to x = {x:.4f}, minimum value = {func(x):.4f}")
        return x, history
    
    def loss_function(self, prediction: float, actual: float) -> float:
        """Squared error loss function used in neural networks"""
        return (prediction - actual) ** 2
    
    def loss_derivative(self, prediction: float, actual: float) -> float:
        """Derivative of loss with respect to prediction"""
        return 2 * (prediction - actual)
    
    def simulate_ai_learning(self, initial_prediction: float, target: float, 
                           learning_rate: float = 0.1, steps: int = 20) -> List[Dict]:
        """
        Simulate how a neural network learns to make better predictions
        """
        prediction = initial_prediction
        history = []
        
        print(f"🤖 Simulating AI learning: Target = {target}, Starting prediction = {initial_prediction}")
        
        for step in range(steps):
            loss = self.loss_function(prediction, target)
            gradient = self.loss_derivative(prediction, target)
            
            history.append({
                'step': step,
                'prediction': prediction,
                'loss': loss,
                'gradient': gradient
            })
            
            # Update prediction using gradient descent
            prediction = prediction - learning_rate * gradient
            
            if step % 5 == 0:
                print(f"   Step {step}: Prediction = {prediction:.3f}, Loss = {loss:.3f}")
        
        print(f"✅ Final prediction: {prediction:.3f} (target: {target})")
        return history
    
    def visualize_function_and_derivative(self):
        """Create visualization showing function and its derivative"""
        x_range = np.linspace(-6, 4, 100)
        y_values = [self.simple_function(x) for x in x_range]
        derivative_values = [self.derivative_function(x) for x in x_range]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot function
        ax1.plot(x_range, y_values, 'b-', linewidth=2, label='f(x) = x² + 2x + 1')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('Function: What AI is trying to minimize')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot derivative
        ax2.plot(x_range, derivative_values, 'r-', linewidth=2, label="f'(x) = 2x + 2")
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('x')
        ax2.set_ylabel("f'(x)")
        ax2.set_title('Derivative: Tells AI which direction to move')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('function_and_derivative.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualization saved as 'function_and_derivative.png'")
    
    def visualize_gradient_descent(self):
        """Visualize the gradient descent learning process"""
        x_range = np.linspace(-6, 4, 100)
        y_values = [self.simple_function(x) for x in x_range]
        
        plt.figure(figsize=(12, 8))
        plt.plot(x_range, y_values, 'b-', linewidth=3, label='f(x) = x² + 2x + 1', alpha=0.7)
        
        # Show multiple gradient descent paths
        start_points = [4, -5, 2, -3]
        colors = ['red', 'green', 'orange', 'purple']
        
        for start_x, color in zip(start_points, colors):
            _, path = self.gradient_descent_1d(
                self.simple_function, self.derivative_function, 
                start_x, learning_rate=0.3, steps=20
            )
            path_y = [self.simple_function(x) for x in path]
            
            # Plot every 3rd point for clarity
            plt.plot(path[::3], path_y[::3], 'o-', color=color, markersize=6, 
                    label=f'Path from x={start_x}', alpha=0.8)
            plt.plot(path[0], path_y[0], 's', color=color, markersize=10, alpha=0.9)
        
        # Mark the true minimum
        true_min_x = -1  # From calculus: f'(x) = 0 when 2x + 2 = 0, so x = -1
        true_min_y = self.simple_function(true_min_x)
        plt.plot(true_min_x, true_min_y, '*', color='gold', markersize=20, 
                label='True Minimum', markeredgecolor='black', markeredgewidth=2)
        
        plt.xlabel('Parameter Value', fontsize=12)
        plt.ylabel('Loss/Error', fontsize=12)
        plt.title('How AI Models Learn: Multiple Gradient Descent Paths', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig('gradient_descent_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Gradient descent visualization saved as 'gradient_descent_visualization.png'")

def main():
    """Main demonstration of calculus concepts for AI/ML"""
    
    print("=" * 60)
    print("🎓 Day 12: Introduction to Calculus for AI/ML")
    print("=" * 60)
    
    # Initialize toolkit
    toolkit = AICalculusToolkit()
    
    # 1. Compare analytical vs numerical derivatives
    print("\n📐 1. Comparing Derivative Calculation Methods")
    print("-" * 50)
    test_points = [0, 1, -2, 3.5]
    for x in test_points:
        result = toolkit.compare_derivatives(x)
        print(f"x = {result['x']:4.1f}: Analytical = {result['analytical']:6.3f}, "
              f"Numerical = {result['numerical']:6.3f}, Error = {result['error']:.2e}")
    
    # 2. Demonstrate gradient descent
    print("\n🎯 2. Gradient Descent Optimization")
    print("-" * 50)
    minimum_x, path = toolkit.gradient_descent_1d(
        toolkit.simple_function, 
        toolkit.derivative_function,
        start_x=5,
        learning_rate=0.2,
        steps=30
    )
    
    # 3. Simulate AI learning process
    print("\n🤖 3. AI Learning Simulation")
    print("-" * 50)
    learning_history = toolkit.simulate_ai_learning(
        initial_prediction=8.0,
        target=3.0,
        learning_rate=0.15,
        steps=25
    )
    
    # 4. Create visualizations
    print("\n📊 4. Creating Visualizations")
    print("-" * 50)
    toolkit.visualize_function_and_derivative()
    toolkit.visualize_gradient_descent()
    
    # 5. Summary and insights
    print("\n🎉 5. Key Insights")
    print("-" * 50)
    print("✅ Derivatives tell us the rate of change (slope) at any point")
    print("✅ Numerical derivatives approximate analytical ones with high accuracy")
    print("✅ Gradient descent uses derivatives to find function minimums")
    print("✅ This is exactly how neural networks learn from their mistakes")
    print("✅ AI optimization is fundamentally about following derivatives downhill")
    
    print(f"\n🎯 Theoretical minimum at x = -1, f(-1) = {toolkit.simple_function(-1)}")
    print(f"🎯 Our algorithm found minimum at x = {minimum_x:.4f}")
    print(f"🎯 Difference: {abs(-1 - minimum_x):.6f}")
    
    print("\n" + "=" * 60)
    print("🚀 Ready for Day 13: Derivatives and Their Applications!")
    print("=" * 60)

if __name__ == "__main__":
    main()
