#!/usr/bin/env python3
"""
Day 14: Chain Rule and Partial Derivatives for AI/ML
A hands-on implementation demonstrating how these concepts power neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sympy as sp
from sympy import symbols, diff, lambdify
import math

class ChainRuleDemo:
    """Demonstrates chain rule and partial derivatives in neural network context."""
    
    def __init__(self):
        self.setup_visualization()
    
    def setup_visualization(self):
        """Setup matplotlib for beautiful visualizations."""
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def demonstrate_chain_rule_basics(self):
        """Show basic chain rule with symbolic math."""
        print("🔗 CHAIN RULE FUNDAMENTALS")
        print("=" * 50)
        
        # Define symbolic variables
        x = symbols('x')
        
        # Create a composite function: f(g(x)) where g(x) = x² and f(u) = sin(u)
        g = x**2
        f_of_g = sp.sin(g)
        
        print(f"Original function: f(g(x)) = sin(x²)")
        print(f"Inner function g(x) = {g}")
        print(f"Outer function f(u) = sin(u)")
        print()
        
        # Calculate derivatives
        g_prime = diff(g, x)
        f_prime = sp.cos(g)  # derivative of sin(u) is cos(u)
        chain_rule_result = f_prime * g_prime
        
        print("🧮 Chain Rule Calculation:")
        print(f"g'(x) = {g_prime}")
        print(f"f'(u) = cos(u), so f'(g(x)) = cos(x²)")
        print(f"Chain rule: f'(g(x)) × g'(x) = cos(x²) × 2x = {chain_rule_result}")
        
        # Verify with sympy's automatic differentiation
        auto_derivative = diff(f_of_g, x)
        print(f"SymPy verification: {auto_derivative}")
        print(f"✅ Chain rule matches automatic differentiation!")
        print()
        
    def demonstrate_partial_derivatives(self):
        """Show partial derivatives in multivariable context."""
        print("∂ PARTIAL DERIVATIVES IN AI")
        print("=" * 50)
        
        # Define a loss function L(w1, w2) = (w1*x1 + w2*x2 - target)²
        w1, w2, x1, x2, target = symbols('w1 w2 x1 x2 target')
        
        # Neural network prediction
        prediction = w1*x1 + w2*x2
        loss = (prediction - target)**2
        
        print("🎯 Loss Function for Simple Neural Network:")
        print(f"Prediction = w1×x1 + w2×x2")
        print(f"Loss = (prediction - target)² = {loss}")
        print()
        
        # Calculate partial derivatives
        partial_w1 = diff(loss, w1)
        partial_w2 = diff(loss, w2)
        
        print("📊 Partial Derivatives (Gradients):")
        print(f"∂L/∂w1 = {partial_w1}")
        print(f"∂L/∂w2 = {partial_w2}")
        print()
        print("💡 These gradients tell us how to adjust each weight to reduce loss!")
        print()
    
    def neural_network_forward_backward(self):
        """Demonstrate chain rule in a simple neural network."""
        print("🧠 NEURAL NETWORK: CHAIN RULE IN ACTION")
        print("=" * 50)
        
        # Network architecture: Input -> Hidden -> Output
        # Simple example with 1 input, 2 hidden neurons, 1 output
        
        print("🏗️  Network Architecture:")
        print("Input (1) -> Hidden Layer (2 neurons) -> Output (1)")
        print()
        
        # Initialize weights and biases
        np.random.seed(42)  # For reproducible results
        w1 = np.random.randn(2, 1) * 0.5  # Input to hidden weights
        b1 = np.zeros((2, 1))              # Hidden biases
        w2 = np.random.randn(1, 2) * 0.5  # Hidden to output weights
        b2 = np.zeros((1, 1))              # Output bias
        
        # Input data
        x = np.array([[0.8]])  # Single input
        target = np.array([[0.2]])  # Target output
        
        print(f"📥 Input: {x[0,0]}")
        print(f"🎯 Target: {target[0,0]}")
        print()
        
        # Forward pass with detailed chain rule tracking
        print("➡️  FORWARD PASS:")
        z1 = np.dot(w1, x) + b1  # Linear combination in hidden layer
        print(f"Hidden layer linear: z1 = W1×x + b1")
        print(f"z1 = {z1.flatten()}")
        
        a1 = self.sigmoid(z1)  # Hidden layer activation
        print(f"Hidden layer activation: a1 = sigmoid(z1)")
        print(f"a1 = {a1.flatten()}")
        
        z2 = np.dot(w2, a1) + b2  # Output layer linear
        print(f"Output layer linear: z2 = W2×a1 + b2")
        print(f"z2 = {z2.flatten()}")
        
        a2 = self.sigmoid(z2)  # Output layer activation
        print(f"Final output: a2 = sigmoid(z2)")
        print(f"a2 = {a2.flatten()}")
        
        # Calculate loss
        loss = 0.5 * (a2 - target)**2
        print(f"📊 Loss = 0.5×(prediction - target)² = {loss[0,0]:.6f}")
        print()
        
        # Backward pass - Chain rule in action!
        print("⬅️  BACKWARD PASS (Chain Rule):")
        
        # Output layer gradients
        dL_da2 = a2 - target
        da2_dz2 = self.sigmoid_derivative(z2)
        dL_dz2 = dL_da2 * da2_dz2
        
        print(f"∂L/∂a2 = (a2 - target) = {dL_da2[0,0]:.6f}")
        print(f"∂a2/∂z2 = sigmoid'(z2) = {da2_dz2[0,0]:.6f}")
        print(f"∂L/∂z2 = ∂L/∂a2 × ∂a2/∂z2 = {dL_dz2[0,0]:.6f}")
        print()
        
        # Hidden layer gradients (chain rule gets longer!)
        dL_dw2 = np.dot(dL_dz2, a1.T)
        dL_da1 = np.dot(w2.T, dL_dz2)
        da1_dz1 = self.sigmoid_derivative(z1)
        dL_dz1 = dL_da1 * da1_dz1
        dL_dw1 = np.dot(dL_dz1, x.T)
        
        print("🔗 Chain rule through hidden layer:")
        print(f"∂L/∂W2 = ∂L/∂z2 × ∂z2/∂W2 = {dL_dw2.flatten()}")
        print(f"∂L/∂a1 = ∂L/∂z2 × ∂z2/∂a1 = {dL_da1.flatten()}")
        print(f"∂L/∂z1 = ∂L/∂a1 × ∂a1/∂z1 = {dL_dz1.flatten()}")
        print(f"∂L/∂W1 = ∂L/∂z1 × ∂z1/∂W1 = {dL_dw1.flatten()}")
        print()
        
        print("🎉 The chain rule just calculated gradients for all weights!")
        print("This is exactly how neural networks learn from their mistakes.")
        
    def visualize_chain_rule_flow(self):
        """Create a visual representation of chain rule flow."""
        print("\n📈 GENERATING CHAIN RULE VISUALIZATION...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Function composition
        x = np.linspace(-3, 3, 1000)
        
        # Composite function: f(g(x)) = sin(x²)
        inner_func = x**2
        outer_func = np.sin(inner_func)
        
        ax1.plot(x, inner_func, 'b-', label='g(x) = x²', linewidth=2)
        ax1.plot(x, outer_func, 'r-', label='f(g(x)) = sin(x²)', linewidth=2)
        ax1.set_title('Chain Rule: Composite Functions', fontsize=14, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5)
        
        # Right plot: Neural network flow
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 8)
        ax2.set_title('Chain Rule in Neural Networks', fontsize=14, fontweight='bold')
        
        # Draw network layers
        layers = [
            {'x': 1, 'y': 4, 'label': 'Input\nx', 'color': 'lightblue'},
            {'x': 3.5, 'y': 5, 'label': 'Hidden\nz₁=W₁x', 'color': 'lightgreen'},
            {'x': 3.5, 'y': 3, 'label': 'Activation\na₁=σ(z₁)', 'color': 'lightgreen'},
            {'x': 6, 'y': 4, 'label': 'Output\nz₂=W₂a₁', 'color': 'lightcoral'},
            {'x': 8.5, 'y': 4, 'label': 'Loss\nL=(y-ŷ)²', 'color': 'orange'}
        ]
        
        # Draw nodes
        for layer in layers:
            circle = plt.Circle((layer['x'], layer['y']), 0.5, 
                              color=layer['color'], alpha=0.7)
            ax2.add_patch(circle)
            ax2.text(layer['x'], layer['y']-1, layer['label'], 
                    ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw forward arrows
        arrows = [(1.5, 4, 1.4, 0.8), (4, 4.7, 1.4, -0.4), (4, 3.3, 1.4, 0.4), (6.5, 4, 1.4, 0)]
        for arrow in arrows:
            ax2.arrow(arrow[0], arrow[1], arrow[2], arrow[3], 
                     head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Draw backward arrows (chain rule)
        back_arrows = [(8, 4, -1.4, 0), (5.5, 4, -1.4, -0.4), (3, 3.3, -1.4, 0.4), (3, 4.7, -1.4, -0.8)]
        for arrow in back_arrows:
            ax2.arrow(arrow[0], arrow[1], arrow[2], arrow[3], 
                     head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        
        ax2.text(5, 1.5, '➡️ Forward Pass', fontsize=12, ha='center', color='black')
        ax2.text(5, 0.8, '⬅️ Backward Pass (Chain Rule)', fontsize=12, ha='center', color='red')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('chain_rule_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved as 'chain_rule_visualization.png'")

def main():
    """Run the complete Day 14 demonstration."""
    print("🎓 Welcome to Day 14: Chain Rule and Partial Derivatives!")
    print("=" * 60)
    print()
    
    demo = ChainRuleDemo()
    
    # Run all demonstrations
    demo.demonstrate_chain_rule_basics()
    demo.demonstrate_partial_derivatives()
    demo.neural_network_forward_backward()
    demo.visualize_chain_rule_flow()
    
    print("\n🎉 CONGRATULATIONS!")
    print("=" * 60)
    print("You've just mastered the mathematical foundation of neural networks!")
    print("🔗 Chain rule: How changes propagate through complex functions")
    print("∂ Partial derivatives: How to optimize multivariable functions")
    print("🧠 Neural networks: Where these concepts power AI learning")
    print()
    print("🚀 Tomorrow (Day 15): We'll use these concepts to implement gradient descent!")
    print("Get ready to build the optimization algorithm that trains all modern AI.")

if __name__ == "__main__":
    main()
