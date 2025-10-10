"""
Day 13: Derivatives and Their Applications in AI/ML
A hands-on implementation of gradient descent using derivatives
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Tuple, List
import json

class DerivativeCalculator:
    """Utility class for calculating derivatives numerically and analytically"""
    
    @staticmethod
    def numerical_derivative(func, x: float, h: float = 1e-7) -> float:
        """Calculate derivative using numerical approximation"""
        return (func(x + h) - func(x - h)) / (2 * h)
    
    @staticmethod
    def analytical_derivative_quadratic(a: float, b: float, c: float, x: float) -> float:
        """Calculate analytical derivative of ax² + bx + c"""
        return 2 * a * x + b

class SimplePredictor:
    """A simple linear predictor that learns using gradient descent"""
    
    def __init__(self, learning_rate: float = 0.01):
        # Initialize parameters randomly
        self.weight = np.random.normal(0, 0.1)
        self.bias = np.random.normal(0, 0.1)
        self.learning_rate = learning_rate
        
        # Track training progress
        self.training_history = []
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions using current parameters"""
        return self.weight * x + self.bias
    
    def calculate_error(self, x: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate mean squared error"""
        predictions = self.predict(x)
        return np.mean((predictions - y_true) ** 2)
    
    def calculate_gradients(self, x: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
        """Calculate gradients of the error with respect to parameters"""
        predictions = self.predict(x)
        errors = predictions - y_true
        n = len(x)
        
        # Derivatives of MSE with respect to weight and bias
        weight_gradient = (2 / n) * np.sum(errors * x)
        bias_gradient = (2 / n) * np.sum(errors)
        
        return weight_gradient, bias_gradient
    
    def train_step(self, x: np.ndarray, y_true: np.ndarray) -> float:
        """Perform one training step using gradient descent"""
        # Calculate gradients
        weight_grad, bias_grad = self.calculate_gradients(x, y_true)
        
        # Update parameters (move opposite to gradient direction)
        self.weight -= self.learning_rate * weight_grad
        self.bias -= self.learning_rate * bias_grad
        
        # Calculate and return current error
        error = self.calculate_error(x, y_true)
        self.training_history.append({
            'error': error,
            'weight': self.weight,
            'bias': self.bias,
            'weight_grad': weight_grad,
            'bias_grad': bias_grad
        })
        
        return error
    
    def train(self, x: np.ndarray, y_true: np.ndarray, epochs: int = 100) -> List[float]:
        """Train the model for multiple epochs"""
        errors = []
        
        for epoch in range(epochs):
            error = self.train_step(x, y_true)
            errors.append(error)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Error = {error:.6f}, Weight = {self.weight:.4f}, Bias = {self.bias:.4f}")
        
        return errors

class HousePricePredictor:
    """A practical example of using derivatives for house price prediction"""
    
    def __init__(self):
        self.predictor = SimplePredictor(learning_rate=0.00001)  # Much smaller learning rate
        self.x_mean = 0
        self.x_std = 1
        self.y_mean = 0
        self.y_std = 1
        
    def normalize_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize data to prevent numerical overflow"""
        self.x_mean = np.mean(x)
        self.x_std = np.std(x)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        
        return x_norm, y_norm
    
    def denormalize_prediction(self, y_pred_norm: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scale"""
        return y_pred_norm * self.y_std + self.y_mean
    
    def generate_sample_data(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic house price data"""
        np.random.seed(42)  # For reproducible results
        
        # Square footage (1000-4000 sq ft)
        square_feet = np.random.uniform(1000, 4000, n_samples)
        
        # True relationship: price = 150 * sqft + 50000 + noise
        true_price_per_sqft = 150
        true_base_price = 50000
        noise = np.random.normal(0, 10000, n_samples)
        
        prices = true_price_per_sqft * square_feet + true_base_price + noise
        
        return square_feet, prices
    
    def train_and_visualize(self):
        """Train the model and create visualizations"""
        print("🏠 Generating house price data...")
        x, y = self.generate_sample_data()
        
        # Normalize data to prevent numerical overflow
        x_norm, y_norm = self.normalize_data(x, y)
        
        print("📚 Training the model using derivatives...")
        errors = self.predictor.train(x_norm, y_norm, epochs=200)
        
        # Create visualizations
        self.plot_training_progress(errors)
        self.plot_final_predictions(x, y)
        self.plot_gradient_descent_path()
        
        # Print final results (convert back to original scale)
        # The learned parameters are in normalized space, so we need to convert them
        # For a linear model y = wx + b, if we normalize both x and y:
        # y_norm = w_norm * x_norm + b_norm
        # Converting back: y = w_norm * (x - x_mean)/x_std * y_std + y_mean + b_norm * y_std
        # So: w_original = w_norm * y_std / x_std, b_original = y_mean + b_norm * y_std - w_norm * x_mean * y_std / x_std
        
        w_original = self.predictor.weight * self.y_std / self.x_std
        b_original = self.y_mean + self.predictor.bias * self.y_std - self.predictor.weight * self.x_mean * self.y_std / self.x_std
        
        print(f"\n🎯 Final Results:")
        print(f"   Learned weight (price per sq ft): ${w_original:.2f}")
        print(f"   Learned bias (base price): ${b_original:.2f}")
        print(f"   Final error: {errors[-1]:.6f}")
        
        return self.predictor
    
    def plot_training_progress(self, errors: List[float]):
        """Plot how error decreases during training"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(errors, 'b-', linewidth=2)
        plt.title('Training Progress: Error vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(errors, 'r-', linewidth=2)
        plt.title('Training Progress: Log Scale')
        plt.xlabel('Epoch')
        plt.ylabel('Log(Mean Squared Error)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_final_predictions(self, x: np.ndarray, y: np.ndarray):
        """Plot the final learned relationship"""
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        plt.scatter(x, y, alpha=0.6, color='lightblue', label='Actual Prices')
        
        # Plot learned line (need to normalize x_line first)
        x_line = np.linspace(x.min(), x.max(), 100)
        x_line_norm = (x_line - self.x_mean) / self.x_std
        y_line_norm = self.predictor.predict(x_line_norm)
        y_line = self.denormalize_prediction(y_line_norm)
        plt.plot(x_line, y_line, 'r-', linewidth=3, label='Learned Relationship')
        
        plt.xlabel('Square Feet')
        plt.ylabel('Price ($)')
        plt.title('House Price Prediction: Actual vs Learned')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_gradient_descent_path(self):
        """Visualize how parameters changed during training"""
        history = self.predictor.training_history
        weights = [h['weight'] for h in history]
        biases = [h['bias'] for h in history]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(weights, 'g-', linewidth=2, marker='o', markersize=3)
        plt.title('Weight Parameter Over Time')
        plt.xlabel('Training Step')
        plt.ylabel('Weight Value')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(biases, 'purple', linewidth=2, marker='s', markersize=3)
        plt.title('Bias Parameter Over Time')
        plt.xlabel('Training Step')
        plt.ylabel('Bias Value')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_price(self, square_feet: float) -> float:
        """Make a price prediction for given square footage"""
        # Normalize the input
        x_norm = (square_feet - self.x_mean) / self.x_std
        # Make prediction in normalized space
        y_pred_norm = self.predictor.predict(np.array([x_norm]))[0]
        # Convert back to original scale
        return self.denormalize_prediction(np.array([y_pred_norm]))[0]

def interactive_derivative_demo():
    """Interactive demonstration of derivatives"""
    print("🧮 Interactive Derivative Demo")
    print("=" * 40)
    
    calc = DerivativeCalculator()
    
    # Define a quadratic function
    def quadratic(x):
        return 2 * x**2 + 3 * x + 1
    
    print("Function: f(x) = 2x² + 3x + 1")
    print("Analytical derivative: f'(x) = 4x + 3")
    
    test_points = [0, 1, 2, -1, 0.5]
    
    print("\nComparing numerical vs analytical derivatives:")
    print("x\tNumerical\tAnalytical\tDifference")
    print("-" * 50)
    
    for x in test_points:
        numerical = calc.numerical_derivative(quadratic, x)
        analytical = calc.analytical_derivative_quadratic(2, 3, 1, x)
        difference = abs(numerical - analytical)
        
        print(f"{x:.1f}\t{numerical:.6f}\t{analytical:.6f}\t{difference:.8f}")

def main():
    """Main function that runs the complete lesson"""
    print("🎓 Day 13: Derivatives and Their Applications in AI/ML")
    print("=" * 60)
    
    # Part 1: Interactive derivative demo
    interactive_derivative_demo()
    
    print("\n" + "=" * 60)
    
    # Part 2: House price prediction with gradient descent
    house_predictor = HousePricePredictor()
    trained_model = house_predictor.train_and_visualize()
    
    # Part 3: Sample predictions (non-interactive for this environment)
    print("\n🔮 Sample predictions:")
    test_sizes = [1500, 2000, 2500, 3000]
    for size in test_sizes:
        predicted_price = house_predictor.predict_price(size)
        print(f"Predicted price for {size:.0f} sq ft: ${predicted_price:,.2f}")
    
    print("\n🎉 Lesson complete! You've successfully implemented gradient descent using derivatives.")
    print("💡 Key takeaways:")
    print("   • Derivatives tell us the direction of steepest change")
    print("   • Gradient descent uses derivatives to minimize error")
    print("   • This is the foundation of how AI systems learn!")

if __name__ == "__main__":
    main()
