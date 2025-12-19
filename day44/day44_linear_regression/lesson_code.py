"""
Day 44: Simple Linear Regression Theory - From Scratch Implementation

This module implements simple linear regression using gradient descent,
demonstrating the mathematical foundation of predictive AI systems.

Real-world applications:
- Zillow: House price prediction
- Netflix: Rating predictions
- Tesla: Stopping distance calculations
- Amazon: Demand forecasting
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys


class SimpleLinearRegression:
    """
    Simple Linear Regression implementation using gradient descent.
    
    Finds optimal line: y = wx + b that minimizes Mean Squared Error.
    
    This is the same algorithm (scaled up) used in production systems
    processing millions of predictions per second.
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        """
        Initialize the linear regression model.
        
        Args:
            learning_rate: Step size for gradient descent (alpha)
            iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = 0.0  # Weight (slope)
        self.b = 0.0  # Bias (intercept)
        self.loss_history = []  # Track training progress
        self.X_mean = 0.0  # For feature scaling
        self.X_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0
        
    def _calculate_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate predictions using current parameters.
        
        Formula: ŷ = wx + b
        
        Args:
            X: Input features (shape: n_samples,)
            
        Returns:
            Predictions (shape: n_samples,)
        """
        return self.w * X + self.b
    
    def _calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error loss.
        
        Formula: MSE = (1/n) × Σ(y_true - y_pred)²
        
        Why squared? Penalizes large errors heavily and enables
        smooth gradient computation.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Mean Squared Error
        """
        n = len(y_true)
        return np.sum((y_true - y_pred) ** 2) / n
    
    def _calculate_gradients(self, X: np.ndarray, y: np.ndarray, 
                            y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Calculate gradients of loss with respect to w and b.
        
        These gradients tell us which direction reduces error.
        
        Formulas:
            ∂MSE/∂w = -(2/n) × Σ x(y - ŷ)
            ∂MSE/∂b = -(2/n) × Σ (y - ŷ)
        
        Args:
            X: Input features
            y: True values
            y_pred: Predicted values
            
        Returns:
            Tuple of (gradient_w, gradient_b)
        """
        n = len(y)
        error = y - y_pred
        
        # Gradient with respect to weight (slope)
        gradient_w = -(2 / n) * np.sum(X * error)
        
        # Gradient with respect to bias (intercept)
        gradient_b = -(2 / n) * np.sum(error)
        
        return gradient_w, gradient_b
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleLinearRegression':
        """
        Train the model using gradient descent.
        
        This is where the magic happens: we iteratively adjust w and b
        to minimize prediction errors across all training examples.
        
        Args:
            X: Training features (shape: n_samples,)
            y: Training targets (shape: n_samples,)
            
        Returns:
            self (for method chaining)
        """
        # Convert to numpy arrays if needed
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Feature scaling to prevent numerical overflow
        # Store scaling parameters for prediction
        self.X_mean = np.mean(X)
        self.X_std = np.std(X)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        
        # Avoid division by zero
        if self.X_std < 1e-10:
            self.X_std = 1.0
        if self.y_std < 1e-10:
            self.y_std = 1.0
        
        # Normalize features and targets
        X_norm = (X - self.X_mean) / self.X_std
        y_norm = (y - self.y_mean) / self.y_std
        
        # Gradient descent optimization
        for iteration in range(self.iterations):
            # Forward pass: compute predictions
            y_pred_norm = self._calculate_predictions(X_norm)
            
            # Calculate loss for monitoring (on normalized data)
            loss = self._calculate_mse(y_norm, y_pred_norm)
            self.loss_history.append(loss)
            
            # Backward pass: compute gradients
            grad_w, grad_b = self._calculate_gradients(X_norm, y_norm, y_pred_norm)
            
            # Gradient clipping to prevent overflow
            max_grad = 1e6
            grad_w = np.clip(grad_w, -max_grad, max_grad)
            grad_b = np.clip(grad_b, -max_grad, max_grad)
            
            # Update parameters (gradient descent step)
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b
            
            # Check for NaN/Inf and reset if needed
            if not np.isfinite(self.w) or not np.isfinite(self.b):
                self.w = 0.0
                self.b = 0.0
            
            # Print progress every 100 iterations
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.iterations} - Loss: {loss:.4f}")
        
        # Transform parameters back to original scale
        # y_norm = w_norm * X_norm + b_norm
        # y = y_norm * y_std + y_mean = (w_norm * X_norm + b_norm) * y_std + y_mean
        # y = w_norm * y_std / X_std * (X - X_mean) + (b_norm * y_std + y_mean - w_norm * y_std * X_mean / X_std)
        # So: w = w_norm * y_std / X_std, b = b_norm * y_std + y_mean - w * X_mean
        w_original = self.w * self.y_std / self.X_std
        b_original = self.b * self.y_std + self.y_mean - w_original * self.X_mean
        self.w = w_original
        self.b = b_original
        
        print(f"\nTraining complete!")
        print(f"Final parameters: w={self.w:.4f}, b={self.b:.4f}")
        print(f"Final MSE: {self.loss_history[-1]:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using learned parameters.
        
        In production, this runs millions of times per second.
        Training is expensive, prediction is cheap!
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        X = np.array(X)
        return self._calculate_predictions(X)
    
    def get_parameters(self) -> Tuple[float, float]:
        """Return learned parameters (w, b)"""
        return self.w, self.b


def generate_sample_data(n_samples: int = 100, noise: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset for demonstration.
    
    Simulates house price data: price increases with square footage,
    plus some random noise (real-world data is never perfect).
    
    Args:
        n_samples: Number of data points
        noise: Amount of random variation
        
    Returns:
        Tuple of (X, y) where X is features, y is targets
    """
    np.random.seed(42)  # Reproducibility
    
    # Feature: Square footage (1000 to 3000 sq ft)
    X = np.random.uniform(1000, 3000, n_samples)
    
    # Target: House price with linear relationship + noise
    # Formula: price = 150 * sqft + 50000 + noise
    true_w = 150  # $150 per square foot
    true_b = 50000  # Base price
    
    y = true_w * X + true_b + np.random.normal(0, noise * 1000, n_samples)
    
    return X, y


def visualize_regression(X: np.ndarray, y: np.ndarray, model: SimpleLinearRegression, 
                        save_path: str = 'regression_line.png'):
    """
    Visualize the fitted regression line.
    
    Shows how well our learned line fits the training data.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X, y, alpha=0.5, label='Training Data', color='blue')
    
    # Plot regression line
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label=f'Fitted Line: y={model.w:.2f}x + {model.b:.2f}')
    
    plt.xlabel('Square Footage')
    plt.ylabel('House Price ($)')
    plt.title('Simple Linear Regression: House Price Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved regression line visualization to {save_path}")
    plt.close()


def visualize_loss_curve(model: SimpleLinearRegression, save_path: str = 'loss_curve.png'):
    """
    Visualize training loss over iterations.
    
    A good loss curve: rapid initial decrease, then plateau.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(model.loss_history, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    
    # Annotate final loss
    final_loss = model.loss_history[-1]
    plt.annotate(f'Final Loss: {final_loss:.2f}',
                xy=(len(model.loss_history)-1, final_loss),
                xytext=(-50, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved loss curve to {save_path}")
    plt.close()


def visualize_predictions(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                         save_path: str = 'predictions.png'):
    """
    Visualize predicted vs actual values.
    
    Perfect predictions would lie on the diagonal line.
    """
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Predictions')
    
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Predicted vs Actual House Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved predictions visualization to {save_path}")
    plt.close()


def run_demo():
    """
    Run complete demonstration of simple linear regression.
    """
    print("=" * 60)
    print("Day 44: Simple Linear Regression Theory Demo")
    print("=" * 60)
    print()
    
    # Generate sample data
    print("Generating sample house price data...")
    X, y = generate_sample_data(n_samples=100, noise=10.0)
    print(f"Generated {len(X)} samples")
    print(f"Features range: {X.min():.0f} - {X.max():.0f} sq ft")
    print(f"Prices range: ${y.min():.0f} - ${y.max():.0f}")
    print()
    
    # Train model
    print("Training linear regression model...")
    model = SimpleLinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X, y)
    print()
    
    # Make predictions
    print("Making predictions on training data...")
    y_pred = model.predict(X)
    final_mse = np.mean((y - y_pred) ** 2)
    print(f"Final MSE: {final_mse:.2f}")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    visualize_regression(X, y, model)
    visualize_loss_curve(model)
    visualize_predictions(X, y, y_pred)
    print()
    
    # Example predictions
    print("Example predictions:")
    test_sizes = [1500, 2000, 2500]
    for size in test_sizes:
        predicted_price = model.predict([size])[0]
        print(f"  {size} sq ft → ${predicted_price:,.0f}")
    print()
    
    print("=" * 60)
    print("Demo complete! Check the generated PNG files.")
    print("=" * 60)


def test_setup():
    """Quick test to verify environment is working."""
    print("Testing environment setup...")
    print(f"NumPy version: {np.__version__}")
    print(f"Matplotlib version: {plt.matplotlib.__version__}")
    print("✓ All imports successful!")
    print("✓ Environment ready!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-setup":
            test_setup()
        elif sys.argv[1] == "--train":
            # Just training
            X, y = generate_sample_data()
            model = SimpleLinearRegression(learning_rate=0.01, iterations=1000)
            model.fit(X, y)
        elif sys.argv[1] == "--demo":
            run_demo()
        elif sys.argv[1] == "--visualize":
            X, y = generate_sample_data()
            model = SimpleLinearRegression(learning_rate=0.01, iterations=1000)
            model.fit(X, y)
            y_pred = model.predict(X)
            visualize_regression(X, y, model)
            visualize_loss_curve(model)
            visualize_predictions(X, y, y_pred)
    else:
        print("Usage:")
        print("  python lesson_code.py --test-setup  # Test environment")
        print("  python lesson_code.py --train       # Train model")
        print("  python lesson_code.py --demo        # Full demo")
        print("  python lesson_code.py --visualize   # Generate plots")
