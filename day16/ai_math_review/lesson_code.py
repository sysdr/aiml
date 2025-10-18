"""
Day 16-22: Linear Algebra & Calculus Review
AI/ML Mathematical Foundation Practice Problems
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import sympy as sp
from sympy import symbols, diff, solve, Matrix
import seaborn as sns

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🧮 Welcome to AI/ML Math Review - Day 16-22")
print("=" * 50)

class MathReviewSession:
    """Interactive math review session for AI/ML foundations"""
    
    def __init__(self):
        self.score = 0
        self.total_problems = 0
    
    def vector_operations_review(self):
        """Review vector operations used in AI systems"""
        print("\n🔢 SECTION 1: Vector Operations for AI")
        print("-" * 40)
        
        # Problem 1: User similarity (like recommendation systems)
        print("Problem 1: User Preference Similarity")
        user_a = np.array([4.2, 3.8, 2.1, 4.9, 1.3])  # Genres: Action, Comedy, Drama, Sci-fi, Horror
        user_b = np.array([4.1, 3.9, 2.3, 4.7, 1.1])
        user_c = np.array([1.2, 4.8, 4.9, 1.3, 0.8])  # Different preferences
        
        print(f"User A preferences: {user_a}")
        print(f"User B preferences: {user_b}")
        print(f"User C preferences: {user_c}")
        
        # Calculate similarities
        sim_ab = self.cosine_similarity(user_a, user_b)
        sim_ac = self.cosine_similarity(user_a, user_c)
        
        print(f"Similarity A-B: {sim_ab:.3f}")
        print(f"Similarity A-C: {sim_ac:.3f}")
        print(f"✨ Users A and B are more similar (like Netflix recommendations!)")
        
        # Problem 2: Feature vector operations
        print("\nProblem 2: Feature Vector Operations")
        image_features = np.array([0.8, 0.2, 0.9, 0.1])  # brightness, contrast, saturation, noise
        adjustment = np.array([0.1, -0.05, 0.05, -0.02])  # enhancement vector
        
        enhanced_features = image_features + adjustment
        print(f"Original features: {image_features}")
        print(f"Enhanced features: {enhanced_features}")
        print("🖼️  This is how AI enhances images!")
        
        self.total_problems += 2
        return True
    
    def matrix_operations_review(self):
        """Review matrix operations for neural networks"""
        print("\n🏗️  SECTION 2: Matrix Operations for Neural Networks")
        print("-" * 40)
        
        # Problem 3: Simple neural network layer
        print("Problem 3: Neural Network Layer Forward Pass")
        
        # Input: batch of 2 samples, each with 3 features
        X = np.array([[1.0, 2.0, 3.0],
                      [2.0, 3.0, 1.0]])
        
        # Weights: 3 input features to 2 output neurons
        W = np.array([[0.1, 0.5],
                      [0.2, 0.3],
                      [0.4, 0.1]])
        
        # Bias for 2 neurons
        b = np.array([0.1, 0.2])
        
        # Forward pass: Y = XW + b
        Y = np.dot(X, W) + b
        
        print(f"Input shape: {X.shape}")
        print(f"Weights shape: {W.shape}")
        print(f"Output shape: {Y.shape}")
        print(f"Output values:\n{Y}")
        print("🧠 This is how neural networks transform data!")
        
        # Problem 4: Data transformation matrices
        print("\nProblem 4: Image Rotation Matrix")
        angle = np.pi / 4  # 45 degrees
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]])
        
        # Original point
        point = np.array([1, 1])
        rotated_point = np.dot(rotation_matrix, point)
        
        print(f"Original point: {point}")
        print(f"Rotated point: {rotated_point}")
        print("🔄 This is how AI rotates images for data augmentation!")
        
        self.total_problems += 2
        return True
    
    def calculus_review(self):
        """Review calculus concepts for optimization"""
        print("\n📐 SECTION 3: Calculus for AI Optimization")
        print("-" * 40)
        
        # Problem 5: Gradient descent visualization
        print("Problem 5: Gradient Descent Optimization")
        
        def loss_function(x):
            return x**2 - 4*x + 5  # Minimum at x=2
        
        def gradient(x):
            return 2*x - 4
        
        # Gradient descent
        x = 0.0  # Starting point
        learning_rate = 0.1
        history = [x]
        
        print("Gradient Descent Steps:")
        for step in range(10):
            grad = gradient(x)
            x = x - learning_rate * grad
            history.append(x)
            if step < 5:  # Show first 5 steps
                print(f"Step {step+1}: x={x:.3f}, loss={loss_function(x):.3f}, gradient={grad:.3f}")
        
        print(f"Final x: {x:.3f} (target: 2.0)")
        print("📉 This is how AI models learn by minimizing loss!")
        
        # Problem 6: Partial derivatives (multi-variable)
        print("\nProblem 6: Partial Derivatives for Multi-Parameter Models")
        
        # Define symbolic variables
        x, y = symbols('x y')
        
        # Multi-variable function (like neural network loss)
        f = x**2 + 2*y**2 + x*y - 4*x - 2*y + 5
        
        # Calculate partial derivatives
        df_dx = diff(f, x)
        df_dy = diff(f, y)
        
        print(f"Function: f(x,y) = {f}")
        print(f"∂f/∂x = {df_dx}")
        print(f"∂f/∂y = {df_dy}")
        
        # Find critical point (where gradients = 0)
        critical_points = solve([df_dx, df_dy], [x, y])
        print(f"Critical point: {critical_points}")
        print("🎯 This is how AI finds optimal parameters!")
        
        self.total_problems += 2
        return True
    
    def advanced_practice(self):
        """Advanced problems connecting all concepts"""
        print("\n🚀 SECTION 4: Advanced AI Math Applications")
        print("-" * 40)
        
        # Problem 7: Principal Component Analysis (PCA) preview
        print("Problem 7: Eigenvalues and Data Compression Preview")
        
        # Create sample data matrix
        np.random.seed(42)
        data = np.random.randn(50, 2)
        data[:, 1] = data[:, 0] + 0.5 * np.random.randn(50)  # Correlated data
        
        # Covariance matrix
        cov_matrix = np.cov(data.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        print(f"Data shape: {data.shape}")
        print(f"Covariance matrix:\n{cov_matrix}")
        print(f"Eigenvalues: {eigenvalues}")
        print("📊 This is how AI compresses data while keeping important info!")
        
        # Problem 8: Backpropagation chain rule
        print("\nProblem 8: Chain Rule for Neural Network Training")
        
        # Simple computation graph: z = x * y, loss = z^2
        x_val, y_val = 2.0, 3.0
        z_val = x_val * y_val  # Forward pass
        loss_val = z_val**2
        
        # Backward pass (chain rule)
        dloss_dz = 2 * z_val
        dz_dx = y_val
        dz_dy = x_val
        
        dloss_dx = dloss_dz * dz_dx  # Chain rule
        dloss_dy = dloss_dz * dz_dy
        
        print(f"Forward: x={x_val}, y={y_val}, z={z_val}, loss={loss_val}")
        print(f"Backward: ∂loss/∂x={dloss_dx}, ∂loss/∂y={dloss_dy}")
        print("⛓️  This is backpropagation - how neural networks learn!")
        
        self.total_problems += 2
        return True
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product
    
    def create_visualizations(self):
        """Create helpful visualizations"""
        print("\n📊 Creating Visualizations...")
        
        # 1. Gradient descent visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss function curve
        x_range = np.linspace(-1, 5, 100)
        y_range = x_range**2 - 4*x_range + 5
        
        ax1.plot(x_range, y_range, 'b-', linewidth=2, label='Loss Function')
        ax1.scatter([2], [1], color='red', s=100, zorder=5, label='Minimum')
        ax1.set_xlabel('Parameter Value')
        ax1.set_ylabel('Loss')
        ax1.set_title('Gradient Descent Optimization')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Vector similarity heatmap
        users = ['User A', 'User B', 'User C']
        user_vectors = [
            [4.2, 3.8, 2.1, 4.9, 1.3],
            [4.1, 3.9, 2.3, 4.7, 1.1], 
            [1.2, 4.8, 4.9, 1.3, 0.8]
        ]
        
        similarity_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                similarity_matrix[i][j] = self.cosine_similarity(
                    np.array(user_vectors[i]), 
                    np.array(user_vectors[j])
                )
        
        im = ax2.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(3))
        ax2.set_yticks(range(3))
        ax2.set_xticklabels(users)
        ax2.set_yticklabels(users)
        ax2.set_title('User Similarity Matrix')
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax2.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                               ha="center", va="center", color="white")
        
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        plt.savefig('math_review_visualizations.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'math_review_visualizations.png'")
    
    def run_complete_review(self):
        """Run the complete math review session"""
        print("Starting comprehensive math review for AI/ML...")
        
        self.vector_operations_review()
        self.matrix_operations_review()
        self.calculus_review()
        self.advanced_practice()
        self.create_visualizations()
        
        print(f"\n🎉 Review Complete!")
        print(f"📊 Problems covered: {self.total_problems}")
        print("🎯 You're ready for probability and statistics!")
        print("\n💡 Key Takeaways:")
        print("• Vectors represent data points and features in AI")
        print("• Matrices transform and process information")
        print("• Derivatives help AI systems optimize and learn")
        print("• These concepts power every modern AI application")

def main():
    """Main function to run the review session"""
    review_session = MathReviewSession()
    review_session.run_complete_review()

if __name__ == "__main__":
    main()
