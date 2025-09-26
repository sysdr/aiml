"""
Day 11: Matrix Multiplication and Dot Products
AI/ML Course - Implementation Code

This module demonstrates matrix multiplication and dot products
specifically for AI/ML applications.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Union

class MatrixMultiplicationLab:
    """
    A comprehensive lab for understanding matrix multiplication in AI context.
    """
    
    def __init__(self):
        """Initialize the lab with sample data."""
        self.setup_sample_data()
    
    def setup_sample_data(self):
        """Create sample datasets for demonstrations."""
        # Movie features: [Action, Comedy, Romance, Sci-Fi, Drama]
        self.movie_features = np.array([
            [0.9, 0.1, 0.2, 0.8, 0.3],  # Avengers: Endgame
            [0.1, 0.9, 0.1, 0.2, 0.4],  # Comedy Special
            [0.2, 0.3, 0.9, 0.1, 0.8],  # The Notebook
            [0.7, 0.2, 0.1, 0.9, 0.4],  # Blade Runner 2049
            [0.3, 0.4, 0.7, 0.2, 0.9],  # Manchester by the Sea
        ])
        
        self.movie_names = [
            "Avengers: Endgame",
            "Comedy Special", 
            "The Notebook",
            "Blade Runner 2049",
            "Manchester by the Sea"
        ]
        
        # User preferences for [Action, Comedy, Romance, Sci-Fi, Drama]
        self.users = {
            "Alex": np.array([0.8, 0.2, 0.6, 0.9, 0.3]),
            "Sam": np.array([0.1, 0.9, 0.8, 0.2, 0.7]),
            "Jordan": np.array([0.6, 0.4, 0.3, 0.8, 0.5])
        }
    
    def manual_dot_product(self, vector_a: List[float], vector_b: List[float]) -> float:
        """
        Calculate dot product manually to understand the mechanics.
        
        Args:
            vector_a: First vector
            vector_b: Second vector
            
        Returns:
            Dot product result
        """
        if len(vector_a) != len(vector_b):
            raise ValueError("Vectors must have the same length")
        
        result = 0
        print(f"Calculating dot product step by step:")
        print(f"Vector A: {vector_a}")
        print(f"Vector B: {vector_b}")
        print("-" * 40)
        
        for i in range(len(vector_a)):
            product = vector_a[i] * vector_b[i]
            result += product
            print(f"Step {i+1}: {vector_a[i]} × {vector_b[i]} = {product}")
        
        print(f"Sum: {result}")
        return result
    
    def manual_matrix_multiply(self, matrix_a: List[List[float]], 
                              matrix_b: List[List[float]]) -> List[List[float]]:
        """
        Multiply matrices manually to understand each step.
        
        Args:
            matrix_a: First matrix
            matrix_b: Second matrix
            
        Returns:
            Result matrix
        """
        rows_a, cols_a = len(matrix_a), len(matrix_a[0])
        rows_b, cols_b = len(matrix_b), len(matrix_b[0])
        
        if cols_a != rows_b:
            raise ValueError(f"Cannot multiply {rows_a}×{cols_a} with {rows_b}×{cols_b}")
        
        print(f"Multiplying {rows_a}×{cols_a} matrix with {rows_b}×{cols_b} matrix")
        print(f"Result will be {rows_a}×{cols_b} matrix")
        
        # Create result matrix filled with zeros
        result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
        
        # Multiply row by column using dot products
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]
        
        return result
    
    def demonstrate_similarity_detection(self):
        """Show how dot products detect similarity in AI applications."""
        print("=== Similarity Detection with Dot Products ===\n")
        
        # Example: Finding similar users based on movie preferences
        user_names = list(self.users.keys())
        similarities = {}
        
        for i, user1 in enumerate(user_names):
            for j, user2 in enumerate(user_names):
                if i < j:  # Avoid duplicate pairs
                    similarity = np.dot(self.users[user1], self.users[user2])
                    similarities[f"{user1} & {user2}"] = similarity
                    
                    print(f"Similarity between {user1} and {user2}:")
                    print(f"  {user1}: {self.users[user1]}")
                    print(f"  {user2}: {self.users[user2]}")
                    print(f"  Dot product: {similarity:.3f}")
                    print()
        
        # Find most similar pair
        most_similar = max(similarities.items(), key=lambda x: x[1])
        print(f"Most similar users: {most_similar[0]} (similarity: {most_similar[1]:.3f})")
        
        return similarities
    
    def build_recommendation_engine(self):
        """Build a movie recommendation system using matrix multiplication."""
        print("=== Movie Recommendation Engine ===\n")
        
        recommendations = {}
        
        for user_name, preferences in self.users.items():
            print(f"Recommendations for {user_name}:")
            print(f"Preferences: {preferences}")
            print()
            
            # Calculate recommendation scores using matrix multiplication
            scores = np.dot(self.movie_features, preferences)
            
            # Create ranked list
            movie_scores = list(zip(self.movie_names, scores))
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommendations[user_name] = movie_scores
            
            print("Ranked recommendations:")
            for i, (movie, score) in enumerate(movie_scores, 1):
                print(f"  {i}. {movie}: {score:.3f}")
            print()
        
        return recommendations
    
    def visualize_matrix_operations(self):
        """Create visualizations to understand matrix operations."""
        print("=== Visualizing Matrix Operations ===\n")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Matrix Multiplication in AI: Visual Understanding', fontsize=16)
        
        # 1. Movie features heatmap
        sns.heatmap(self.movie_features, 
                   annot=True, 
                   cmap='Blues',
                   xticklabels=['Action', 'Comedy', 'Romance', 'Sci-Fi', 'Drama'],
                   yticklabels=self.movie_names,
                   ax=axes[0, 0])
        axes[0, 0].set_title('Movie Features Matrix')
        
        # 2. User preferences
        user_matrix = np.array(list(self.users.values()))
        sns.heatmap(user_matrix,
                   annot=True,
                   cmap='Greens', 
                   xticklabels=['Action', 'Comedy', 'Romance', 'Sci-Fi', 'Drama'],
                   yticklabels=list(self.users.keys()),
                   ax=axes[0, 1])
        axes[0, 1].set_title('User Preferences Matrix')
        
        # 3. Recommendation scores
        scores_matrix = np.dot(self.movie_features, user_matrix.T)
        sns.heatmap(scores_matrix,
                   annot=True,
                   cmap='Oranges',
                   xticklabels=list(self.users.keys()),
                   yticklabels=self.movie_names,
                   ax=axes[1, 0])
        axes[1, 0].set_title('Recommendation Scores\n(Movies × Users)')
        
        # 4. Matrix multiplication visualization
        axes[1, 1].text(0.1, 0.8, 'Matrix Multiplication:', fontsize=14, fontweight='bold')
        axes[1, 1].text(0.1, 0.6, 'Movies (5×5) × Users (5×3) = Scores (5×3)', fontsize=12)
        axes[1, 1].text(0.1, 0.4, 'Each cell = dot product of movie features\nwith user preferences', fontsize=10)
        axes[1, 1].text(0.1, 0.2, 'Higher scores = better matches!', fontsize=12, color='red')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('matrix_operations_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'matrix_operations_visualization.png'")
    
    def neural_network_demo(self):
        """Demonstrate how matrix multiplication works in neural networks."""
        print("=== Neural Network Matrix Operations ===\n")
        
        # Simple neural network layer
        # Input: 3 features per example, 2 examples (batch)
        input_data = np.array([
            [1.0, 0.5, -0.2],  # Example 1
            [0.8, -0.3, 0.6]   # Example 2
        ])
        
        # Weights: 3 input features → 4 hidden neurons
        weights = np.array([
            [0.2, -0.1, 0.4, 0.3],
            [0.5, 0.3, -0.2, 0.1], 
            [-0.1, 0.6, 0.2, -0.4]
        ])
        
        # Bias: one for each hidden neuron
        bias = np.array([0.1, -0.05, 0.2, 0.0])
        
        print("Neural Network Forward Pass:")
        print(f"Input shape: {input_data.shape} (batch_size=2, features=3)")
        print(f"Weights shape: {weights.shape} (input_features=3, hidden_neurons=4)")
        print(f"Bias shape: {bias.shape} (hidden_neurons=4)")
        print()
        
        # Forward pass: input × weights + bias
        hidden_output = np.dot(input_data, weights) + bias
        
        print(f"Hidden layer output shape: {hidden_output.shape}")
        print("Hidden layer output:")
        print(hidden_output)
        print()
        
        # Apply activation function (ReLU)
        activated_output = np.maximum(0, hidden_output)
        print("After ReLU activation:")
        print(activated_output)
        
        return activated_output

def main():
    """Main function to run all demonstrations."""
    print("🧮 Day 11: Matrix Multiplication and Dot Products for AI")
    print("=" * 60)
    print()
    
    # Create lab instance
    lab = MatrixMultiplicationLab()
    
    # Run demonstrations
    print("1. Manual Dot Product Calculation:")
    print("-" * 40)
    sample_a = [1, 2, 3]
    sample_b = [4, 5, 6]
    result = lab.manual_dot_product(sample_a, sample_b)
    print(f"NumPy verification: {np.dot(sample_a, sample_b)}")
    print("\n")
    
    print("2. Similarity Detection:")
    print("-" * 40)
    lab.demonstrate_similarity_detection()
    print()
    
    print("3. Movie Recommendation Engine:")
    print("-" * 40)
    recommendations = lab.build_recommendation_engine()
    print()
    
    print("4. Neural Network Demo:")
    print("-" * 40)
    lab.neural_network_demo()
    print()
    
    print("5. Creating Visualizations...")
    print("-" * 40)
    lab.visualize_matrix_operations()
    
    print("\n✅ Lab complete! You've mastered matrix multiplication for AI.")
    print("💡 Key takeaway: Every AI prediction involves matrix multiplication!")

if __name__ == "__main__":
    main()
