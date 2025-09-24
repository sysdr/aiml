"""
Day 10: Matrices and Matrix Operations
Interactive lesson demonstrating matrix fundamentals for AI/ML applications
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MatrixLessonAI:
    """
    A comprehensive matrix operations class focused on AI/ML applications
    """
    
    def __init__(self):
        print("🎯 Welcome to Day 10: Matrices and Matrix Operations")
        print("Today we're building the mathematical foundation for AI systems!\n")
    
    def demo_matrix_basics(self):
        """Demonstrate basic matrix creation and properties"""
        print("=" * 50)
        print("📊 MATRIX BASICS: Building Blocks of AI")
        print("=" * 50)
        
        # Create different types of matrices commonly used in AI
        print("Creating different matrix types used in AI systems:\n")
        
        # Data matrix (like a small dataset)
        data_matrix = np.array([
            [1.2, 3.4, 2.1],  # Sample 1: features
            [2.8, 1.9, 4.2],  # Sample 2: features  
            [3.1, 2.7, 1.8],  # Sample 3: features
            [1.9, 4.1, 3.3]   # Sample 4: features
        ])
        print(f"📋 Data Matrix (4 samples, 3 features):")
        print(f"Shape: {data_matrix.shape}")
        print(f"Data:\n{data_matrix}\n")
        
        # Weight matrix (like neural network weights)
        np.random.seed(42)  # For reproducible results
        weight_matrix = np.random.randn(3, 2) * 0.5
        print(f"🧠 Neural Network Weight Matrix:")
        print(f"Shape: {weight_matrix.shape}")
        print(f"Weights:\n{weight_matrix}\n")
        
        # Identity matrix (for transformations)
        identity = np.eye(3)
        print(f"🔄 Identity Matrix (preserves data unchanged):")
        print(f"Shape: {identity.shape}")
        print(f"Matrix:\n{identity}\n")
        
        return data_matrix, weight_matrix, identity
    
    def demo_matrix_indexing(self, matrix: np.ndarray):
        """Demonstrate matrix indexing techniques crucial for AI data manipulation"""
        print("=" * 50)
        print("🎯 MATRIX INDEXING: Accessing AI Data")
        print("=" * 50)
        
        print("Matrix we're working with:")
        print(f"{matrix}\n")
        
        # Single element access
        print(f"🔍 Single element [0,1]: {matrix[0, 1]}")
        
        # Row access (getting one data sample)
        print(f"📝 First sample (row 0): {matrix[0, :]}")
        
        # Column access (getting one feature across all samples)
        print(f"📊 Second feature (column 1): {matrix[:, 1]}")
        
        # Slice access (subset of data)
        print(f"📋 First 2 samples, last 2 features:\n{matrix[:2, 1:]}")
        
        # Boolean indexing (filtering data)
        mask = matrix > 2.5
        print(f"\n🔎 Values greater than 2.5: {matrix[mask]}")
        
        print()
    
    def demo_matrix_operations(self):
        """Demonstrate essential matrix operations for AI"""
        print("=" * 50)
        print("⚡ MATRIX OPERATIONS: The Math Behind AI")
        print("=" * 50)
        
        # Create sample matrices
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        scalar = 2
        
        print("Matrix A:")
        print(f"{A}\n")
        print("Matrix B:")
        print(f"{B}\n")
        
        # Element-wise operations
        print(f"🔢 Element-wise Addition (A + B):")
        print(f"{A + B}\n")
        
        print(f"🔢 Element-wise Subtraction (A - B):")
        print(f"{A - B}\n")
        
        print(f"📈 Scalar Multiplication (A * {scalar}):")
        print(f"{A * scalar}\n")
        
        print(f"🔄 Matrix Transpose (A.T):")
        print(f"{A.T}\n")
        
        # Statistical operations
        print(f"📊 Matrix Statistics for A:")
        print(f"   Sum: {np.sum(A)}")
        print(f"   Mean: {np.mean(A):.2f}")
        print(f"   Max: {np.max(A)}")
        print(f"   Min: {np.min(A)}")
        print(f"   Row means: {np.mean(A, axis=1)}")
        print(f"   Column means: {np.mean(A, axis=0)}\n")
    
    def build_image_filter(self):
        """Build an image brightness filter using matrix operations"""
        print("=" * 50)
        print("🖼️  BUILDING AN IMAGE FILTER: Computer Vision Basics")
        print("=" * 50)
        
        # Create a sample "image" matrix
        def create_sample_image(size=8):
            """Create a sample grayscale image with patterns"""
            image = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    # Create a gradient pattern
                    image[i, j] = (i + j) * 255 / (2 * size - 2)
            return image.astype(int)
        
        # Create and display original image
        original = create_sample_image()
        print("📸 Original 'Image' (8x8 matrix):")
        print(f"{original}\n")
        
        # Apply brightness adjustments
        bright_image = np.clip(original + 80, 0, 255)
        dark_image = np.clip(original - 80, 0, 255)
        
        print("☀️  Brightened Image (+80):")
        print(f"{bright_image}\n")
        
        print("🌙 Darkened Image (-80):")
        print(f"{dark_image}\n")
        
        # Analyze the changes
        print("📈 Analysis:")
        print(f"   Original average brightness: {np.mean(original):.1f}")
        print(f"   Brightened average: {np.mean(bright_image):.1f}")
        print(f"   Darkened average: {np.mean(dark_image):.1f}")
        
        # Show how this relates to real computer vision
        print("\n🔍 Real-world Connection:")
        print("   • Instagram filters use similar matrix operations")
        print("   • Medical imaging enhances contrast this way")
        print("   • Security cameras adjust brightness automatically")
        print("   • Your phone's auto-brightness uses these principles")
        
        return original, bright_image, dark_image
    
    def demonstrate_ai_connections(self):
        """Show how matrices connect to actual AI systems"""
        print("=" * 50)
        print("🤖 MATRICES IN REAL AI SYSTEMS")
        print("=" * 50)
        
        # Simulate different AI data representations
        print("📊 How AI Systems Use Matrices:\n")
        
        # 1. Image recognition data
        print("1. 🖼️  Computer Vision:")
        image_shape = (224, 224, 3)  # Standard CNN input
        print(f"   • Color image: {image_shape[0]}×{image_shape[1]}×{image_shape[2]} matrix")
        print(f"   • Total values: {np.prod(image_shape):,} numbers per image")
        print(f"   • Each pixel has 3 values (Red, Green, Blue)\n")
        
        # 2. Neural network weights
        print("2. 🧠 Neural Network Weights:")
        layer_sizes = [784, 128, 64, 10]
        for i in range(len(layer_sizes)-1):
            weight_shape = (layer_sizes[i], layer_sizes[i+1])
            print(f"   • Layer {i+1}: {weight_shape[0]}×{weight_shape[1]} weight matrix")
        print()
        
        # 3. Natural Language Processing
        print("3. 📝 Language Models:")
        vocab_size = 50000
        embedding_dim = 512
        print(f"   • Word embeddings: {vocab_size}×{embedding_dim} matrix")
        print(f"   • Each word becomes a {embedding_dim}-dimensional vector")
        print(f"   • ChatGPT uses matrices this large and larger!\n")
        
        # 4. Recommendation systems
        print("4. 🎬 Recommendation Systems:")
        users, items = 1000000, 50000
        print(f"   • User-item matrix: {users:,}×{items:,}")
        print(f"   • Netflix/Spotify use matrices like this")
        print(f"   • Matrix factorization finds hidden patterns\n")
        
        print("🎯 Key Insight:")
        print("   Every AI breakthrough is really a breakthrough in")
        print("   manipulating matrices more effectively!")

def main():
    """Run the complete matrix lesson"""
    lesson = MatrixLessonAI()
    
    try:
        # Run all demonstrations
        data_matrix, weights, identity = lesson.demo_matrix_basics()
        lesson.demo_matrix_indexing(data_matrix)
        lesson.demo_matrix_operations()
        lesson.build_image_filter()
        lesson.demonstrate_ai_connections()
        
        print("\n" + "=" * 50)
        print("🎉 LESSON COMPLETE!")
        print("=" * 50)
        print("✅ You now understand:")
        print("   • How to create and manipulate matrices in Python")
        print("   • Why matrices are fundamental to AI systems")
        print("   • How basic matrix operations power computer vision")
        print("   • The connection between math and real AI applications")
        print("\n🚀 Next: Tomorrow we'll explore matrix multiplication")
        print("   and see how it enables neural networks to learn!")
        
    except Exception as e:
        print(f"❌ Error during lesson: {e}")
        print("💡 Make sure you have NumPy installed: pip install numpy")

if __name__ == "__main__":
    main()
