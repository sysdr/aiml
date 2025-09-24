#!/bin/bash

# Day 10: Matrices and Matrix Operations - Implementation Package Generator
# This script creates all necessary files for the lesson

echo "🚀 Generating Day 10: Matrices and Matrix Operations lesson files..."

# Create the setup script
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 10: Matrices and Matrix Operations environment..."

# Check if Python 3.11+ is available
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 11 ]); then
    echo "❌ Python 3.11+ required. Current version: $python_version"
    echo "Please install Python 3.11 or higher"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete! To get started:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run the lesson: python lesson_code.py"
echo "3. Run tests: python test_lesson.py"
echo "4. Open Jupyter notebook: jupyter notebook matrices_lesson.ipynb"
EOF

# Create the main lesson implementation
cat > lesson_code.py << 'EOF'
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
EOF

# Create test file
cat > test_lesson.py << 'EOF'
"""
Test suite for Day 10: Matrices and Matrix Operations
Verifies that students understand key concepts
"""

import numpy as np
import unittest
from io import StringIO
import sys

class TestMatrixOperations(unittest.TestCase):
    """Test basic matrix operations understanding"""
    
    def setUp(self):
        """Set up test matrices"""
        self.matrix_a = np.array([[1, 2, 3], [4, 5, 6]])
        self.matrix_b = np.array([[7, 8, 9], [10, 11, 12]])
        self.small_matrix = np.array([[1, 2], [3, 4]])
    
    def test_matrix_creation(self):
        """Test that students can create matrices correctly"""
        # Test zeros matrix
        zeros = np.zeros((3, 4))
        self.assertEqual(zeros.shape, (3, 4))
        self.assertTrue(np.all(zeros == 0))
        
        # Test identity matrix
        identity = np.eye(3)
        self.assertEqual(identity.shape, (3, 3))
        self.assertEqual(identity[0, 0], 1)
        self.assertEqual(identity[0, 1], 0)
    
    def test_matrix_indexing(self):
        """Test matrix indexing operations"""
        # Test single element access
        self.assertEqual(self.matrix_a[0, 1], 2)
        self.assertEqual(self.matrix_a[1, 2], 6)
        
        # Test row access
        np.testing.assert_array_equal(self.matrix_a[0, :], [1, 2, 3])
        
        # Test column access
        np.testing.assert_array_equal(self.matrix_a[:, 1], [2, 5])
    
    def test_matrix_operations(self):
        """Test basic matrix operations"""
        # Test addition
        result = self.matrix_a + self.matrix_b
        expected = np.array([[8, 10, 12], [14, 16, 18]])
        np.testing.assert_array_equal(result, expected)
        
        # Test scalar multiplication
        result = self.small_matrix * 2
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_array_equal(result, expected)
        
        # Test transpose
        result = self.small_matrix.T
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(result, expected)
    
    def test_image_filter_simulation(self):
        """Test image filtering operations"""
        # Create simple test image
        image = np.array([[100, 150], [50, 200]])
        
        # Test brightness adjustment
        bright = np.clip(image + 50, 0, 255)
        dark = np.clip(image - 50, 0, 255)
        
        # Verify brightness changes
        self.assertGreater(np.mean(bright), np.mean(image))
        self.assertLess(np.mean(dark), np.mean(image))
        
        # Verify clipping works
        self.assertTrue(np.all(bright >= 0))
        self.assertTrue(np.all(bright <= 255))
    
    def test_matrix_properties(self):
        """Test understanding of matrix properties"""
        # Test shape
        self.assertEqual(self.matrix_a.shape, (2, 3))
        
        # Test size
        self.assertEqual(self.matrix_a.size, 6)
        
        # Test statistical operations
        self.assertEqual(np.sum(self.small_matrix), 10)
        self.assertEqual(np.mean(self.small_matrix), 2.5)

class TestAIConnections(unittest.TestCase):
    """Test understanding of AI connections"""
    
    def test_data_representation(self):
        """Test understanding of how AI represents data as matrices"""
        # Simulate image data
        image_height, image_width, channels = 224, 224, 3
        image_size = image_height * image_width * channels
        
        # Verify understanding of data sizes
        self.assertEqual(image_size, 150528)
        
        # Simulate neural network layer
        input_size, output_size = 784, 128
        weight_matrix_size = input_size * output_size
        self.assertEqual(weight_matrix_size, 100352)
    
    def test_matrix_shapes_compatibility(self):
        """Test understanding of matrix shape requirements"""
        # These shapes should be compatible for operations
        a = np.random.randn(3, 4)
        b = np.random.randn(3, 4)
        
        # Element-wise operations require same shape
        result = a + b
        self.assertEqual(result.shape, (3, 4))
        
        # Transpose changes shape
        a_t = a.T
        self.assertEqual(a_t.shape, (4, 3))

def run_student_challenges():
    """Interactive challenges for students"""
    print("🎯 STUDENT CHALLENGES")
    print("=" * 40)
    
    print("\nChallenge 1: Create a 5x5 identity matrix")
    identity_5x5 = np.eye(5)
    print("✅ Solution shape:", identity_5x5.shape)
    
    print("\nChallenge 2: Create a 3x3 matrix of random values")
    random_matrix = np.random.rand(3, 3)
    print("✅ Created random matrix with shape:", random_matrix.shape)
    
    print("\nChallenge 3: Extract the diagonal from a matrix")
    test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    diagonal = np.diag(test_matrix)
    print("✅ Diagonal elements:", diagonal)
    
    print("\nChallenge 4: Calculate row and column means")
    row_means = np.mean(test_matrix, axis=1)
    col_means = np.mean(test_matrix, axis=0)
    print("✅ Row means:", row_means)
    print("✅ Column means:", col_means)
    
    print("\n🎉 All challenges completed successfully!")

if __name__ == "__main__":
    print("🧪 Running Day 10 Matrix Operations Tests...\n")
    
    # Capture test output
    test_output = StringIO()
    runner = unittest.TextTestRunner(stream=test_output, verbosity=2)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMatrixOperations))
    suite.addTest(unittest.makeSuite(TestAIConnections))
    
    # Run tests
    result = runner.run(suite)
    
    # Display results
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print("\n" + "=" * 50)
    run_student_challenges()
EOF

# Create requirements file
cat > requirements.txt << 'EOF'
# Day 10: Matrices and Matrix Operations - Dependencies
# Core scientific computing
numpy>=1.24.0
matplotlib>=3.6.0

# Interactive learning
jupyter>=1.0.0
ipywidgets>=8.0.0

# Development and testing
pytest>=7.0.0

# Optional: For enhanced visualizations
seaborn>=0.12.0
plotly>=5.17.0
EOF

# Create README file
cat > README.md << 'EOF'
# Day 10: Matrices and Matrix Operations

Welcome to Day 10 of your AI/ML journey! Today we explore matrices - the fundamental data structures that power all modern AI systems.

## 🎯 Learning Objectives

By the end of this lesson, you'll be able to:
- Create and manipulate matrices using NumPy
- Understand how matrices represent data in AI systems
- Perform basic matrix operations essential for machine learning
- Build a simple image filter using matrix operations
- Connect matrix mathematics to real-world AI applications

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   source venv/bin/activate
   ```

2. **Run the Interactive Lesson**
   ```bash
   python lesson_code.py
   ```

3. **Test Your Understanding**
   ```bash
   python test_lesson.py
   ```

4. **Optional: Jupyter Notebook**
   ```bash
   jupyter notebook
   # Open matrices_lesson.ipynb
   ```

## 📚 What You'll Learn

### Core Concepts
- **Matrix Fundamentals**: Understanding matrices as AI data structures
- **Matrix Creation**: Different ways to create matrices for AI applications
- **Indexing & Slicing**: Accessing and modifying matrix data
- **Basic Operations**: Addition, multiplication, transpose, and statistics
- **AI Applications**: How matrices power computer vision and neural networks

### Hands-On Project
Build an image brightness filter that demonstrates:
- Matrix representation of images
- Element-wise operations for image processing
- Real-world applications in computer vision

## 🔗 Connections to AI

This lesson shows how matrices appear in:
- **Computer Vision**: Images as pixel matrices
- **Neural Networks**: Weights and activations as matrices
- **Natural Language Processing**: Word embeddings and attention matrices
- **Recommendation Systems**: User-item interaction matrices

## 📁 File Structure

```
day10-matrices/
├── setup.sh              # Environment setup script
├── lesson_code.py         # Main interactive lesson
├── test_lesson.py         # Verification tests
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── venv/                 # Virtual environment (created by setup)
```

## 🛠️ Technical Requirements

- Python 3.11+
- NumPy for matrix operations
- Matplotlib for visualizations
- Jupyter (optional) for interactive exploration

## 💡 Key Insights

1. **Matrices aren't just math** - they're the language AI systems use to represent and process information
2. **Every AI operation** involves matrix manipulations at some level
3. **Simple operations** like addition and scalar multiplication power complex AI behaviors
4. **Understanding matrices** is essential for debugging and improving AI systems

## 🎯 Success Criteria

You've mastered this lesson when you can:
- [ ] Create matrices of different types and sizes
- [ ] Access specific elements, rows, and columns confidently
- [ ] Perform basic matrix operations without errors
- [ ] Explain how a simple image filter works using matrices
- [ ] Connect matrix operations to at least 3 AI applications

## 🚀 Next Steps

Tomorrow (Day 11) we'll explore:
- Matrix multiplication and dot products
- How matrix multiplication enables neural network learning
- Building your first simple neural network layer
- Performance considerations for large matrix operations

## 🆘 Getting Help

If you encounter issues:
1. Check that your Python version is 3.11+
2. Ensure all dependencies are installed correctly
3. Run the test suite to verify your understanding
4. Review the lesson code for examples and explanations

## 🏆 Challenge Yourself

After completing the basic lesson, try these extensions:
1. Create a contrast adjustment filter (multiply by factors > 1)
2. Implement a simple blur effect using matrix averaging
3. Experiment with different matrix shapes and operations
4. Research how your favorite AI application uses matrices

Remember: Every AI expert started with understanding these fundamentals. You're building the mathematical foundation that will support your entire AI journey!
EOF

# Create Jupyter notebook for interactive learning
cat > matrices_lesson.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Day 10: Matrices and Matrix Operations\n",
    "\n",
    "Welcome to the interactive version of today's lesson! This notebook lets you experiment with matrices in a hands-on way.\n",
    "\n",
    "## 🎯 Today's Goal\n",
    "Understand how matrices power AI systems and build your first computer vision filter."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import necessary libraries\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Set up nice plotting\n",
    "plt.style.use('seaborn-v0_8')\n",
    "np.random.seed(42)\n",
    "\n",
    "print(\"🚀 Ready to explore matrices for AI!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Creating Matrices for AI Applications\n",
    "\n",
    "Let's start by creating different types of matrices commonly used in AI systems."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create different matrix types\n",
    "data_matrix = np.array([\n",
    "    [1.2, 3.4, 2.1],  # Sample 1\n",
    "    [2.8, 1.9, 4.2],  # Sample 2\n",
    "    [3.1, 2.7, 1.8],  # Sample 3\n",
    "    [1.9, 4.1, 3.3]   # Sample 4\n",
    "])\n",
    "\n",
    "print(\"📊 Data Matrix (like a small dataset):\")\n",
    "print(f\"Shape: {data_matrix.shape}\")\n",
    "print(data_matrix)\n",
    "\n",
    "# Neural network weights\n",
    "weights = np.random.randn(3, 2) * 0.5\n",
    "print(\"\\n🧠 Neural Network Weights:\")\n",
    "print(f\"Shape: {weights.shape}\")\n",
    "print(weights)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Matrix Indexing and Slicing\n",
    "\n",
    "Practice accessing different parts of matrices - crucial for AI data manipulation!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Practice matrix indexing\n",
    "print(\"🎯 Matrix Indexing Practice:\")\n",
    "print(f\"Single element [0,1]: {data_matrix[0, 1]}\")\n",
    "print(f\"First row (sample): {data_matrix[0, :]}\")\n",
    "print(f\"Second column (feature): {data_matrix[:, 1]}\")\n",
    "print(f\"Subset [0:2, 1:3]:\\n{data_matrix[0:2, 1:3]}\")\n",
    "\n",
    "# Try your own indexing here!\n",
    "# TODO: Get the last row of the matrix\n",
    "# TODO: Get the first and last columns\n",
    "# TODO: Get all values greater than 2.5"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Building an Image Filter\n",
    "\n",
    "Now let's create our computer vision project - an image brightness filter!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a sample \"image\"\n",
    "def create_sample_image(size=8):\n",
    "    image = np.zeros((size, size))\n",
    "    for i in range(size):\n",
    "        for j in range(size):\n",
    "            image[i, j] = (i + j) * 255 / (2 * size - 2)\n",
    "    return image.astype(int)\n",
    "\n",
    "# Create and visualize images\n",
    "original = create_sample_image()\n",
    "bright = np.clip(original + 80, 0, 255)\n",
    "dark = np.clip(original - 80, 0, 255)\n",
    "\n",
    "# Plot the results\n",
    "fig, axes = plt.subplots(1, 3, figsize=(12, 4))\n",
    "\n",
    "axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)\n",
    "axes[0].set_title('Original Image')\n",
    "axes[0].axis('off')\n",
    "\n",
    "axes[1].imshow(bright, cmap='gray', vmin=0, vmax=255)\n",
    "axes[1].set_title('Brightened (+80)')\n",
    "axes[1].axis('off')\n",
    "\n",
    "axes[2].imshow(dark, cmap='gray', vmin=0, vmax=255)\n",
    "axes[2].set_title('Darkened (-80)')\n",
    "axes[2].axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f\"📊 Brightness Analysis:\")\n",
    "print(f\"Original average: {np.mean(original):.1f}\")\n",
    "print(f\"Brightened average: {np.mean(bright):.1f}\")\n",
    "print(f\"Darkened average: {np.mean(dark):.1f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Experiment Zone\n",
    "\n",
    "Try creating your own matrix operations and filters!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Experiment with different operations\n",
    "A = np.array([[1, 2], [3, 4]])\n",
    "B = np.array([[5, 6], [7, 8]])\n",
    "\n",
    "print(\"🧪 Experiment with these matrices:\")\n",
    "print(f\"A = \\n{A}\")\n",
    "print(f\"B = \\n{B}\")\n",
    "\n",
    "# Try these operations:\n",
    "print(f\"\\nA + B = \\n{A + B}\")\n",
    "print(f\"\\nA * 3 = \\n{A * 3}\")\n",
    "print(f\"\\nA.T = \\n{A.T}\")\n",
    "\n",
    "# TODO: Try your own operations!\n",
    "# What happens if you multiply A and B element-wise?\n",
    "# Can you create a contrast filter (multiply by 1.5)?\n",
    "# What about combining brightness and contrast adjustments?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. AI Connections Visualization\n",
    "\n",
    "Let's visualize how matrices appear in different AI applications."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize different AI matrix uses\n",
    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
    "\n",
    "# 1. Image data\n",
    "rgb_image = np.random.rand(32, 32, 3)\n",
    "axes[0, 0].imshow(rgb_image)\n",
    "axes[0, 0].set_title('Computer Vision\\n32×32×3 RGB Image Matrix')\n",
    "axes[0, 0].axis('off')\n",
    "\n",
    "# 2. Neural network weights\n",
    "weights_viz = np.random.randn(20, 20)\n",
    "sns.heatmap(weights_viz, ax=axes[0, 1], cmap='coolwarm', center=0, cbar=False)\n",
    "axes[0, 1].set_title('Neural Network\\n20×20 Weight Matrix')\n",
    "\n",
    "# 3. Word embeddings\n",
    "embeddings = np.random.randn(50, 10)\n",
    "sns.heatmap(embeddings, ax=axes[1, 0], cmap='viridis', cbar=False)\n",
    "axes[1, 0].set_title('Word Embeddings\\n50 words × 10 dimensions')\n",
    "\n",
    "# 4. User-item matrix\n",
    "user_item = np.random.choice([0, 1, 2, 3, 4, 5], size=(20, 15), p=[0.7, 0.1, 0.1, 0.05, 0.03, 0.02])\n",
    "sns.heatmap(user_item, ax=axes[1, 1], cmap='Blues', cbar=False)\n",
    "axes[1, 1].set_title('Recommendation System\\n20 users × 15 items')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"🤖 These are all matrices used in real AI systems!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎉 Congratulations!\n",
    "\n",
    "You've completed Day 10! You now understand:\n",
    "- How to create and manipulate matrices in Python\n",
    "- Why matrices are fundamental to AI systems\n",
    "- How to build basic computer vision filters\n",
    "- The connection between mathematics and real AI applications\n",
    "\n",
    "### 🚀 Next Steps\n",
    "Tomorrow we'll explore matrix multiplication - the operation that makes neural networks possible!\n",
    "\n",
    "### 💡 Challenge\n",
    "Try modifying the image filter to create:\n",
    "1. A contrast adjustment (multiply by a factor)\n",
    "2. A threshold effect (values above X become 255, below become 0)\n",
    "3. A combination filter (brightness + contrast)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Set executable permissions
chmod +x setup.sh

echo ""
echo "✅ All files generated successfully!"
echo ""
echo "📁 Generated files:"
echo "   • setup.sh - Environment setup script"
echo "   • lesson_code.py - Main interactive lesson"
echo "   • test_lesson.py - Understanding verification tests"
echo "   • requirements.txt - Python dependencies"
echo "   • README.md - Complete setup and learning guide"
echo "   • matrices_lesson.ipynb - Interactive Jupyter notebook"
echo ""
echo "🚀 To get started:"
echo "   1. chmod +x setup.sh && ./setup.sh"
echo "   2. source venv/bin/activate"
echo "   3. python lesson_code.py"
echo ""
echo "🎯 This lesson will take 2-3 hours and teach you the matrix"
echo "   foundations that power all modern AI systems!"
