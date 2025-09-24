#!/bin/bash

# Day 8: Introduction to Linear Algebra - File Generator
# Creates all necessary files for the lesson

echo "Creating Day 8 Linear Algebra lesson files..."

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.24.3
matplotlib==3.7.2
jupyter==1.0.0
ipython==8.14.0
EOF

# Create setup.sh
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Day 8: Introduction to Linear Algebra environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python $python_version detected (meets requirement: $required_version+)"
else
    echo "✗ Python $required_version+ required. Current version: $python_version"
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv ai_course_env

# Activate virtual environment
echo "Activating virtual environment..."
source ai_course_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Setup complete!"
echo ""
echo "To activate the environment later, run:"
echo "source ai_course_env/bin/activate"
echo ""
echo "Then run the lesson with:"
echo "python lesson_code.py"
EOF

# Make setup.sh executable
chmod +x setup.sh

# Create lesson_code.py
cat > lesson_code.py << 'EOF'
"""
Day 8: Introduction to Linear Algebra for AI
Interactive lesson demonstrating core linear algebra concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def demo_vectors():
    """Demonstrate vector concepts and operations"""
    print("=" * 50)
    print("DEMO 1: Understanding Vectors")
    print("=" * 50)
    
    # Customer preference vectors
    print("Representing customer preferences as vectors:")
    print("Vector format: [electronics, books, clothing, sports]")
    
    customer_alice = np.array([0.9, 0.2, 0.1, 0.8])
    customer_bob = np.array([0.1, 0.9, 0.7, 0.2])
    customer_charlie = np.array([0.6, 0.6, 0.4, 0.6])
    
    print(f"Alice:   {customer_alice}")
    print(f"Bob:     {customer_bob}")
    print(f"Charlie: {customer_charlie}")
    
    # Vector operations
    print("\nVector Operations:")
    
    # Addition - combining preferences
    combined_prefs = customer_alice + customer_bob
    print(f"Alice + Bob preferences: {combined_prefs}")
    
    # Scalar multiplication - amplifying preferences
    amplified_alice = 2 * customer_alice
    print(f"Amplified Alice (2x): {amplified_alice}")
    
    # Magnitude (length) of vector
    alice_magnitude = np.linalg.norm(customer_alice)
    print(f"Alice's preference strength: {alice_magnitude:.3f}")
    
    return customer_alice, customer_bob, customer_charlie

def demo_similarity():
    """Demonstrate similarity calculations using dot products"""
    print("\n" + "=" * 50)
    print("DEMO 2: Measuring Similarity")
    print("=" * 50)
    
    # Create some user preference vectors
    users = {
        "Alice": np.array([0.9, 0.2, 0.1, 0.8]),
        "Bob": np.array([0.1, 0.9, 0.7, 0.2]),
        "Charlie": np.array([0.6, 0.6, 0.4, 0.6]),
        "Diana": np.array([0.8, 0.3, 0.2, 0.7])
    }
    
    print("Calculating user similarities using dot products:")
    print("(Higher values = more similar preferences)")
    
    user_names = list(users.keys())
    similarities = {}
    
    for i, user1 in enumerate(user_names):
        for j, user2 in enumerate(user_names):
            if i < j:  # Avoid duplicates
                similarity = np.dot(users[user1], users[user2])
                similarities[f"{user1}-{user2}"] = similarity
                print(f"{user1} ↔ {user2}: {similarity:.3f}")
    
    # Find most similar pair
    most_similar = max(similarities.items(), key=lambda x: x[1])
    print(f"\nMost similar users: {most_similar[0]} (score: {most_similar[1]:.3f})")
    
    return similarities

def demo_matrices():
    """Demonstrate matrix operations and transformations"""
    print("\n" + "=" * 50)
    print("DEMO 3: Working with Matrices")
    print("=" * 50)
    
    # User-item rating matrix
    print("User-Item Rating Matrix:")
    print("Rows: Users, Columns: [Electronics, Books, Clothing, Sports]")
    
    ratings = np.array([
        [5, 2, 1, 5],  # Alice
        [1, 5, 4, 2],  # Bob  
        [3, 4, 4, 3],  # Charlie
        [5, 3, 2, 4]   # Diana
    ])
    
    users = ["Alice", "Bob", "Charlie", "Diana"]
    categories = ["Electronics", "Books", "Clothing", "Sports"]
    
    print("\nRatings Matrix:")
    print("     ", "  ".join(f"{cat[:4]:>4}" for cat in categories))
    for i, user in enumerate(users):
        print(f"{user:8} {ratings[i]}")
    
    # Matrix operations
    print("\nMatrix Operations:")
    
    # Average ratings per category
    avg_ratings = np.mean(ratings, axis=0)
    print("Average ratings per category:")
    for i, cat in enumerate(categories):
        print(f"  {cat}: {avg_ratings[i]:.2f}")
    
    # User preference strength (row sums)
    user_activity = np.sum(ratings, axis=1)
    print("\nTotal user activity (sum of ratings):")
    for i, user in enumerate(users):
        print(f"  {user}: {user_activity[i]}")
    
    # Matrix transformation example
    print("\nApplying recommendation weights...")
    # Transform ratings to recommendation scores
    rec_weights = np.array([
        [1.2, 0.8, 0.5],  # Electronics -> [Premium, Standard, Budget]
        [0.3, 1.1, 0.9],  # Books -> [Premium, Standard, Budget] 
        [0.2, 0.9, 1.2],  # Clothing -> [Premium, Standard, Budget]
        [1.0, 0.7, 0.4]   # Sports -> [Premium, Standard, Budget]
    ])
    
    recommendations = np.dot(ratings, rec_weights)
    rec_categories = ["Premium", "Standard", "Budget"]
    
    print("Generated recommendation scores:")
    print("     ", "  ".join(f"{cat:>8}" for cat in rec_categories))
    for i, user in enumerate(users):
        scores_str = "  ".join(f"{score:8.2f}" for score in recommendations[i])
        print(f"{user:8} {scores_str}")
    
    return ratings, recommendations

class SimpleRecommender:
    """A basic recommendation system using linear algebra"""
    
    def __init__(self):
        self.user_profiles = {}
        self.item_features = {}
        self.interaction_history = []
    
    def add_user(self, user_id: str, preferences: List[float]):
        """Add a user with their preference vector"""
        self.user_profiles[user_id] = np.array(preferences)
        print(f"Added user '{user_id}' with preferences: {preferences}")
    
    def add_item(self, item_id: str, features: List[float]):
        """Add an item with its feature vector"""
        self.item_features[item_id] = np.array(features)
        print(f"Added item '{item_id}' with features: {features}")
    
    def calculate_score(self, user_id: str, item_id: str) -> float:
        """Calculate preference score using dot product"""
        if user_id not in self.user_profiles or item_id not in self.item_features:
            return 0.0
        
        user_prefs = self.user_profiles[user_id]
        item_features = self.item_features[item_id]
        
        # Use dot product to calculate compatibility
        score = np.dot(user_prefs, item_features)
        return score
    
    def recommend(self, user_id: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """Generate top-N recommendations for a user"""
        if user_id not in self.user_profiles:
            return []
        
        scores = []
        for item_id in self.item_features:
            score = self.calculate_score(user_id, item_id)
            scores.append((item_id, score))
        
        # Sort by score (descending) and return top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    def explain_recommendation(self, user_id: str, item_id: str):
        """Explain why an item was recommended"""
        if user_id not in self.user_profiles or item_id not in self.item_features:
            print("User or item not found.")
            return
        
        user_prefs = self.user_profiles[user_id]
        item_features = self.item_features[item_id]
        
        print(f"\nRecommendation Explanation for {user_id} → {item_id}:")
        print(f"User preferences: {user_prefs}")
        print(f"Item features:    {item_features}")
        
        # Component-wise multiplication shows how each feature contributes
        contributions = user_prefs * item_features
        print(f"Contributions:    {contributions}")
        print(f"Total score:      {np.sum(contributions):.3f}")

def demo_recommendation_system():
    """Demonstrate the recommendation system in action"""
    print("\n" + "=" * 50)
    print("DEMO 4: AI Recommendation System")
    print("=" * 50)
    
    # Create recommender
    recommender = SimpleRecommender()
    
    print("Setting up recommendation system...")
    print("Feature dimensions: [Tech, Books, Entertainment, Sports]")
    
    # Add users
    recommender.add_user("tech_lover", [0.9, 0.2, 0.3, 0.1])
    recommender.add_user("bookworm", [0.1, 0.9, 0.4, 0.2])
    recommender.add_user("athlete", [0.2, 0.1, 0.3, 0.9])
    recommender.add_user("balanced", [0.5, 0.5, 0.5, 0.5])
    
    # Add items
    recommender.add_item("laptop", [0.95, 0.1, 0.2, 0.0])
    recommender.add_item("sci_fi_novel", [0.3, 0.9, 0.6, 0.0])
    recommender.add_item("action_movie", [0.2, 0.1, 0.9, 0.3])
    recommender.add_item("tennis_racket", [0.1, 0.0, 0.2, 0.95])
    recommender.add_item("programming_book", [0.8, 0.8, 0.2, 0.0])
    recommender.add_item("fitness_tracker", [0.6, 0.1, 0.3, 0.8])
    
    # Generate recommendations
    print("\n" + "-" * 30)
    print("RECOMMENDATIONS")
    print("-" * 30)
    
    for user_id in recommender.user_profiles:
        recommendations = recommender.recommend(user_id, top_n=3)
        print(f"\nTop recommendations for {user_id}:")
        for i, (item_id, score) in enumerate(recommendations, 1):
            print(f"  {i}. {item_id} (score: {score:.3f})")
    
    # Detailed explanation for one recommendation
    print("\n" + "-" * 30)
    print("RECOMMENDATION EXPLANATION")
    print("-" * 30)
    recommender.explain_recommendation("tech_lover", "laptop")
    
    return recommender

def visualize_vectors():
    """Create a simple 2D visualization of vectors"""
    print("\n" + "=" * 50)
    print("DEMO 5: Vector Visualization")
    print("=" * 50)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Demo 1: Vector addition
    ax1.set_title("Vector Addition")
    ax1.set_xlim(-1, 6)
    ax1.set_ylim(-1, 5)
    ax1.grid(True, alpha=0.3)
    
    # Define vectors
    v1 = np.array([2, 3])
    v2 = np.array([3, 1])
    v_sum = v1 + v2
    
    # Plot vectors
    ax1.arrow(0, 0, v1[0], v1[1], head_width=0.15, head_length=0.2, fc='blue', ec='blue', label='Vector A')
    ax1.arrow(0, 0, v2[0], v2[1], head_width=0.15, head_length=0.2, fc='red', ec='red', label='Vector B')
    ax1.arrow(0, 0, v_sum[0], v_sum[1], head_width=0.15, head_length=0.2, fc='green', ec='green', label='A + B', linewidth=2)
    
    # Show addition visually
    ax1.arrow(v1[0], v1[1], v2[0], v2[1], head_width=0.1, head_length=0.15, fc='red', ec='red', alpha=0.5, linestyle='--')
    
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Demo 2: User preferences in 2D space
    ax2.set_title("User Preferences (Tech vs Books)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # User preference data (tech_preference, book_preference)
    users_2d = {
        'Alice': [0.9, 0.2],
        'Bob': [0.1, 0.9],
        'Charlie': [0.6, 0.6],
        'Diana': [0.8, 0.3]
    }
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, (user, prefs) in enumerate(users_2d.items()):
        ax2.scatter(prefs[0], prefs[1], c=colors[i], s=100, label=user)
        ax2.annotate(user, (prefs[0], prefs[1]), xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Tech Preference')
    ax2.set_ylabel('Book Preference')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('linear_algebra_demo.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'linear_algebra_demo.png'")
    plt.show()

def main():
    """Run all demonstrations"""
    print("🚀 Day 8: Introduction to Linear Algebra for AI")
    print("Welcome to the mathematical foundation of AI systems!")
    
    try:
        # Run all demos
        demo_vectors()
        demo_similarity()
        demo_matrices()
        demo_recommendation_system()
        
        # Create visualization
        try:
            visualize_vectors()
        except Exception as e:
            print(f"Visualization skipped (matplotlib issue): {e}")
        
        print("\n" + "=" * 50)
        print("🎉 LESSON COMPLETE!")
        print("=" * 50)
        print("You've learned:")
        print("✓ How to represent data as vectors and matrices")
        print("✓ How to calculate similarity using dot products")
        print("✓ How to transform data using matrix operations")
        print("✓ How to build a simple AI recommendation system")
        print("\nNext: Tomorrow we'll dive deeper into vector operations and spaces!")
        
    except Exception as e:
        print(f"Error during lesson: {e}")
        print("Please check your environment setup and try again.")

if __name__ == "__main__":
    main()
EOF

# Create test_lesson.py
cat > test_lesson.py << 'EOF'
"""
Day 8: Test Suite for Linear Algebra Concepts
Verify understanding of core concepts
"""

import numpy as np
import sys
from lesson_code import SimpleRecommender

def test_vector_operations():
    """Test basic vector operations"""
    print("Testing vector operations...")
    
    # Test vector creation
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    # Test addition
    v_sum = v1 + v2
    expected_sum = np.array([5, 7, 9])
    assert np.array_equal(v_sum, expected_sum), "Vector addition failed"
    
    # Test scalar multiplication
    v_scaled = 2 * v1
    expected_scaled = np.array([2, 4, 6])
    assert np.array_equal(v_scaled, expected_scaled), "Scalar multiplication failed"
    
    # Test dot product
    dot_product = np.dot(v1, v2)
    expected_dot = 32  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert dot_product == expected_dot, f"Dot product failed: got {dot_product}, expected {expected_dot}"
    
    print("✓ Vector operations test passed")

def test_matrix_operations():
    """Test basic matrix operations"""
    print("Testing matrix operations...")
    
    # Test matrix creation
    matrix = np.array([[1, 2], [3, 4]])
    
    # Test matrix multiplication
    vector = np.array([1, 2])
    result = np.dot(matrix, vector)
    expected = np.array([5, 11])  # [1*1+2*2, 3*1+4*2] = [5, 11]
    assert np.array_equal(result, expected), "Matrix-vector multiplication failed"
    
    # Test matrix properties
    assert matrix.shape == (2, 2), "Matrix shape incorrect"
    
    print("✓ Matrix operations test passed")

def test_similarity_calculation():
    """Test similarity calculations"""
    print("Testing similarity calculations...")
    
    # Test similarity between identical vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    similarity = np.dot(v1, v2)
    assert similarity == 1.0, "Identical vector similarity should be 1.0"
    
    # Test similarity between orthogonal vectors
    v3 = np.array([1, 0])
    v4 = np.array([0, 1])
    similarity = np.dot(v3, v4)
    assert similarity == 0.0, "Orthogonal vector similarity should be 0.0"
    
    print("✓ Similarity calculation test passed")

def test_recommender_system():
    """Test the recommendation system"""
    print("Testing recommendation system...")
    
    recommender = SimpleRecommender()
    
    # Add test data
    recommender.add_user("test_user", [1.0, 0.0, 0.0])
    recommender.add_item("perfect_match", [1.0, 0.0, 0.0])
    recommender.add_item("no_match", [0.0, 1.0, 1.0])
    
    # Test score calculation
    perfect_score = recommender.calculate_score("test_user", "perfect_match")
    no_match_score = recommender.calculate_score("test_user", "no_match")
    
    assert perfect_score > no_match_score, "Perfect match should score higher than no match"
    assert perfect_score == 1.0, f"Perfect match score should be 1.0, got {perfect_score}"
    assert no_match_score == 0.0, f"No match score should be 0.0, got {no_match_score}"
    
    # Test recommendations
    recommendations = recommender.recommend("test_user", top_n=2)
    assert len(recommendations) == 2, "Should return 2 recommendations"
    assert recommendations[0][0] == "perfect_match", "Perfect match should be first recommendation"
    
    print("✓ Recommendation system test passed")

def test_numpy_environment():
    """Test that numpy is properly installed and working"""
    print("Testing NumPy environment...")
    
    # Test numpy version
    numpy_version = np.__version__
    print(f"NumPy version: {numpy_version}")
    
    # Test basic numpy functionality
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0, "NumPy mean calculation failed"
    assert arr.std() > 0, "NumPy standard deviation calculation failed"
    
    # Test linear algebra functions
    matrix = np.random.rand(3, 3)
    determinant = np.linalg.det(matrix)
    assert isinstance(determinant, (int, float)), "Determinant calculation failed"
    
    print("✓ NumPy environment test passed")

def run_comprehension_quiz():
    """Interactive quiz to test understanding"""
    print("\n" + "=" * 50)
    print("COMPREHENSION QUIZ")
    print("=" * 50)
    
    score = 0
    total_questions = 4
    
    # Question 1
    print("\nQuestion 1: What does a vector represent in AI systems?")
    print("A) A direction in space")
    print("B) A list of numbers representing data features")
    print("C) A programming function")
    print("D) A type of database")
    
    answer = input("Your answer (A/B/C/D): ").upper().strip()
    if answer == 'B':
        print("✓ Correct! Vectors encode data features as numbers.")
        score += 1
    else:
        print("✗ Incorrect. Vectors represent data features as lists of numbers.")
    
    # Question 2
    print("\nQuestion 2: What does the dot product measure?")
    print("A) The length of a vector")
    print("B) The angle between vectors")
    print("C) The similarity between vectors")
    print("D) The dimension of a matrix")
    
    answer = input("Your answer (A/B/C/D): ").upper().strip()
    if answer == 'C':
        print("✓ Correct! Dot product measures similarity between vectors.")
        score += 1
    else:
        print("✗ Incorrect. Dot product measures similarity between vectors.")
    
    # Question 3
    print("\nQuestion 3: In a recommendation system, higher dot product scores mean:")
    print("A) Less compatibility")
    print("B) More compatibility")
    print("C) Random compatibility")
    print("D) No relationship")
    
    answer = input("Your answer (A/B/C/D): ").upper().strip()
    if answer == 'B':
        print("✓ Correct! Higher scores indicate better matches.")
        score += 1
    else:
        print("✗ Incorrect. Higher dot product scores mean more compatibility.")
    
    # Question 4
    print("\nQuestion 4: Matrix multiplication is used in AI to:")
    print("A) Store data only")
    print("B) Transform data between representations")
    print("C) Delete unnecessary data")
    print("D) Compress file sizes")
    
    answer = input("Your answer (A/B/C/D): ").upper().strip()
    if answer == 'B':
        print("✓ Correct! Matrix multiplication transforms data representations.")
        score += 1
    else:
        print("✗ Incorrect. Matrix multiplication transforms data between representations.")
    
    # Show results
    print(f"\nQuiz Results: {score}/{total_questions} ({(score/total_questions)*100:.0f}%)")
    
    if score == total_questions:
        print("🎉 Perfect score! You understand the fundamentals!")
    elif score >= total_questions * 0.75:
        print("👍 Great job! You have a solid understanding.")
    elif score >= total_questions * 0.5:
        print("📚 Good start! Review the concepts and try again.")
    else:
        print("📖 Keep studying! Review the lesson materials.")
    
    return score

def main():
    """Run all tests"""
    print("🧪 Day 8: Linear Algebra Test Suite")
    print("=" * 50)
    
    try:
        # Run technical tests
        test_numpy_environment()
        test_vector_operations()
        test_matrix_operations()
        test_similarity_calculation()
        test_recommender_system()
        
        print("\n🎉 All technical tests passed!")
        
        # Run comprehension quiz
        quiz_score = run_comprehension_quiz()
        
        # Overall assessment
        print("\n" + "=" * 50)
        print("ASSESSMENT COMPLETE")
        print("=" * 50)
        print("✅ Technical implementation: PASSED")
        print(f"📝 Conceptual understanding: {quiz_score}/4")
        
        if quiz_score >= 3:
            print("\n🚀 Ready for Day 9: Vectors and Vector Operations!")
        else:
            print("\n📚 Consider reviewing today's material before proceeding.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Please review your implementation and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Create README.md
cat > README.md << 'EOF'
# Day 8: Introduction to Linear Algebra for AI

Welcome to Day 8 of the 180-Day AI/ML Course! Today you'll learn the mathematical foundation that powers all AI systems.

## Quick Start

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
source ai_course_env/bin/activate
```

### 2. Run the Lesson
```bash
python lesson_code.py
```

### 3. Test Your Understanding
```bash
python test_lesson.py
```

## What You'll Learn

- **Vectors**: How to represent data as mathematical objects
- **Matrices**: How to organize and transform data efficiently  
- **Dot Products**: How to measure similarity between data points
- **Matrix Operations**: How to build AI transformations
- **Recommendation Systems**: How to apply these concepts in practice

## Files Overview

- `lesson_code.py` - Interactive demonstrations of linear algebra concepts
- `test_lesson.py` - Comprehensive tests and quiz to verify understanding
- `setup.sh` - Environment setup script
- `requirements.txt` - Python dependencies

## Key Concepts

### Vectors
```python
customer_preferences = np.array([0.9, 0.2, 0.1, 0.8])
# Represents: [tech, books, clothes, sports] preferences
```

### Similarity with Dot Products
```python
similarity = np.dot(user_a_prefs, user_b_prefs)
# Higher values = more similar users
```

### Matrix Transformations
```python
recommendations = np.dot(user_ratings, recommendation_weights)
# Transform ratings into personalized recommendations
```

## Learning Objectives

By the end of this lesson, you should be able to:

✅ Create and manipulate NumPy vectors and matrices  
✅ Calculate similarity between data points using dot products  
✅ Understand how linear algebra powers AI recommendation systems  
✅ Transform data using matrix operations  
✅ Build a simple recommendation engine from scratch  

## Next Steps

**Tomorrow (Day 9)**: Vectors and Vector Operations
- Vector spaces and geometric interpretation
- Advanced vector operations (cross products, projections)
- Applications in computer graphics and AI

## Troubleshooting

**Python Version Issues:**
- Ensure Python 3.8+ is installed
- Use `python3 --version` to check

**Import Errors:**
- Activate virtual environment: `source ai_course_env/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**Visualization Issues:**
- Install additional backend: `pip install tkinter` (Linux)
- Run in Jupyter notebook for better visualization support

## Resources

- NumPy Documentation: https://numpy.org/doc/
- Linear Algebra Khan Academy: https://www.khanacademy.org/math/linear-algebra
- 3Blue1Brown Linear Algebra Series: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

## Course Progress

📍 **Current**: Day 8 - Introduction to Linear Algebra  
⬅️ **Previous**: Day 7 - Project: Command-Line Game  
➡️ **Next**: Day 9 - Vectors and Vector Operations  

---

*Part of the 180-Day AI/ML Course - Building production AI systems from scratch*
EOF

echo "✅ All files created successfully!"
echo ""
echo "To get started:"
echo "1. chmod +x setup.sh && ./setup.sh"
echo "2. source ai_course_env/bin/activate" 
echo "3. python lesson_code.py"
echo ""
echo "Files created:"
echo "- setup.sh (environment setup)"
echo "- lesson_code.py (main lesson)" 
echo "- test_lesson.py (tests and quiz)"
echo "- requirements.txt (dependencies)"
echo "- README.md (quick start guide)"
