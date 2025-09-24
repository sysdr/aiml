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
