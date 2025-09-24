"""
Test Suite for Day 9: Vectors and Vector Operations
Verify your understanding with these hands-on tests
"""

import numpy as np
import sys
from lesson_code import VectorExplorer, AIRecommendationEngine

class VectorTests:
    """Test suite to verify vector operation understanding"""
    
    def __init__(self):
        self.passed = 0
        self.total = 0
        print("🧪 Starting Vector Operations Test Suite")
        print("=" * 50)
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and track results"""
        self.total += 1
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}")
                self.passed += 1
            else:
                print(f"❌ {test_name}")
        except Exception as e:
            print(f"❌ {test_name} - Error: {e}")
    
    def test_basic_vector_operations(self):
        """Test understanding of basic vector operations"""
        v1 = np.array([3.0, 4.0])
        v2 = np.array([1.0, 2.0])
        
        # Test vector addition
        addition_result = v1 + v2
        expected_addition = np.array([4.0, 6.0])
        
        # Test dot product
        dot_result = np.dot(v1, v2)
        expected_dot = 11.0  # 3*1 + 4*2 = 11
        
        # Test magnitude
        magnitude_result = np.linalg.norm(v1)
        expected_magnitude = 5.0  # sqrt(3^2 + 4^2) = 5
        
        return (np.allclose(addition_result, expected_addition) and 
                abs(dot_result - expected_dot) < 0.001 and
                abs(magnitude_result - expected_magnitude) < 0.001)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        # Identical vectors should have similarity = 1
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])
        
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        return abs(similarity - 1.0) < 0.001
    
    def test_recommendation_engine(self):
        """Test the AI recommendation engine"""
        engine = AIRecommendationEngine()
        
        # Test with action movie preferences
        action_preferences = np.array([5.0, 0.0, 1.0, 1.0, 3.0])
        recommendations = engine.recommend_movies(action_preferences, top_n=3)
        
        # Should return 3 recommendations with scores
        if len(recommendations) != 3:
            return False
        
        # All scores should be between -1 and 1 for cosine similarity
        for movie, score in recommendations:
            if not (-1 <= score <= 1):
                return False
        
        # Scores should be in descending order
        scores = [score for _, score in recommendations]
        return all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    def test_vector_similarity_logic(self):
        """Test that similar vectors have higher similarity scores"""
        # Two similar preference vectors
        user1 = np.array([5.0, 1.0, 4.0, 2.0, 5.0])  # Loves action and sci-fi
        user2 = np.array([4.5, 1.5, 4.2, 2.1, 4.8])  # Very similar taste
        user3 = np.array([1.0, 5.0, 1.0, 5.0, 1.0])  # Loves romance and drama
        
        # Calculate similarities
        sim_1_2 = np.dot(user1, user2) / (np.linalg.norm(user1) * np.linalg.norm(user2))
        sim_1_3 = np.dot(user1, user3) / (np.linalg.norm(user1) * np.linalg.norm(user3))
        
        # user1 should be more similar to user2 than user3
        return sim_1_2 > sim_1_3
    
    def test_vector_normalization(self):
        """Test vector normalization"""
        v = np.array([3.0, 4.0])
        normalized_v = v / np.linalg.norm(v)
        
        # Normalized vector should have magnitude 1
        magnitude = np.linalg.norm(normalized_v)
        return abs(magnitude - 1.0) < 0.001
    
    def run_all_tests(self):
        """Run all tests and show results"""
        self.run_test("Basic Vector Operations", self.test_basic_vector_operations)
        self.run_test("Cosine Similarity Calculation", self.test_cosine_similarity)
        self.run_test("Recommendation Engine Functionality", self.test_recommendation_engine)
        self.run_test("Vector Similarity Logic", self.test_vector_similarity_logic)
        self.run_test("Vector Normalization", self.test_vector_normalization)
        
        print("\n" + "=" * 50)
        print(f"📊 TEST RESULTS: {self.passed}/{self.total} passed")
        
        if self.passed == self.total:
            print("🎉 CONGRATULATIONS! You've mastered vector operations!")
            print("🚀 Ready for Day 10: Matrices and Matrix Operations")
        else:
            print("📚 Review the lesson and try again. You've got this!")
        
        print("=" * 50)
        return self.passed == self.total

def interactive_quiz():
    """Interactive quiz to test understanding"""
    print("\n🎯 INTERACTIVE VECTOR QUIZ")
    print("=" * 30)
    
    score = 0
    total_questions = 3
    
    # Question 1
    print("Question 1: If vector A = [2, 3] and vector B = [1, 4],")
    print("what is the dot product A • B?")
    print("a) [2, 12]")
    print("b) 14")
    print("c) [3, 7]")
    print("d) 6")
    
    answer1 = input("Your answer (a/b/c/d): ").lower().strip()
    if answer1 == 'b':
        print("✅ Correct! A • B = 2×1 + 3×4 = 14")
        score += 1
    else:
        print("❌ Incorrect. A • B = 2×1 + 3×4 = 14")
    
    # Question 2
    print("\nQuestion 2: In AI recommendation systems, what does")
    print("a higher cosine similarity between user vectors indicate?")
    print("a) Users have opposite preferences")
    print("b) Users have similar preferences") 
    print("c) Users are the same person")
    print("d) The calculation failed")
    
    answer2 = input("Your answer (a/b/c/d): ").lower().strip()
    if answer2 == 'b':
        print("✅ Correct! Higher cosine similarity means more similar preferences")
        score += 1
    else:
        print("❌ Incorrect. Higher cosine similarity indicates similar preferences")
    
    # Question 3
    print("\nQuestion 3: Vector normalization ensures that:")
    print("a) All vector components are positive")
    print("b) The vector has magnitude 1")
    print("c) The vector becomes [0, 0, 0]")
    print("d) Vectors can't be compared")
    
    answer3 = input("Your answer (a/b/c/d): ").lower().strip()
    if answer3 == 'b':
        print("✅ Correct! Normalization creates unit vectors with magnitude 1")
        score += 1
    else:
        print("❌ Incorrect. Normalization creates unit vectors with magnitude 1")
    
    print(f"\n📊 Quiz Score: {score}/{total_questions}")
    if score == total_questions:
        print("🏆 Perfect! You understand vectors for AI!")
    elif score >= 2:
        print("👍 Good job! Minor review recommended")
    else:
        print("📚 Review the lesson and try again")

if __name__ == "__main__":
    # Run automated tests
    test_suite = VectorTests()
    all_passed = test_suite.run_all_tests()
    
    # Run interactive quiz if tests pass
    if all_passed:
        interactive_quiz()
    else:
        print("\n🔄 Fix the test failures before taking the quiz!")
