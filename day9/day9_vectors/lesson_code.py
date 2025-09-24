"""
Day 9: Vectors and Vector Operations for AI
A hands-on exploration of vectors as the foundation of AI systems
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json

class VectorExplorer:
    """Interactive class to explore vector concepts for AI"""
    
    def __init__(self):
        print("🎯 Welcome to Vector Explorer for AI!")
        print("Today we'll discover how vectors power modern AI systems\n")
    
    def demonstrate_basic_vectors(self):
        """Show basic vector creation and visualization"""
        print("=" * 50)
        print("🔢 BASIC VECTORS: AI's Building Blocks")
        print("=" * 50)
        
        # Create sample vectors representing user preferences
        user1_preferences = np.array([4.2, 1.5, 5.0, 2.8, 3.1])  # [action, romance, comedy, drama, sci-fi]
        user2_preferences = np.array([2.1, 4.8, 2.0, 4.5, 1.2])
        user3_preferences = np.array([4.5, 1.2, 4.8, 2.5, 3.8])  # Similar to user1
        
        print(f"User 1 preferences: {user1_preferences}")
        print(f"User 2 preferences: {user2_preferences}")
        print(f"User 3 preferences: {user3_preferences}")
        print("\n🎬 Dimensions: [Action, Romance, Comedy, Drama, Sci-Fi]")
        
        return user1_preferences, user2_preferences, user3_preferences
    
    def vector_operations_demo(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray):
        """Demonstrate key vector operations used in AI"""
        print("\n" + "=" * 50)
        print("🔧 VECTOR OPERATIONS: The Math Behind AI")
        print("=" * 50)
        
        # Vector addition - combining preferences
        print("➕ VECTOR ADDITION (Combining preferences):")
        combined_preferences = (v1 + v3) / 2  # Average similar users
        print(f"Average of User 1 & 3: {combined_preferences}")
        
        # Dot product - similarity measure
        print("\n📊 DOT PRODUCT (Similarity measurement):")
        similarity_1_2 = np.dot(v1, v2)
        similarity_1_3 = np.dot(v1, v3)
        print(f"User 1 ⋅ User 2 = {similarity_1_2:.2f}")
        print(f"User 1 ⋅ User 3 = {similarity_1_3:.2f}")
        print(f"User 1 is more similar to User {'3' if similarity_1_3 > similarity_1_2 else '2'}")
        
        # Vector magnitude
        print("\n📏 VECTOR MAGNITUDE (Strength of preferences):")
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)
        print(f"User 1 magnitude: {mag_v1:.2f}")
        print(f"User 2 magnitude: {mag_v2:.2f}")
        print(f"User {'1' if mag_v1 > mag_v2 else '2'} has stronger overall preferences")
        
        # Cosine similarity - the gold standard in AI
        print("\n🎯 COSINE SIMILARITY (AI's favorite metric):")
        cos_sim_1_2 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_sim_1_3 = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
        print(f"User 1 ↔ User 2: {cos_sim_1_2:.3f}")
        print(f"User 1 ↔ User 3: {cos_sim_1_3:.3f}")
        print("📈 Range: -1 (opposite) to 1 (identical)")

class AIRecommendationEngine:
    """A real AI recommendation system using vector operations"""
    
    def __init__(self):
        # Movie database with feature vectors
        self.movies = {
            "The Matrix": np.array([5.0, 1.0, 2.0, 3.0, 5.0]),
            "Titanic": np.array([2.0, 5.0, 1.0, 5.0, 1.0]),
            "Avengers Endgame": np.array([5.0, 2.0, 3.0, 3.0, 4.0]),
            "The Notebook": np.array([1.0, 5.0, 2.0, 4.0, 1.0]),
            "Interstellar": np.array([3.0, 2.0, 1.0, 4.0, 5.0]),
            "Guardians of Galaxy": np.array([4.0, 1.5, 4.0, 2.5, 4.5]),
            "Pride and Prejudice": np.array([1.0, 4.5, 2.5, 4.0, 0.5]),
            "Mad Max Fury Road": np.array([5.0, 1.0, 1.5, 2.0, 3.0]),
            "Her": np.array([1.0, 4.0, 2.0, 4.5, 3.5]),
            "Blade Runner 2049": np.array([4.0, 2.0, 1.0, 4.0, 5.0])
        }
        
        print("🎬 AI Movie Recommendation Engine Initialized!")
        print(f"📚 Loaded {len(self.movies)} movies in database")
        print("🔍 Features: [Action, Romance, Comedy, Drama, Sci-Fi]\n")
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors"""
        return np.linalg.norm(v1 - v2)
    
    def recommend_movies(self, user_preferences: np.ndarray, method: str = 'cosine', top_n: int = 3) -> List[Tuple[str, float]]:
        """Generate movie recommendations based on user preference vector"""
        recommendations = []
        
        for movie, features in self.movies.items():
            if method == 'cosine':
                score = self.cosine_similarity(user_preferences, features)
            elif method == 'euclidean':
                # Convert distance to similarity (smaller distance = higher similarity)
                distance = self.euclidean_distance(user_preferences, features)
                score = 1 / (1 + distance)  # Convert to similarity score
            else:
                raise ValueError("Method must be 'cosine' or 'euclidean'")
            
            recommendations.append((movie, score))
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
    
    def explain_recommendation(self, user_prefs: np.ndarray, movie_name: str):
        """Explain why a movie was recommended using vector analysis"""
        if movie_name not in self.movies:
            print(f"❌ Movie '{movie_name}' not found in database")
            return
        
        movie_features = self.movies[movie_name]
        similarity = self.cosine_similarity(user_prefs, movie_features)
        
        print(f"\n🎯 Why we recommended '{movie_name}':")
        print(f"📊 Overall similarity: {similarity:.3f}")
        print(f"👤 Your preferences:  {user_prefs}")
        print(f"🎬 Movie features:    {movie_features}")
        
        # Analyze dimension by dimension
        genres = ["Action", "Romance", "Comedy", "Drama", "Sci-Fi"]
        print("\n📈 Breakdown by genre:")
        for i, genre in enumerate(genres):
            user_score = user_prefs[i]
            movie_score = movie_features[i]
            match = "✅" if abs(user_score - movie_score) < 1.5 else "❌"
            print(f"  {genre:8}: You={user_score:.1f}, Movie={movie_score:.1f} {match}")

def interactive_demo():
    """Run an interactive demonstration of vector concepts"""
    print("🌟 INTERACTIVE VECTOR DEMO FOR AI")
    print("=" * 60)
    
    # Initialize our tools
    explorer = VectorExplorer()
    engine = AIRecommendationEngine()
    
    # Demonstrate basic vector concepts
    v1, v2, v3 = explorer.demonstrate_basic_vectors()
    explorer.vector_operations_demo(v1, v2, v3)
    
    print("\n" + "=" * 60)
    print("🎬 MOVIE RECOMMENDATION DEMO")
    print("=" * 60)
    
    # Demo different user types
    user_profiles = {
        "Action Fan": np.array([5.0, 1.0, 3.0, 2.0, 4.0]),
        "Romance Lover": np.array([1.0, 5.0, 2.5, 4.0, 1.0]),
        "Sci-Fi Enthusiast": np.array([3.0, 1.5, 2.0, 3.0, 5.0]),
        "Balanced Viewer": np.array([3.5, 3.0, 3.5, 3.5, 3.0])
    }
    
    for profile_name, preferences in user_profiles.items():
        print(f"\n👤 {profile_name.upper()} RECOMMENDATIONS:")
        print(f"📊 Preferences: {preferences}")
        
        recommendations = engine.recommend_movies(preferences, method='cosine', top_n=3)
        
        print("🏆 Top 3 recommendations:")
        for i, (movie, score) in enumerate(recommendations, 1):
            print(f"  {i}. {movie:20} (similarity: {score:.3f})")
        
        # Explain the top recommendation
        if recommendations:
            top_movie = recommendations[0][0]
            engine.explain_recommendation(preferences, top_movie)
    
    print("\n" + "=" * 60)
    print("🎓 VECTOR LEARNING COMPLETE!")
    print("=" * 60)
    print("✅ You now understand how vectors power AI recommendations!")
    print("✅ Tomorrow: Matrices - collections of vectors for neural networks")
    print("🚀 Keep building your AI foundation!")

if __name__ == "__main__":
    interactive_demo()
