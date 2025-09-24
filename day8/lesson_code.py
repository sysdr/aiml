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
