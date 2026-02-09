"""
Movie Recommender System - Main Entry Point
Days 106-112 Project: Production-Ready Hybrid Recommender

This system implements the same architectural patterns used by
Netflix, Spotify, and YouTube to serve recommendations to millions
of users daily.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

from utils.data_loader import MovieLensLoader
from models.collaborative_filtering import CollaborativeFilter
from models.content_based import ContentBasedFilter
from models.hybrid_recommender import HybridRecommender
from utils.evaluator import RecommenderEvaluator


class MovieRecommenderSystem:
    """
    Production-ready movie recommendation system.
    
    Architecture mirrors Netflix's system:
    1. Data ingestion pipeline
    2. Collaborative filtering engine (matrix factorization)
    3. Content-based engine (feature similarity)
    4. Hybrid prediction layer
    5. Evaluation framework
    """
    
    def __init__(self, data_path: str = "data/ml-100k"):
        self.data_path = data_path
        self.loader = MovieLensLoader(data_path)
        self.collab_model = None
        self.content_model = None
        self.hybrid_model = None
        self.evaluator = RecommenderEvaluator()
        
        # Data containers
        self.ratings_df = None
        self.movies_df = None
        self.train_df = None
        self.test_df = None
        self.user_item_matrix = None
        
    def load_and_prepare_data(self):
        """Load and preprocess data."""
        print("📊 Loading MovieLens 100K dataset...")
        
        self.ratings_df, self.movies_df, _ = self.loader.load_data()
        
        # Create sparse matrix
        self.user_item_matrix = self.loader.create_user_item_matrix()
        
        # Temporal train-test split
        self.train_df, self.test_df = self.loader.train_test_split_temporal(
            test_size=0.2
        )
        
        # Print statistics
        stats = self.loader.get_user_stats()
        print("\n📈 Dataset Statistics:")
        print(f"  Users: {stats['n_users']}")
        print(f"  Movies: {stats['n_movies']}")
        print(f"  Ratings: {stats['n_ratings']}")
        print(f"  Sparsity: {stats['sparsity']:.2%}")
        print(f"  Avg ratings/user: {stats['avg_ratings_per_user']:.1f}")
        print(f"  Avg ratings/movie: {stats['avg_ratings_per_movie']:.1f}")
        
    def train_collaborative_model(self, n_factors: int = 50):
        """Train collaborative filtering model."""
        print("\n🤖 Training Collaborative Filtering Model...")
        start_time = time.time()
        
        # Create training matrix
        train_matrix = self.loader.create_user_item_matrix()
        
        # Train SVD
        self.collab_model = CollaborativeFilter(n_factors=n_factors)
        self.collab_model.fit(train_matrix)
        
        elapsed = time.time() - start_time
        print(f"✅ Collaborative model trained in {elapsed:.2f}s")
        print(f"  Latent factors: {n_factors}")
        print(f"  Matrix shape: {train_matrix.shape}")
        
    def train_content_model(self):
        """Train content-based filtering model."""
        print("\n📝 Training Content-Based Filtering Model...")
        start_time = time.time()
        
        # Extract genre features
        genre_features = self.loader.get_genre_features()
        
        # Train content model
        self.content_model = ContentBasedFilter()
        self.content_model.fit(self.movies_df, genre_features)
        
        elapsed = time.time() - start_time
        print(f"✅ Content model trained in {elapsed:.2f}s")
        print(f"  Feature dimensions: {self.content_model.item_features.shape[1]}")
        
    def create_hybrid_model(self):
        """Create hybrid recommender combining both models."""
        print("\n🔄 Creating Hybrid Recommender...")
        
        self.hybrid_model = HybridRecommender(
            self.collab_model,
            self.content_model,
            min_alpha=0.2,
            max_alpha=0.8
        )
        
        print("✅ Hybrid model created")
        print(f"  Min collaborative weight: {self.hybrid_model.min_alpha}")
        print(f"  Max collaborative weight: {self.hybrid_model.max_alpha}")
        
    def generate_recommendations(
        self,
        user_id: int,
        n_recommendations: int = 10
    ):
        """Generate recommendations for a user."""
        
        # Get user's rating history
        user_ratings = self.train_df[self.train_df['user_id'] == user_id]
        user_rated_items = user_ratings['movie_id'].values - 1  # 0-indexed
        user_rating_values = user_ratings['rating'].values
        
        # Generate hybrid recommendations
        recommended_items, scores, metadata = self.hybrid_model.recommend(
            user_id - 1,  # 0-indexed
            user_rated_items,
            user_rating_values,
            n_recommendations=n_recommendations
        )
        
        # Get movie titles
        recommended_movies = self.movies_df.iloc[recommended_items]
        
        return recommended_movies, scores, metadata
    
    def demonstrate_recommendations(self):
        """Show sample recommendations for different user profiles."""
        print("\n🎬 Generating Sample Recommendations...")
        print("=" * 70)
        
        # Find users with different experience levels
        user_rating_counts = self.train_df.groupby('user_id').size()
        
        # New user (few ratings) - use first user if no users with < 10 ratings
        new_users = user_rating_counts[user_rating_counts < 20]
        if len(new_users) > 0:
            new_user = new_users.index[0]
        else:
            new_user = user_rating_counts.index[0]
        
        # Established user (many ratings) - use user with most ratings if none > 100
        established_users = user_rating_counts[user_rating_counts > 50]
        if len(established_users) > 0:
            established_user = established_users.index[0]
        else:
            established_user = user_rating_counts.nlargest(1).index[0]
        
        # Generate recommendations for both
        for user_id, label in [(new_user, "NEW USER"), 
                               (established_user, "ESTABLISHED USER")]:
            print(f"\n{label} (User {user_id}):")
            print("-" * 70)
            
            # Get user's rating history
            user_history = self.train_df[self.train_df['user_id'] == user_id]
            print(f"Rating history: {len(user_history)} movies")
            
            # Show some highly-rated movies
            top_rated = user_history.nlargest(3, 'rating')
            print("\nTop-rated movies:")
            for _, row in top_rated.iterrows():
                movie_title = self.movies_df[
                    self.movies_df['movie_id'] == row['movie_id']
                ]['title'].values[0]
                print(f"  • {movie_title} (Rating: {row['rating']})")
            
            # Generate recommendations
            recommended_movies, scores, metadata = self.generate_recommendations(
                user_id, n_recommendations=5
            )
            
            print(f"\nRecommendations (blend: {metadata['alpha']:.2f} collab, "
                  f"{metadata['content_weight']:.2f} content):")
            for i, (_, movie) in enumerate(recommended_movies.iterrows()):
                print(f"  {i+1}. {movie['title']} (score: {scores[i]:.2f})")
        
        print("\n" + "=" * 70)
    
    def evaluate_performance(self):
        """Evaluate model performance on test set."""
        print("\n📊 Evaluating Model Performance...")
        print("=" * 70)
        
        predictions = []
        actuals = []
        
        # Sample evaluation on subset of test data
        test_sample = self.test_df.sample(n=min(1000, len(self.test_df)))
        
        for _, row in test_sample.iterrows():
            user_id = row['user_id'] - 1  # 0-indexed
            movie_id = row['movie_id'] - 1
            actual_rating = row['rating']
            
            # Get user history from training set
            user_history = self.train_df[self.train_df['user_id'] == row['user_id']]
            user_rated_items = user_history['movie_id'].values - 1
            user_rating_values = user_history['rating'].values
            
            # Predict
            predicted_rating = self.hybrid_model.predict(
                user_id,
                movie_id,
                user_rated_items,
                user_rating_values,
                len(user_rated_items)
            )
            
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Compute metrics
        rmse = self.evaluator.rmse(predictions, actuals)
        mae = self.evaluator.mae(predictions, actuals)
        
        print(f"\nPrediction Accuracy Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"\n  Baseline (always predict 3.5): RMSE ≈ 1.11")
        print(f"  Netflix Prize goal: RMSE < 0.86")
        print(f"  Our model improvement: {((1.11 - rmse) / 1.11 * 100):.1f}% vs baseline")
        
        print("\n" + "=" * 70)
        
        return {'rmse': rmse, 'mae': mae}
    
    def visualize_results(self):
        """Create visualization of model performance."""
        print("\n📈 Generating Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Movie Recommender System Analysis', fontsize=16, y=1.00)
        
        # 1. Rating distribution
        ax1 = axes[0, 0]
        sns.histplot(data=self.ratings_df, x='rating', bins=5, ax=ax1)
        ax1.set_title('Rating Distribution')
        ax1.set_xlabel('Rating')
        ax1.set_ylabel('Count')
        
        # 2. Ratings per user
        ax2 = axes[0, 1]
        ratings_per_user = self.ratings_df.groupby('user_id').size()
        ax2.hist(ratings_per_user, bins=50, edgecolor='black')
        ax2.set_title('Ratings per User Distribution')
        ax2.set_xlabel('Number of Ratings')
        ax2.set_ylabel('Number of Users')
        ax2.axvline(ratings_per_user.mean(), color='red', 
                   linestyle='--', label=f'Mean: {ratings_per_user.mean():.1f}')
        ax2.legend()
        
        # 3. Most popular movies
        ax3 = axes[1, 0]
        top_movies = self.ratings_df.groupby('movie_id').size().nlargest(10)
        movie_titles = [self.movies_df[self.movies_df['movie_id'] == mid]['title'].values[0][:20] 
                       for mid in top_movies.index]
        ax3.barh(movie_titles, top_movies.values)
        ax3.set_title('Top 10 Most Rated Movies')
        ax3.set_xlabel('Number of Ratings')
        ax3.invert_yaxis()
        
        # 4. Genre distribution
        ax4 = axes[1, 1]
        genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror']
        genre_counts = self.movies_df[genre_cols].sum().sort_values(ascending=False)[:10]
        ax4.bar(range(len(genre_counts)), genre_counts.values)
        ax4.set_xticks(range(len(genre_counts)))
        ax4.set_xticklabels(genre_counts.index, rotation=45, ha='right')
        ax4.set_title('Top 10 Genres')
        ax4.set_ylabel('Number of Movies')
        
        plt.tight_layout()
        plt.savefig('recommender_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved to 'recommender_analysis.png'")
    
    def save_models(self, output_dir: str = "models"):
        """Save trained models to disk."""
        Path(output_dir).mkdir(exist_ok=True)
        
        self.collab_model.save(f"{output_dir}/collaborative_model.pkl")
        self.content_model.save(f"{output_dir}/content_model.pkl")
        self.hybrid_model.save(f"{output_dir}/hybrid_config.pkl")
        
        print(f"\n💾 Models saved to '{output_dir}/' directory")


def main():
    """
    Main execution flow for movie recommender system.
    
    This 7-day project demonstrates production-ready recommendation
    system design, from data pipeline through evaluation.
    """
    
    print("=" * 70)
    print("🎬 MOVIE RECOMMENDER SYSTEM - DAYS 106-112 PROJECT")
    print("=" * 70)
    print("\nBuilding production-ready hybrid recommendation engine...")
    print("Architecture: Collaborative + Content-Based Filtering")
    print()
    
    # Initialize system
    recommender = MovieRecommenderSystem()
    
    # Phase 1: Data Pipeline
    recommender.load_and_prepare_data()
    
    # Phase 2: Train Models
    recommender.train_collaborative_model(n_factors=50)
    recommender.train_content_model()
    
    # Phase 3: Create Hybrid System
    recommender.create_hybrid_model()
    
    # Phase 4: Demonstration
    recommender.demonstrate_recommendations()
    
    # Phase 5: Evaluation
    metrics = recommender.evaluate_performance()
    
    # Phase 6: Visualization
    recommender.visualize_results()
    
    # Phase 7: Save Models
    recommender.save_models()
    
    print("\n" + "=" * 70)
    print("✅ PROJECT COMPLETE!")
    print("=" * 70)
    print("\nWhat we built:")
    print("  • Data ingestion pipeline handling 100K ratings")
    print("  • Collaborative filtering with matrix factorization (SVD)")
    print("  • Content-based filtering with feature similarity")
    print("  • Hybrid system with adaptive blending")
    print("  • Comprehensive evaluation framework")
    print("  • Production-ready architecture")
    print()
    print("Real-world parallels:")
    print("  • Netflix: 200M users, 80% engagement from recommendations")
    print("  • Spotify: Audio analysis enables instant cold-start recommendations")
    print("  • YouTube: Two-stage architecture (retrieval + ranking)")
    print()


if __name__ == "__main__":
    main()

