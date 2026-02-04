"""
Day 105: Content-Based Filtering Implementation
Production-grade recommendation system using TF-IDF and cosine similarity
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import json
from datetime import datetime


class ContentBasedRecommender:
    """
    Production-ready content-based filtering system.
    
    Architecture:
    - Feature extraction using TF-IDF with configurable parameters
    - Similarity computation with cached results
    - Hybrid scoring with business logic overlay
    - Incremental updates for new items
    """
    
    def __init__(self, 
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2):
        """
        Initialize the recommender system.
        
        Args:
            max_features: Maximum vocabulary size for TF-IDF
            ngram_range: Range of n-grams to extract
            min_df: Minimum document frequency for terms
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=min_df,
            strip_accents='unicode',
            lowercase=True
        )
        
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.item_indices = {}
        self.items_df = None
        self.fitted = False
        
        # Performance metrics
        self.metrics = {
            'fit_time': 0,
            'total_items': 0,
            'vocabulary_size': 0,
            'recommendations_served': 0
        }
    
    def fit(self, items_df: pd.DataFrame, text_column: str = 'combined_features'):
        """
        Fit the recommender on item catalog.
        
        Args:
            items_df: DataFrame with item metadata
            text_column: Column containing combined text features
        """
        start_time = datetime.now()
        
        print(f"🔄 Fitting content-based recommender on {len(items_df)} items...")
        
        # Store item data
        self.items_df = items_df.copy()
        
        # Create item index mapping
        self.item_indices = {
            item_id: idx for idx, item_id in enumerate(items_df['item_id'])
        }
        
        # Extract TF-IDF features
        print("📊 Computing TF-IDF features...")
        self.tfidf_matrix = self.vectorizer.fit_transform(
            items_df[text_column].fillna('')
        )
        
        # Compute similarity matrix
        print("🔍 Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        # Update metrics
        self.metrics['fit_time'] = (datetime.now() - start_time).total_seconds()
        self.metrics['total_items'] = len(items_df)
        self.metrics['vocabulary_size'] = len(self.vectorizer.vocabulary_)
        self.fitted = True
        
        print(f"✅ Fit complete in {self.metrics['fit_time']:.2f}s")
        print(f"   Vocabulary size: {self.metrics['vocabulary_size']}")
        print(f"   Matrix shape: {self.tfidf_matrix.shape}")
        print(f"   Matrix density: {self.tfidf_matrix.nnz / np.prod(self.tfidf_matrix.shape):.4f}")
    
    def get_recommendations(self, 
                          item_id: str,
                          n_recommendations: int = 10,
                          apply_boost: bool = True,
                          diversity_threshold: float = 0.95) -> List[Dict]:
        """
        Generate content-based recommendations for an item.
        
        Args:
            item_id: Target item ID
            n_recommendations: Number of recommendations to return
            apply_boost: Whether to apply popularity boosting
            diversity_threshold: Maximum similarity for diversity filtering
            
        Returns:
            List of recommendation dictionaries with scores
        """
        if not self.fitted:
            raise ValueError("Recommender must be fitted before generating recommendations")
        
        if item_id not in self.item_indices:
            raise ValueError(f"Item {item_id} not found in catalog")
        
        # Get item index
        item_idx = self.item_indices[item_id]
        
        # Get similarity scores
        similarity_scores = self.similarity_matrix[item_idx]
        
        # Create scored items
        scored_items = []
        for idx, score in enumerate(similarity_scores):
            if idx == item_idx:  # Skip self
                continue
            
            current_item_id = self.items_df.iloc[idx]['item_id']
            item_data = self.items_df.iloc[idx].to_dict()
            
            # Base similarity score
            final_score = score
            
            # Apply popularity boost if enabled
            if apply_boost and 'popularity' in item_data:
                popularity_factor = np.log1p(item_data['popularity']) / 10
                final_score = final_score * (1 + popularity_factor)
            
            scored_items.append({
                'item_id': current_item_id,
                'similarity_score': float(score),
                'final_score': float(final_score),
                'item_data': item_data
            })
        
        # Sort by final score
        scored_items.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Apply diversity filtering
        if diversity_threshold < 1.0:
            scored_items = self._apply_diversity_filter(
                scored_items, 
                diversity_threshold
            )
        
        # Update metrics
        self.metrics['recommendations_served'] += 1
        
        return scored_items[:n_recommendations]
    
    def _apply_diversity_filter(self, 
                               scored_items: List[Dict],
                               threshold: float) -> List[Dict]:
        """
        Apply diversity filtering to avoid overly similar recommendations.
        
        Args:
            scored_items: List of scored recommendation candidates
            threshold: Maximum similarity between recommended items
            
        Returns:
            Filtered list of diverse recommendations
        """
        if not scored_items:
            return scored_items
        
        diverse_items = [scored_items[0]]  # Always include top item
        
        for candidate in scored_items[1:]:
            # Check similarity with already selected items
            is_diverse = True
            candidate_idx = self.item_indices[candidate['item_id']]
            
            for selected in diverse_items:
                selected_idx = self.item_indices[selected['item_id']]
                similarity = self.similarity_matrix[candidate_idx, selected_idx]
                
                if similarity > threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_items.append(candidate)
        
        return diverse_items
    
    def add_new_item(self, item_data: Dict, text_features: str):
        """
        Incrementally add a new item to the system.
        
        Args:
            item_data: Dictionary with item metadata
            text_features: Combined text features for the item
        """
        if not self.fitted:
            raise ValueError("Recommender must be fitted before adding items")
        
        print(f"➕ Adding new item: {item_data['item_id']}")
        
        # Transform new item using existing vectorizer
        new_tfidf = self.vectorizer.transform([text_features])
        
        # Compute similarities with existing items
        new_similarities = cosine_similarity(new_tfidf, self.tfidf_matrix)[0]
        
        # Update data structures
        new_idx = len(self.items_df)
        self.item_indices[item_data['item_id']] = new_idx
        
        # Add to DataFrame
        new_row = pd.DataFrame([item_data])
        self.items_df = pd.concat([self.items_df, new_row], ignore_index=True)
        
        # Update TF-IDF matrix (sparse concatenation)
        from scipy.sparse import vstack
        self.tfidf_matrix = vstack([self.tfidf_matrix, new_tfidf])
        
        # Update similarity matrix
        new_col = np.append(new_similarities, 1.0).reshape(-1, 1)
        new_row = np.append(new_similarities, 1.0).reshape(1, -1)
        
        self.similarity_matrix = np.vstack([
            np.hstack([self.similarity_matrix, new_similarities.reshape(-1, 1)]),
            new_row
        ])
        
        self.metrics['total_items'] += 1
        print(f"✅ Item added. Total items: {self.metrics['total_items']}")
    
    def get_feature_importance(self, item_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get most important features for an item.
        
        Args:
            item_id: Target item ID
            top_n: Number of top features to return
            
        Returns:
            List of (feature, weight) tuples
        """
        if item_id not in self.item_indices:
            raise ValueError(f"Item {item_id} not found")
        
        item_idx = self.item_indices[item_id]
        feature_vector = self.tfidf_matrix[item_idx].toarray()[0]
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Create feature-weight pairs
        features = [(feature_names[i], feature_vector[i]) 
                   for i in range(len(feature_vector)) 
                   if feature_vector[i] > 0]
        
        # Sort by weight
        features.sort(key=lambda x: x[1], reverse=True)
        
        return features[:top_n]
    
    def get_metrics(self) -> Dict:
        """Return system performance metrics."""
        return self.metrics.copy()


def create_sample_dataset() -> pd.DataFrame:
    """
    Create a sample movie dataset for demonstration.
    
    Returns:
        DataFrame with movie metadata
    """
    movies = [
        {
            'item_id': 'movie_001',
            'title': 'The Matrix',
            'genres': 'Science Fiction Action',
            'description': 'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
            'director': 'Wachowski',
            'actors': 'Keanu Reeves Laurence Fishburne',
            'year': 1999,
            'popularity': 9500
        },
        {
            'item_id': 'movie_002',
            'title': 'Inception',
            'genres': 'Science Fiction Thriller',
            'description': 'A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.',
            'director': 'Christopher Nolan',
            'actors': 'Leonardo DiCaprio',
            'year': 2010,
            'popularity': 8900
        },
        {
            'item_id': 'movie_003',
            'title': 'The Notebook',
            'genres': 'Romance Drama',
            'description': 'A poor yet passionate young man falls in love with a rich young woman, giving her a sense of freedom.',
            'director': 'Nick Cassavetes',
            'actors': 'Ryan Gosling Rachel McAdams',
            'year': 2004,
            'popularity': 7200
        },
        {
            'item_id': 'movie_004',
            'title': 'Blade Runner 2049',
            'genres': 'Science Fiction Thriller',
            'description': 'A young blade runner discovers a long-buried secret that leads him to track down former blade runner Rick Deckard.',
            'director': 'Denis Villeneuve',
            'actors': 'Ryan Gosling Harrison Ford',
            'year': 2017,
            'popularity': 7800
        },
        {
            'item_id': 'movie_005',
            'title': 'Interstellar',
            'genres': 'Science Fiction Adventure',
            'description': 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival.',
            'director': 'Christopher Nolan',
            'actors': 'Matthew McConaughey Anne Hathaway',
            'year': 2014,
            'popularity': 9100
        },
        {
            'item_id': 'movie_006',
            'title': 'Pride and Prejudice',
            'genres': 'Romance Drama',
            'description': 'Sparks fly when spirited Elizabeth Bennet meets single, rich, and proud Mr. Darcy.',
            'director': 'Joe Wright',
            'actors': 'Keira Knightley',
            'year': 2005,
            'popularity': 6500
        },
        {
            'item_id': 'movie_007',
            'title': 'The Dark Knight',
            'genres': 'Action Crime Thriller',
            'description': 'Batman must accept one of the greatest psychological and physical tests to fight injustice.',
            'director': 'Christopher Nolan',
            'actors': 'Christian Bale Heath Ledger',
            'year': 2008,
            'popularity': 9800
        },
        {
            'item_id': 'movie_008',
            'title': 'Ex Machina',
            'genres': 'Science Fiction Thriller',
            'description': 'A young programmer is selected to participate in a groundbreaking experiment in synthetic intelligence.',
            'director': 'Alex Garland',
            'actors': 'Alicia Vikander',
            'year': 2014,
            'popularity': 7000
        }
    ]
    
    df = pd.DataFrame(movies)
    
    # Create combined features for TF-IDF
    df['combined_features'] = (
        df['title'] + ' ' +
        df['genres'] + ' ' +
        df['description'] + ' ' +
        df['director'] + ' ' +
        df['actors']
    )
    
    return df


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("Day 105: Content-Based Filtering - Production System Demo")
    print("=" * 70)
    print()
    
    # Create sample dataset
    print("📚 Loading sample movie dataset...")
    movies_df = create_sample_dataset()
    print(f"   Loaded {len(movies_df)} movies")
    print()
    
    # Initialize recommender
    recommender = ContentBasedRecommender(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1
    )
    
    # Fit the model
    recommender.fit(movies_df, text_column='combined_features')
    print()
    
    # Generate recommendations
    target_movie = 'movie_001'  # The Matrix
    target_title = movies_df[movies_df['item_id'] == target_movie]['title'].values[0]
    
    print(f"🎬 Generating recommendations for: {target_title}")
    print()
    
    recommendations = recommender.get_recommendations(
        target_movie,
        n_recommendations=5,
        apply_boost=True,
        diversity_threshold=0.85
    )
    
    print("📊 Top Recommendations:")
    print("-" * 70)
    for i, rec in enumerate(recommendations, 1):
        item = rec['item_data']
        print(f"{i}. {item['title']} ({item['year']})")
        print(f"   Genres: {item['genres']}")
        print(f"   Similarity: {rec['similarity_score']:.3f} | Final Score: {rec['final_score']:.3f}")
        print()
    
    # Show feature importance
    print("🔍 Most Important Features for 'The Matrix':")
    print("-" * 70)
    features = recommender.get_feature_importance(target_movie, top_n=10)
    for feature, weight in features:
        print(f"   {feature}: {weight:.4f}")
    print()
    
    # Demonstrate incremental update
    print("➕ Adding a new movie to the catalog...")
    new_movie = {
        'item_id': 'movie_009',
        'title': 'Tron Legacy',
        'genres': 'Science Fiction Action',
        'description': 'The son of a virtual world designer goes looking for his father and ends up inside the digital world.',
        'director': 'Joseph Kosinski',
        'actors': 'Jeff Bridges',
        'year': 2010,
        'popularity': 6800
    }
    
    new_features = (
        new_movie['title'] + ' ' +
        new_movie['genres'] + ' ' +
        new_movie['description'] + ' ' +
        new_movie['director'] + ' ' +
        new_movie['actors']
    )
    
    recommender.add_new_item(new_movie, new_features)
    print()
    
    # Get recommendations for the new item
    print(f"🎬 Recommendations for new movie: {new_movie['title']}")
    print("-" * 70)
    new_recs = recommender.get_recommendations('movie_009', n_recommendations=3)
    for i, rec in enumerate(new_recs, 1):
        item = rec['item_data']
        print(f"{i}. {item['title']} - Similarity: {rec['similarity_score']:.3f}")
    print()
    
    # Show metrics
    print("📈 System Metrics:")
    print("-" * 70)
    metrics = recommender.get_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    print()
    
    print("✅ Demo complete!")
    print()
    print("Next Steps:")
    print("- Experiment with different n-gram ranges")
    print("- Try different similarity thresholds")
    print("- Implement custom business logic boosters")
    print("- Add more diverse feature types (numerical, categorical)")


if __name__ == "__main__":
    main()
