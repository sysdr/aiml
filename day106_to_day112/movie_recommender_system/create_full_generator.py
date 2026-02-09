#!/usr/bin/env python3
"""
Script to create the full generator bash script with all file content.
This recreates the original 2299-line setup.sh generator.
"""

# Read the current partial setup.sh
with open('setup.sh', 'r') as f:
    current_content = f.read()

# The generator needs to create all Python files
# Since I have the structure from the original read, I'll create a script
# that generates the complete generator

# For now, let's append the Python file generation to setup.sh
# We'll use heredoc syntax for each file

additional_content = '''

# Generate utils/data_loader.py
cat > utils/data_loader.py << 'DATALOADEREOF'
"""
Data loading and preprocessing for MovieLens 100K dataset.
Handles user-item ratings, movie metadata, and train-test splits.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


class MovieLensLoader:
    """
    Loads and preprocesses MovieLens 100K dataset.
    
    Production systems process millions of interaction events daily.
    Netflix's data pipeline handles 1.5TB of logs, maintaining
    real-time user profiles updated within seconds.
    """
    
    def __init__(self, data_path: str = "data/ml-100k"):
        self.data_path = data_path
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        self.user_item_matrix = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load ratings, movies, and user data."""
        
        # Load ratings (user_id, movie_id, rating, timestamp)
        self.ratings_df = pd.read_csv(
            f"{self.data_path}/u.data",
            sep='\\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        # Load movie metadata
        self.movies_df = pd.read_csv(
            f"{self.data_path}/u.item",
            sep='|',
            names=['movie_id', 'title', 'release_date', 'video_release_date',
                   'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                   'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                   'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
            encoding='latin-1'
        )
        
        # Load user demographics (optional)
        self.users_df = pd.read_csv(
            f"{self.data_path}/u.user",
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            encoding='latin-1'
        )
        
        return self.ratings_df, self.movies_df, self.users_df
    
    def create_user_item_matrix(self) -> csr_matrix:
        """
        Create sparse user-item rating matrix.
        Shape: (n_users, n_items)
        
        Sparse representation is crucial for efficiency.
        Netflix's matrix: 200M users × 15K titles = 3 trillion cells,
        but only ~0.01% filled (20-30 billion actual ratings).
        """
        n_users = self.ratings_df['user_id'].max()
        n_movies = self.ratings_df['movie_id'].max()
        
        # Create sparse matrix
        self.user_item_matrix = csr_matrix(
            (self.ratings_df['rating'],
             (self.ratings_df['user_id'] - 1,
              self.ratings_df['movie_id'] - 1)),
            shape=(n_users, n_movies)
        )
        
        return self.user_item_matrix
    
    def train_test_split_temporal(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data maintaining temporal order per user.
        
        Production systems use time-based splits to prevent
        data leakage. You can't predict the future using
        information from the future!
        """
        
        # Sort by timestamp
        sorted_ratings = self.ratings_df.sort_values('timestamp')
        
        # Split per user to maintain temporal integrity
        train_list = []
        test_list = []
        
        for user_id in sorted_ratings['user_id'].unique():
            user_ratings = sorted_ratings[sorted_ratings['user_id'] == user_id]
            n_ratings = len(user_ratings)
            split_idx = int(n_ratings * (1 - test_size))
            
            train_list.append(user_ratings.iloc[:split_idx])
            test_list.append(user_ratings.iloc[split_idx:])
        
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        return train_df, test_df
    
    def get_genre_features(self) -> np.ndarray:
        """
        Extract genre feature matrix for content-based filtering.
        Shape: (n_movies, n_genres)
        """
        genre_columns = ['unknown', 'Action', 'Adventure', 'Animation',
                        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                        'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        return self.movies_df[genre_columns].values
    
    def get_user_stats(self) -> Dict:
        """Compute dataset statistics."""
        return {
            'n_users': self.ratings_df['user_id'].nunique(),
            'n_movies': self.ratings_df['movie_id'].nunique(),
            'n_ratings': len(self.ratings_df),
            'sparsity': 1 - (len(self.ratings_df) / 
                           (self.ratings_df['user_id'].nunique() * 
                            self.ratings_df['movie_id'].nunique())),
            'avg_ratings_per_user': self.ratings_df.groupby('user_id').size().mean(),
            'avg_ratings_per_movie': self.ratings_df.groupby('movie_id').size().mean()
        }
DATALOADEREOF

echo "✅ Generated utils/data_loader.py"

# Continue with other files...
# (Due to length, I'll create a script that generates the rest)

echo ""
echo "✅ Generator script updated with data_loader.py"
echo "Note: Full generator needs all file content. Creating files directly may be faster."
'''

# Append to setup.sh
with open('setup.sh', 'a') as f:
    f.write(additional_content)

print("Updated setup.sh with data_loader.py generation")
print("Due to the large size, creating files directly would be more efficient.")
