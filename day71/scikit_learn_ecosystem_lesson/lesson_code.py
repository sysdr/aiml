"""
Day 71: The Scikit-learn Ecosystem
Production-grade ML pipeline demonstrating the complete scikit-learn architecture
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')


class MovieRecommendationDataset:
    """Generate synthetic movie rating dataset similar to MovieLens"""
    
    def __init__(self, n_users=1000, n_movies=500, n_ratings=50000, random_state=42):
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_ratings = n_ratings
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate(self):
        """Create synthetic rating data with realistic patterns"""
        
        # Generate user IDs and movie IDs
        user_ids = np.random.randint(0, self.n_users, self.n_ratings)
        movie_ids = np.random.randint(0, self.n_movies, self.n_ratings)
        
        # Create user and movie latent factors (simulating preferences)
        user_factors = np.random.randn(self.n_users, 10)
        movie_factors = np.random.randn(self.n_movies, 10)
        
        # Generate ratings based on latent factor interaction
        ratings = []
        timestamps = []
        base_time = 1609459200  # 2021-01-01
        
        for user_id, movie_id in zip(user_ids, movie_ids):
            # Base rating from latent factors
            base_rating = np.dot(user_factors[user_id], movie_factors[movie_id])
            
            # Add noise and clip to 1-5 range
            rating = np.clip(base_rating + np.random.randn() * 0.5, 1, 5)
            ratings.append(rating)
            
            # Generate timestamp (within 2 years)
            timestamp = base_time + np.random.randint(0, 63072000)
            timestamps.append(timestamp)
        
        # Create DataFrame
        df = pd.DataFrame({
            'user_id': user_ids,
            'movie_id': movie_ids,
            'rating': ratings,
            'timestamp': timestamps
        })
        
        # Add movie metadata
        movie_genres = np.random.choice(
            ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance', 'Horror'],
            size=self.n_movies
        )
        movie_years = np.random.randint(1980, 2024, size=self.n_movies)
        
        # Add user metadata
        user_ages = np.random.randint(18, 70, size=self.n_users)
        user_countries = np.random.choice(
            ['US', 'UK', 'CA', 'AU', 'DE', 'FR'],
            size=self.n_users
        )
        
        # Merge metadata
        df['genre'] = df['movie_id'].map(lambda x: movie_genres[x])
        df['movie_year'] = df['movie_id'].map(lambda x: movie_years[x])
        df['user_age'] = df['user_id'].map(lambda x: user_ages[x])
        df['user_country'] = df['user_id'].map(lambda x: user_countries[x])
        
        return df


class UserMovieFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that creates interaction features between users and movies.
    This demonstrates how to extend sklearn's ecosystem with domain-specific logic.
    """
    
    def __init__(self, create_temporal=True):
        self.create_temporal = create_temporal
        self.user_avg_ratings = {}
        self.movie_avg_ratings = {}
        
    def fit(self, X, y=None):
        """Learn user and movie statistics from training data"""
        df = X.copy()
        
        # Calculate average ratings per user
        self.user_avg_ratings = df.groupby('user_id')['rating'].mean().to_dict()
        
        # Calculate average ratings per movie
        self.movie_avg_ratings = df.groupby('movie_id')['rating'].mean().to_dict()
        
        return self
    
    def transform(self, X):
        """Create engineered features"""
        df = X.copy()
        
        # User deviation from average
        df['user_avg_rating'] = df['user_id'].map(
            lambda x: self.user_avg_ratings.get(x, df['rating'].mean())
        )
        
        # Movie deviation from average
        df['movie_avg_rating'] = df['movie_id'].map(
            lambda x: self.movie_avg_ratings.get(x, df['rating'].mean())
        )
        
        # Interaction: user tendency × movie quality
        df['user_movie_interaction'] = df['user_avg_rating'] * df['movie_avg_rating']
        
        if self.create_temporal:
            # Extract temporal features
            df['hour_of_day'] = (df['timestamp'] % 86400) // 3600
            df['day_of_week'] = (df['timestamp'] // 86400) % 7
        
        # Rating count features
        user_counts = df['user_id'].value_counts()
        df['user_rating_count'] = df['user_id'].map(user_counts)
        
        movie_counts = df['movie_id'].value_counts()
        df['movie_rating_count'] = df['movie_id'].map(movie_counts)
        
        return df


class SklearnEcosystemPipeline:
    """
    Production-grade ML pipeline demonstrating all components of sklearn ecosystem:
    - Preprocessing (scaling, encoding)
    - Feature engineering (custom transformers)
    - Model selection (cross-validation, hyperparameter tuning)
    - Evaluation (multiple metrics)
    - Deployment (pipeline serialization)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None
        self.best_params = None
        self.cv_scores = None
        
    def create_features(self, df):
        """Prepare features for modeling"""
        
        # Apply custom transformer
        feature_transformer = UserMovieFeatureTransformer()
        df = feature_transformer.fit_transform(df)
        
        # Encode categorical variables
        le_genre = LabelEncoder()
        le_country = LabelEncoder()
        
        df['genre_encoded'] = le_genre.fit_transform(df['genre'])
        df['country_encoded'] = le_country.fit_transform(df['user_country'])
        
        # Select features for modeling
        feature_cols = [
            'user_id', 'movie_id', 'user_age', 'movie_year',
            'genre_encoded', 'country_encoded',
            'user_avg_rating', 'movie_avg_rating', 'user_movie_interaction',
            'hour_of_day', 'day_of_week',
            'user_rating_count', 'movie_rating_count'
        ]
        
        X = df[feature_cols]
        y = df['rating']
        
        return X, y, feature_transformer, le_genre, le_country
    
    def build_pipeline(self):
        """
        Construct sklearn Pipeline with preprocessing and model.
        This is the production pattern - everything in one object.
        """
        
        pipeline = Pipeline([
            # Step 1: Scale numerical features
            ('scaler', StandardScaler()),
            
            # Step 2: Train model
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])
        
        return pipeline
    
    def train(self, X, y, perform_cv=True, tune_hyperparameters=False):
        """Train the pipeline with optional cross-validation and tuning"""
        
        print("Building pipeline...")
        self.pipeline = self.build_pipeline()
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [5, 10, 15],
                'model__min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            self.pipeline = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self.best_params}")
        else:
            self.pipeline.fit(X, y)
        
        if perform_cv:
            print("Performing cross-validation...")
            cv_scores = cross_val_score(
                self.pipeline, X, y,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            self.cv_scores = np.sqrt(-cv_scores)
            print(f"CV RMSE: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})")
        
        return self.pipeline
    
    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation using sklearn.metrics"""
        
        y_pred = self.pipeline.predict(X_test)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def save_pipeline(self, filepath='production_pipeline.pkl'):
        """Save complete pipeline for production deployment"""
        joblib.dump(self.pipeline, filepath)
        print(f"Pipeline saved to {filepath}")
        return filepath
    
    @staticmethod
    def load_pipeline(filepath='production_pipeline.pkl'):
        """Load pipeline from disk"""
        return joblib.load(filepath)


def demonstrate_ecosystem():
    """
    Complete demonstration of scikit-learn ecosystem in production:
    1. Data generation
    2. Feature engineering with custom transformers
    3. Pipeline construction
    4. Training with cross-validation
    5. Evaluation with multiple metrics
    6. Production export
    """
    
    print("=" * 60)
    print("Scikit-learn Ecosystem: Production ML Pipeline Demo")
    print("=" * 60)
    print()
    
    # Step 1: Generate synthetic movie rating dataset
    print("Step 1: Generating movie rating dataset...")
    dataset = MovieRecommendationDataset(
        n_users=1000,
        n_movies=500,
        n_ratings=50000
    )
    df = dataset.generate()
    print(f"✓ Dataset created: {len(df)} ratings from {df['user_id'].nunique()} users")
    print(f"  Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")
    print(f"  Average rating: {df['rating'].mean():.2f}")
    print()
    
    # Step 2: Feature engineering
    print("Step 2: Engineering features...")
    ml_pipeline = SklearnEcosystemPipeline()
    X, y, feature_transformer, le_genre, le_country = ml_pipeline.create_features(df)
    print(f"✓ Features created: {X.shape[1]} features from {X.shape[0]} samples")
    print(f"  Feature columns: {list(X.columns)[:5]}...")
    print()
    
    # Step 3: Train-test split
    print("Step 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print()
    
    # Step 4: Build and train pipeline
    print("Step 4: Training pipeline with cross-validation...")
    ml_pipeline.train(X_train, y_train, perform_cv=True, tune_hyperparameters=False)
    print("✓ Training complete")
    print()
    
    # Step 5: Evaluate on test set
    print("Step 5: Evaluating on test set...")
    metrics, predictions = ml_pipeline.evaluate(X_test, y_test)
    print("Test Set Performance:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print()
    
    # Step 6: Save for production
    print("Step 6: Saving pipeline for production deployment...")
    pipeline_path = ml_pipeline.save_pipeline()
    print(f"✓ Pipeline serialized to {pipeline_path}")
    print()
    
    # Step 7: Demonstrate production usage
    print("Step 7: Simulating production inference...")
    loaded_pipeline = SklearnEcosystemPipeline.load_pipeline(pipeline_path)
    
    # Make predictions on new data
    sample_data = X_test.head(5)
    sample_predictions = loaded_pipeline.predict(sample_data)
    
    print("Sample predictions:")
    for i, (idx, row) in enumerate(sample_data.iterrows()):
        print(f"  User {int(row['user_id'])}, Movie {int(row['movie_id'])}: "
              f"Predicted {sample_predictions[i]:.2f}, Actual {y_test.iloc[i]:.2f}")
    print()
    
    print("=" * 60)
    print("Ecosystem Demonstration Complete!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("✓ Custom transformers extend sklearn seamlessly")
    print("✓ Pipelines bundle preprocessing + modeling atomically")
    print("✓ Cross-validation ensures generalization")
    print("✓ Serialization enables production deployment")
    print("✓ Consistent API across all components")
    print()
    
    return {
        'pipeline': ml_pipeline,
        'metrics': metrics,
        'data': (X_train, X_test, y_train, y_test)
    }


if __name__ == '__main__':
    results = demonstrate_ecosystem()
