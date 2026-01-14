"""
Day 72: Data Preprocessing and Feature Scaling
Production-grade preprocessing pipeline for real-time AI systems
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Production-grade data preprocessing system.
    Handles missing data, categorical encoding, and feature scaling.
    
    Design Pattern: Fit on training data, transform on all data
    """
    
    def __init__(self, strategy='robust'):
        """
        Initialize preprocessor with scaling strategy.
        
        Args:
            strategy: 'standard', 'minmax', or 'robust'
        """
        self.strategy = strategy
        self.preprocessing_pipeline = None
        self.feature_names = None
        self.fitted = False
        
    def create_pipeline(self, numerical_features, categorical_features):
        """
        Build preprocessing pipeline with separate transformers for different feature types.
        
        This mirrors production systems at Netflix, Spotify where different
        feature types require different preprocessing strategies.
        """
        
        # Numerical pipeline: impute missing values, then scale
        if self.strategy == 'standard':
            scaler = StandardScaler()
        elif self.strategy == 'minmax':
            scaler = MinMaxScaler()
        else:  # robust
            scaler = RobustScaler()
            
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Median handles outliers better
            ('scaler', scaler)
        ])
        
        # Categorical pipeline: impute missing values, then encode
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        self.preprocessing_pipeline = ColumnTransformer([
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        return self.preprocessing_pipeline
    
    def fit(self, X, numerical_features, categorical_features):
        """
        Fit preprocessing pipeline on training data.
        Learns scaling parameters and categorical mappings.
        
        CRITICAL: Only call this on training data to prevent data leakage.
        """
        if self.preprocessing_pipeline is None:
            self.create_pipeline(numerical_features, categorical_features)
        
        self.preprocessing_pipeline.fit(X)
        self.fitted = True
        
        # Store feature names for debugging
        self.feature_names = self._get_feature_names(
            numerical_features, categorical_features
        )
        
        return self
    
    def transform(self, X):
        """
        Transform data using fitted parameters.
        Can be called on training, validation, or production data.
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        return self.preprocessing_pipeline.transform(X)
    
    def fit_transform(self, X, numerical_features, categorical_features):
        """Convenience method: fit and transform in one call."""
        self.fit(X, numerical_features, categorical_features)
        return self.transform(X)
    
    def _get_feature_names(self, numerical_features, categorical_features):
        """Extract feature names after transformation."""
        feature_names = []
        
        # Numerical features keep their names
        feature_names.extend(numerical_features)
        
        # Categorical features expand to one-hot encoded columns
        cat_encoder = self.preprocessing_pipeline.named_transformers_['cat']
        onehot = cat_encoder.named_steps['onehot']
        
        for i, cat_feature in enumerate(categorical_features):
            categories = onehot.categories_[i]
            feature_names.extend([f"{cat_feature}_{cat}" for cat in categories])
        
        return feature_names
    
    def save(self, filepath):
        """Save fitted preprocessor for production deployment."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        joblib.dump(self.preprocessing_pipeline, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load fitted preprocessor from disk."""
        pipeline = joblib.load(filepath)
        processor = DataPreprocessor()
        processor.preprocessing_pipeline = pipeline
        processor.fitted = True
        return processor


class MissingDataHandler:
    """
    Handles different strategies for missing data.
    Production systems need robust missing data handling.
    """
    
    @staticmethod
    def analyze_missing(df):
        """
        Analyze missing data patterns.
        Returns percentage of missing values per column.
        """
        missing_stats = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum(),
            'missing_percentage': (df.isnull().sum() / len(df)) * 100
        })
        
        missing_stats = missing_stats[missing_stats['missing_count'] > 0]
        missing_stats = missing_stats.sort_values('missing_percentage', ascending=False)
        
        return missing_stats
    
    @staticmethod
    def create_missing_indicator(df, columns):
        """
        Create binary indicators for missing values.
        
        Example: If 'age' is missing, create 'age_was_missing' column.
        LinkedIn uses this pattern - missingness itself can be predictive.
        """
        df_with_indicators = df.copy()
        
        for col in columns:
            if df[col].isnull().any():
                df_with_indicators[f'{col}_was_missing'] = df[col].isnull().astype(int)
        
        return df_with_indicators
    
    @staticmethod
    def impute_simple(df, numerical_strategy='median', categorical_strategy='most_frequent'):
        """
        Simple imputation: fill with statistical value.
        Fast and effective for most use cases.
        """
        df_imputed = df.copy()
        
        # Convert None to NaN for proper handling
        df_imputed = df_imputed.replace([None], np.nan)
        
        # Numerical columns
        numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            # Filter out columns that are completely missing
            numerical_cols_with_data = [col for col in numerical_cols if df_imputed[col].notna().any()]
            numerical_cols_all_missing = [col for col in numerical_cols if col not in numerical_cols_with_data]
            
            if len(numerical_cols_with_data) > 0:
                imputer = SimpleImputer(strategy=numerical_strategy)
                df_imputed[numerical_cols_with_data] = imputer.fit_transform(df_imputed[numerical_cols_with_data])
            
            # Fill completely missing columns with 0
            for col in numerical_cols_all_missing:
                df_imputed[col] = 0
        
        # Categorical columns
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            # Filter out columns that are completely missing
            categorical_cols_with_data = [col for col in categorical_cols if df_imputed[col].notna().any()]
            categorical_cols_all_missing = [col for col in categorical_cols if col not in categorical_cols_with_data]
            
            if len(categorical_cols_with_data) > 0:
                imputer = SimpleImputer(strategy=categorical_strategy)
                df_imputed[categorical_cols_with_data] = imputer.fit_transform(df_imputed[categorical_cols_with_data])
            
            # Fill completely missing columns with 'missing'
            for col in categorical_cols_all_missing:
                df_imputed[col] = 'missing'
        
        return df_imputed
    
    @staticmethod
    def impute_knn(df, n_neighbors=5):
        """
        KNN imputation: estimate missing values using similar samples.
        More sophisticated but slower. Used for critical features.
        """
        # Only works on numerical data
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return df
        
        df_imputed = df.copy()
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        return df_imputed


class FeatureScalingComparison:
    """
    Compare different scaling strategies on same data.
    Helps understand which scaler works best for your use case.
    """
    
    def __init__(self):
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler()
        }
        self.results = {}
    
    def fit_all(self, X):
        """Fit all scalers on data."""
        for name, scaler in self.scalers.items():
            scaler.fit(X)
    
    def transform_all(self, X):
        """Transform data with all scalers and store results."""
        for name, scaler in self.scalers.items():
            self.results[name] = scaler.transform(X)
        
        return self.results
    
    def compare_statistics(self, feature_index=0):
        """
        Compare statistics of scaled features.
        Returns mean, std, min, max for each scaling method.
        """
        stats = {}
        
        for name, scaled_data in self.results.items():
            feature = scaled_data[:, feature_index]
            stats[name] = {
                'mean': np.mean(feature),
                'std': np.std(feature),
                'min': np.min(feature),
                'max': np.max(feature)
            }
        
        return pd.DataFrame(stats).T


class CategoricalEncoder:
    """
    Handles different categorical encoding strategies.
    Production choice depends on cardinality and target relationship.
    """
    
    @staticmethod
    def onehot_encode(df, columns, drop_first=False):
        """
        One-hot encoding: create binary column for each category.
        
        Use when: Low cardinality (<10 categories), no ordinal relationship
        Used by: Airbnb (property type), Uber (payment method)
        """
        df_encoded = df.copy()
        
        for col in columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(col, axis=1)
        
        return df_encoded
    
    @staticmethod
    def label_encode(df, columns):
        """
        Label encoding: map categories to integers.
        
        WARNING: Creates false ordinal relationships.
        Only use for tree-based models or ordinal features.
        """
        df_encoded = df.copy()
        encoders = {}
        
        for col in columns:
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder
        
        return df_encoded, encoders
    
    @staticmethod
    def frequency_encode(df, columns):
        """
        Frequency encoding: replace category with its frequency.
        
        Use when: High cardinality features
        Example: Instagram hashtags, LinkedIn job titles
        """
        df_encoded = df.copy()
        
        for col in columns:
            frequencies = df[col].value_counts(normalize=True)
            df_encoded[col] = df[col].map(frequencies)
        
        return df_encoded
    
    @staticmethod
    def target_encode(df, column, target, smoothing=1.0):
        """
        Target encoding: replace category with target mean.
        
        Use when: High cardinality + strong target relationship
        Used by: Stripe (merchant category fraud rates)
        
        Smoothing prevents overfitting on rare categories.
        """
        # Calculate global mean
        global_mean = target.mean()
        
        # Calculate category means
        category_means = df.groupby(column)[target.name].agg(['mean', 'count'])
        
        # Apply smoothing: blend category mean with global mean
        # More samples = trust category mean more
        smoothed_means = (
            category_means['mean'] * category_means['count'] + 
            global_mean * smoothing
        ) / (category_means['count'] + smoothing)
        
        # Map to dataframe
        df_encoded = df.copy()
        df_encoded[f'{column}_encoded'] = df[column].map(smoothed_means)
        
        return df_encoded


def create_sample_dataset(n_samples=1000):
    """
    Create realistic sample dataset with various data quality issues.
    Simulates user behavior data from streaming platform.
    """
    np.random.seed(42)
    
    # Numerical features
    age = np.random.normal(35, 15, n_samples).clip(18, 80)
    session_duration = np.random.exponential(30, n_samples).clip(1, 200)
    watch_hours = np.random.normal(25, 10, n_samples).clip(0, 100)
    last_login_days = np.random.exponential(5, n_samples).clip(0, 90)
    
    # Categorical features
    device_types = np.random.choice(['mobile', 'desktop', 'tablet', 'tv'], n_samples, 
                                    p=[0.5, 0.3, 0.15, 0.05])
    subscription_tier = np.random.choice(['free', 'basic', 'premium'], n_samples,
                                         p=[0.4, 0.35, 0.25])
    region = np.random.choice(['US', 'EU', 'ASIA', 'OTHER'], n_samples,
                              p=[0.4, 0.3, 0.2, 0.1])
    
    # Boolean feature
    premium_user = (subscription_tier == 'premium').astype(int)
    
    # Create dataframe
    df = pd.DataFrame({
        'age': age,
        'device_type': device_types,
        'session_duration': session_duration,
        'watch_hours': watch_hours,
        'last_login_days': last_login_days,
        'subscription_tier': subscription_tier,
        'region': region,
        'premium_user': premium_user
    })
    
    # Introduce missing values (realistic patterns)
    # Age: 5% missing (users skip demographic info)
    df.loc[np.random.choice(df.index, int(0.05 * n_samples)), 'age'] = np.nan
    
    # Last login: 10% missing (new users)
    df.loc[np.random.choice(df.index, int(0.10 * n_samples)), 'last_login_days'] = np.nan
    
    # Watch hours: 3% missing (data collection issues)
    df.loc[np.random.choice(df.index, int(0.03 * n_samples)), 'watch_hours'] = np.nan
    
    # Add outliers to session_duration (bot behavior)
    outlier_indices = np.random.choice(df.index, 20)
    df.loc[outlier_indices, 'session_duration'] = np.random.uniform(500, 1000, 20)
    
    return df


def demonstrate_preprocessing():
    """
    Complete demonstration of preprocessing pipeline.
    Shows production patterns used by major tech companies.
    """
    print("=" * 70)
    print("DAY 72: DATA PREPROCESSING AND FEATURE SCALING")
    print("Building Netflix-Grade Preprocessing Pipeline")
    print("=" * 70)
    
    # Step 1: Generate realistic dataset
    print("\n[STEP 1] Generating sample dataset (n=1000 users)")
    df = create_sample_dataset(1000)
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Step 2: Analyze missing data
    print("\n[STEP 2] Analyzing Missing Data Patterns")
    handler = MissingDataHandler()
    missing_stats = handler.analyze_missing(df)
    if len(missing_stats) > 0:
        print("\nMissing value analysis:")
        print(missing_stats.to_string())
    else:
        print("No missing values detected")
    
    # Step 3: Create missing indicators
    print("\n[STEP 3] Creating Missing Value Indicators")
    df_with_indicators = handler.create_missing_indicator(
        df, ['age', 'last_login_days', 'watch_hours']
    )
    new_cols = [col for col in df_with_indicators.columns if col not in df.columns]
    if new_cols:
        print(f"Added indicator columns: {new_cols}")
    
    # Step 4: Split data
    print("\n[STEP 4] Splitting Data (Train/Test)")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    
    # Step 5: Define feature types
    numerical_features = ['age', 'session_duration', 'watch_hours', 'last_login_days']
    categorical_features = ['device_type', 'subscription_tier', 'region']
    
    print(f"\nNumerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Step 6: Compare scaling strategies
    print("\n[STEP 5] Comparing Scaling Strategies")
    print("Testing: StandardScaler, MinMaxScaler, RobustScaler")
    
    # Impute missing values first (required for scaling comparison)
    train_imputed = handler.impute_simple(train_df[numerical_features])
    
    scaler_comparison = FeatureScalingComparison()
    scaler_comparison.fit_all(train_imputed)
    scaler_comparison.transform_all(train_imputed)
    
    print("\nStatistics for 'session_duration' after scaling:")
    stats = scaler_comparison.compare_statistics(feature_index=1)
    print(stats)
    
    # Step 7: Build production pipeline
    print("\n[STEP 6] Building Production Pipeline (RobustScaler)")
    preprocessor = DataPreprocessor(strategy='robust')
    
    X_train_transformed = preprocessor.fit_transform(
        train_df, numerical_features, categorical_features
    )
    
    X_test_transformed = preprocessor.transform(test_df)
    
    print(f"\nTransformed training data shape: {X_train_transformed.shape}")
    print(f"Transformed test data shape: {X_test_transformed.shape}")
    print(f"\nOriginal features: {len(numerical_features) + len(categorical_features)}")
    print(f"After one-hot encoding: {X_train_transformed.shape[1]} features")
    
    # Step 8: Save pipeline
    print("\n[STEP 7] Saving Pipeline for Production")
    preprocessor.save('preprocessor_pipeline.pkl')
    
    # Demonstrate loading
    loaded_preprocessor = DataPreprocessor.load('preprocessor_pipeline.pkl')
    print("Pipeline loaded successfully - ready for production inference")
    
    # Step 9: Categorical encoding demonstration
    print("\n[STEP 8] Categorical Encoding Strategies")
    encoder = CategoricalEncoder()
    
    # One-hot encoding
    df_onehot = encoder.onehot_encode(df[['device_type']].copy(), ['device_type'])
    print(f"\nOne-hot encoding 'device_type':")
    print(f"Original: 1 column")
    print(f"After encoding: {df_onehot.shape[1]} columns")
    print(f"Columns: {df_onehot.columns.tolist()}")
    
    # Frequency encoding
    df_freq = encoder.frequency_encode(df[['device_type']].copy(), ['device_type'])
    print(f"\nFrequency encoding 'device_type':")
    print(df_freq['device_type'].value_counts().head())
    
    print("\n" + "=" * 70)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("=" * 70)
    print("\n✓ Missing data handled with multiple strategies")
    print("✓ Features scaled using robust scaler (resistant to outliers)")
    print("✓ Categorical variables properly encoded")
    print("✓ Pipeline saved for production deployment")
    print("\nNext: Day 73 - Chaining these steps into scikit-learn Pipelines")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--mode' and sys.argv[2] == 'explore':
        # Quick exploration mode
        df = create_sample_dataset(100)
        print(df.head(10))
        print("\nDataset Info:")
        print(df.info())
        print("\nMissing Values:")
        print(df.isnull().sum())
    else:
        # Full demonstration
        demonstrate_preprocessing()
