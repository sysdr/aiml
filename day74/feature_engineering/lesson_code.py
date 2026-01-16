"""
Day 74: Feature Engineering - Production-Ready Implementation
Demonstrates feature transformation techniques used in real ML systems
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, PolynomialFeatures
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineeringPipeline:
    """
    Comprehensive feature engineering pipeline for mixed data types.
    Mirrors patterns used at Netflix, Uber, and Stripe for production ML.
    """
    
    def __init__(self, scaling_strategy='standard', encoding_strategy='onehot'):
        """
        Initialize feature engineering pipeline.
        
        Args:
            scaling_strategy: 'standard', 'minmax', or 'robust'
            encoding_strategy: 'onehot' or 'label'
        """
        self.scaling_strategy = scaling_strategy
        self.encoding_strategy = encoding_strategy
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessor = None
        self.feature_names = []
        
    def analyze_features(self, X):
        """Automatically detect numeric and categorical features."""
        self.numeric_features = X.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        self.categorical_features = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        print(f"\n{'='*60}")
        print(f"Feature Analysis:")
        print(f"{'='*60}")
        print(f"Numeric features ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
    def build_preprocessor(self):
        """Build ColumnTransformer for mixed data types."""
        
        # Select scaler based on strategy
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        scaler = scalers[self.scaling_strategy]
        
        # Build numeric transformer
        numeric_transformer = Pipeline(steps=[
            ('scaler', scaler)
        ])
        
        # Build categorical transformer
        if self.encoding_strategy == 'onehot':
            categorical_transformer = Pipeline(steps=[
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        else:
            categorical_transformer = Pipeline(steps=[
                ('encoder', LabelEncoder())
            ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'
        )
        
        return self.preprocessor
    
    def fit_transform(self, X, y=None):
        """Fit and transform the data."""
        self.analyze_features(X)
        self.build_preprocessor()
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Store feature names for later use
        self._extract_feature_names()
        
        print(f"\nOriginal shape: {X.shape}")
        print(f"Transformed shape: {X_transformed.shape}")
        
        return X_transformed
    
    def transform(self, X):
        """Transform new data using fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Pipeline not fitted yet. Call fit_transform first.")
        return self.preprocessor.transform(X)
    
    def _extract_feature_names(self):
        """Extract feature names after transformation."""
        feature_names = []
        
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                if isinstance(transformer.named_steps['encoder'], OneHotEncoder):
                    cat_names = transformer.named_steps['encoder'].get_feature_names_out(features)
                    feature_names.extend(cat_names)
                else:
                    feature_names.extend(features)
        
        self.feature_names = feature_names


class PolynomialFeatureCreator:
    """
    Creates polynomial and interaction features.
    Used by Tesla for trajectory prediction and Meta for ad targeting.
    """
    
    def __init__(self, degree=2, interaction_only=False):
        """
        Initialize polynomial feature creator.
        
        Args:
            degree: Maximum degree of polynomial features
            interaction_only: If True, only create interaction terms (no x^2, x^3)
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        self.feature_names = []
        
    def fit_transform(self, X, feature_names=None):
        """Create polynomial features."""
        X_poly = self.poly.fit_transform(X)
        
        if feature_names:
            self.feature_names = self.poly.get_feature_names_out(feature_names)
        
        print(f"\n{'='*60}")
        print(f"Polynomial Feature Creation:")
        print(f"{'='*60}")
        print(f"Degree: {self.degree}, Interaction only: {self.interaction_only}")
        print(f"Original features: {X.shape[1]}")
        print(f"Polynomial features: {X_poly.shape[1]}")
        print(f"Feature explosion ratio: {X_poly.shape[1] / X.shape[1]:.2f}x")
        
        return X_poly
    
    def transform(self, X):
        """Transform new data."""
        return self.poly.transform(X)


class FeatureBinner:
    """
    Discretizes continuous features into bins.
    Used by insurance companies and financial institutions.
    """
    
    def __init__(self, n_bins=5, strategy='quantile'):
        """
        Initialize feature binner.
        
        Args:
            n_bins: Number of bins
            strategy: 'uniform', 'quantile', or 'kmeans'
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges = {}
        
    def fit_transform(self, X, feature_name):
        """Create bins for a feature."""
        if self.strategy == 'quantile':
            self.bin_edges[feature_name] = np.percentile(
                X, np.linspace(0, 100, self.n_bins + 1)
            )
        elif self.strategy == 'uniform':
            self.bin_edges[feature_name] = np.linspace(
                X.min(), X.max(), self.n_bins + 1
            )
        
        binned = np.digitize(X, self.bin_edges[feature_name][1:-1])
        
        print(f"\n{'='*60}")
        print(f"Feature Binning: {feature_name}")
        print(f"{'='*60}")
        print(f"Strategy: {self.strategy}")
        print(f"Number of bins: {self.n_bins}")
        print(f"Bin edges: {self.bin_edges[feature_name]}")
        print(f"Value distribution across bins:")
        unique, counts = np.unique(binned, return_counts=True)
        for bin_id, count in zip(unique, counts):
            print(f"  Bin {bin_id}: {count} samples ({count/len(binned)*100:.1f}%)")
        
        return binned


class FeatureSelector:
    """
    Selects most important features using statistical tests.
    Reduces dimensionality and prevents overfitting.
    """
    
    def __init__(self, k=10, score_func='f_classif'):
        """
        Initialize feature selector.
        
        Args:
            k: Number of top features to select
            score_func: 'f_classif' or 'mutual_info'
        """
        self.k = k
        score_funcs = {
            'f_classif': f_classif,
            'mutual_info': mutual_info_classif
        }
        self.selector = SelectKBest(score_func=score_funcs[score_func], k=k)
        self.feature_scores = None
        self.selected_features = None
        
    def fit_transform(self, X, y, feature_names=None):
        """Select top k features."""
        X_selected = self.selector.fit_transform(X, y)
        
        # Get feature scores
        self.feature_scores = self.selector.scores_
        
        # Get selected feature indices
        selected_indices = self.selector.get_support(indices=True)
        
        if feature_names:
            self.selected_features = [feature_names[i] for i in selected_indices]
        
        print(f"\n{'='*60}")
        print(f"Feature Selection:")
        print(f"{'='*60}")
        print(f"Original features: {X.shape[1]}")
        print(f"Selected features: {X_selected.shape[1]}")
        print(f"Reduction: {(1 - X_selected.shape[1]/X.shape[1])*100:.1f}%")
        
        if self.selected_features:
            print(f"\nTop {min(10, len(self.selected_features))} selected features:")
            for i, feat in enumerate(self.selected_features[:10], 1):
                print(f"  {i}. {feat}")
        
        return X_selected
    
    def transform(self, X):
        """Transform new data."""
        return self.selector.transform(X)


def create_sample_dataset():
    """
    Create a realistic customer churn dataset.
    Mirrors telecom/subscription business data.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Numeric features
    age = np.random.randint(18, 70, n_samples)
    tenure = np.random.randint(0, 72, n_samples)  # months
    monthly_charges = np.random.uniform(20, 120, n_samples)
    total_charges = monthly_charges * tenure + np.random.normal(0, 50, n_samples)
    
    # Categorical features
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples)
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Credit card'], n_samples)
    
    # Target: churn (higher charges + month-to-month = higher churn)
    churn_prob = (
        0.3 * (contract_types == 'Month-to-month') +
        0.2 * (monthly_charges > 80) +
        0.1 * (tenure < 12) +
        np.random.uniform(0, 0.2, n_samples)
    )
    churn = (churn_prob > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract': contract_types,
        'internet_service': internet_service,
        'payment_method': payment_method,
        'churn': churn
    })
    
    return df


def demonstrate_feature_engineering():
    """
    Complete feature engineering workflow demonstration.
    Shows the full pipeline from raw data to model training.
    """
    print("="*70)
    print("Day 74: Feature Engineering - Production Implementation")
    print("="*70)
    
    # Load data
    print("\n1. Loading Sample Dataset...")
    df = create_sample_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nChurn distribution:")
    print(df['churn'].value_counts(normalize=True))
    
    # Separate features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Split data - CRITICAL: Always split before feature engineering
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 2. Basic Feature Engineering
    print("\n" + "="*70)
    print("2. Basic Feature Scaling and Encoding")
    print("="*70)
    
    fe_pipeline = FeatureEngineeringPipeline(
        scaling_strategy='standard',
        encoding_strategy='onehot'
    )
    X_train_basic = fe_pipeline.fit_transform(X_train)
    X_test_basic = fe_pipeline.transform(X_test)
    
    # 3. Polynomial Features
    print("\n" + "="*70)
    print("3. Creating Polynomial and Interaction Features")
    print("="*70)
    
    # Only apply to numeric features to avoid explosion
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train_numeric = X_train[numeric_cols].values
    X_test_numeric = X_test[numeric_cols].values
    
    poly_creator = PolynomialFeatureCreator(degree=2, interaction_only=True)
    X_train_poly = poly_creator.fit_transform(
        X_train_numeric,
        feature_names=numeric_cols.tolist()
    )
    X_test_poly = poly_creator.transform(X_test_numeric)
    
    # 4. Feature Binning Example
    print("\n" + "="*70)
    print("4. Feature Binning Demonstration")
    print("="*70)
    
    # Extract numeric columns before binning to avoid feature mismatch
    X_train_numeric_only = X_train.select_dtypes(include=['int64', 'float64']).copy()
    X_test_numeric_only = X_test.select_dtypes(include=['int64', 'float64']).copy()
    
    binner = FeatureBinner(n_bins=5, strategy='quantile')
    X_train['monthly_charges_binned'] = binner.fit_transform(
        X_train['monthly_charges'].values,
        'monthly_charges'
    )
    
    # 5. Model Training Comparison
    print("\n" + "="*70)
    print("5. Model Performance Comparison")
    print("="*70)
    
    # Baseline: Original features
    print("\nBaseline Model (Logistic Regression on raw features):")
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Use only numeric features for baseline (use pre-extracted to avoid binned column)
    X_train_numeric_only = X_train_numeric_only.values
    X_test_numeric_only = X_test_numeric_only.values
    
    baseline_model.fit(X_train_numeric_only, y_train)
    baseline_pred = baseline_model.predict(X_test_numeric_only)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    print(f"Accuracy: {baseline_acc:.4f}")
    
    # With feature engineering
    print("\nWith Feature Engineering (scaled + encoded):")
    fe_model = LogisticRegression(max_iter=1000, random_state=42)
    fe_model.fit(X_train_basic, y_train)
    fe_pred = fe_model.predict(X_test_basic)
    fe_acc = accuracy_score(y_test, fe_pred)
    print(f"Accuracy: {fe_acc:.4f}")
    print(f"Improvement: {(fe_acc - baseline_acc)*100:.2f}%")
    
    # With polynomial features
    print("\nWith Polynomial Features:")
    poly_model = RandomForestClassifier(n_estimators=100, random_state=42)
    poly_model.fit(X_train_poly, y_train)
    poly_pred = poly_model.predict(X_test_poly)
    poly_acc = accuracy_score(y_test, poly_pred)
    print(f"Accuracy: {poly_acc:.4f}")
    print(f"Improvement over baseline: {(poly_acc - baseline_acc)*100:.2f}%")
    
    # 6. Feature Selection
    print("\n" + "="*70)
    print("6. Feature Selection")
    print("="*70)
    
    selector = FeatureSelector(k=10, score_func='f_classif')
    X_train_selected = selector.fit_transform(
        X_train_basic, y_train, fe_pipeline.feature_names
    )
    X_test_selected = selector.transform(X_test_basic)
    
    # Model with selected features
    print("\nModel with Feature Selection:")
    selected_model = LogisticRegression(max_iter=1000, random_state=42)
    selected_model.fit(X_train_selected, y_train)
    selected_pred = selected_model.predict(X_test_selected)
    selected_acc = accuracy_score(y_test, selected_pred)
    print(f"Accuracy: {selected_acc:.4f}")
    print(f"Improvement: {(selected_acc - baseline_acc)*100:.2f}%")
    
    # Final Summary
    print("\n" + "="*70)
    print("SUMMARY: Feature Engineering Impact")
    print("="*70)
    print(f"Baseline (raw numeric):        {baseline_acc:.4f}")
    print(f"With scaling + encoding:       {fe_acc:.4f}  (+{(fe_acc-baseline_acc)*100:.2f}%)")
    print(f"With polynomial features:      {poly_acc:.4f}  (+{(poly_acc-baseline_acc)*100:.2f}%)")
    print(f"With feature selection:        {selected_acc:.4f}  (+{(selected_acc-baseline_acc)*100:.2f}%)")
    print("\nKey Insight: Feature engineering often provides 10-30% performance lift")
    print("in production ML systems. At Netflix, Uber, and Stripe, feature engineering")
    print("teams are often larger than the teams building the actual models.")


if __name__ == "__main__":
    demonstrate_feature_engineering()
