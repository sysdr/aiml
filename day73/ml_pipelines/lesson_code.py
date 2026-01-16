"""
Day 73: Pipelines - Chaining Steps Together
Production-grade ML workflow automation
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib
import warnings
warnings.filterwarnings('ignore')


def demonstrate_basic_pipeline():
    """
    Basic pipeline: scaler + classifier
    Shows fundamental pipeline construction and usage
    """
    print("=" * 60)
    print("PART 1: Basic Pipeline Construction")
    print("=" * 60)
    
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split data BEFORE any preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    print("\nPipeline structure:")
    for name, step in pipeline.steps:
        print(f"  {name}: {step.__class__.__name__}")
    
    # Train pipeline (fits scaler on train data, then classifier)
    print("\nTraining pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Training accuracy: {train_score:.4f}")
    print(f"  Test accuracy: {test_score:.4f}")
    print(f"  Generalization gap: {train_score - test_score:.4f}")
    
    return pipeline


def demonstrate_data_leakage_prevention():
    """
    Compare proper pipeline usage vs data leakage scenario
    Shows why pipelines matter for correct evaluation
    """
    print("\n" + "=" * 60)
    print("PART 2: Data Leakage Prevention")
    print("=" * 60)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        random_state=42
    )
    
    # WRONG WAY: Scale before split (data leakage)
    print("\n❌ WRONG: Scaling before train/test split")
    scaler_wrong = StandardScaler()
    X_scaled_wrong = scaler_wrong.fit_transform(X)  # Sees ALL data
    X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong = train_test_split(
        X_scaled_wrong, y, test_size=0.2, random_state=42
    )
    
    clf_wrong = LogisticRegression(random_state=42, max_iter=1000)
    clf_wrong.fit(X_train_wrong, y_train_wrong)
    wrong_score = clf_wrong.score(X_test_wrong, y_test_wrong)
    print(f"  Test accuracy (with leakage): {wrong_score:.4f}")
    
    # RIGHT WAY: Pipeline prevents leakage
    print("\n✅ CORRECT: Pipeline with proper isolation")
    X_train_right, X_test_right, y_train_right, y_test_right = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    pipeline_right = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    pipeline_right.fit(X_train_right, y_train_right)  # Scaler fits only on train
    right_score = pipeline_right.score(X_test_right, y_test_right)
    print(f"  Test accuracy (no leakage): {right_score:.4f}")
    
    # Show the difference
    difference = wrong_score - right_score
    print(f"\n📊 Performance difference: {difference:.4f}")
    print(f"   Leaked model appears {difference*100:.2f}% better (falsely optimistic)")
    
    return pipeline_right


def demonstrate_column_transformer():
    """
    ColumnTransformer for heterogeneous data
    Production pattern for real-world datasets with mixed types
    """
    print("\n" + "=" * 60)
    print("PART 3: ColumnTransformer for Mixed Data Types")
    print("=" * 60)
    
    # Create realistic dataset with numerical and categorical features
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
        'employment': np.random.choice(['Full-time', 'Part-time', 'Self-employed'], n_samples),
        'owns_home': np.random.choice(['Yes', 'No'], n_samples)
    })
    
    # Add some missing values (realistic scenario)
    data.loc[np.random.choice(data.index, 50), 'income'] = np.nan
    data.loc[np.random.choice(data.index, 30), 'age'] = np.nan
    
    # Target variable (loan approval)
    y = (data['credit_score'] > 650).astype(int)
    
    print(f"\nDataset shape: {data.shape}")
    print(f"\nFeature types:")
    print(data.dtypes)
    print(f"\nMissing values:")
    print(data.isnull().sum())
    
    # Define column types
    numerical_features = ['age', 'income', 'credit_score']
    categorical_features = ['city', 'employment', 'owns_home']
    
    # Create preprocessing pipelines for each type
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Full pipeline with classifier
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("\n📋 Pipeline structure:")
    print("  1. Numerical pipeline:")
    print("     - Impute missing values (median)")
    print("     - Scale features (standardization)")
    print("  2. Categorical pipeline:")
    print("     - Impute missing values (constant)")
    print("     - One-hot encode categories")
    print("  3. Random Forest classifier")
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=42
    )
    
    print("\nTraining full pipeline...")
    full_pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = full_pipeline.score(X_train, y_train)
    test_score = full_pipeline.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Training accuracy: {train_score:.4f}")
    print(f"  Test accuracy: {test_score:.4f}")
    
    # Show transformed feature count
    X_transformed = preprocessor.fit_transform(X_train)
    print(f"\n🔄 Feature transformation:")
    print(f"  Original features: {data.shape[1]}")
    print(f"  Transformed features: {X_transformed.shape[1]}")
    print(f"  (One-hot encoding expanded categorical features)")
    
    return full_pipeline


def demonstrate_cross_validation_with_pipeline():
    """
    Cross-validation with pipelines
    Proper CV ensures no data leakage across folds
    """
    print("\n" + "=" * 60)
    print("PART 4: Cross-Validation with Pipelines")
    print("=" * 60)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    print("\nRunning 5-fold cross-validation...")
    print("(Each fold: fit scaler on train folds, transform test fold)")
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        pipeline, X, y, 
        cv=5, 
        scoring='accuracy'
    )
    
    print(f"\nCross-validation scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    
    print(f"\nMean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return pipeline, cv_scores


def demonstrate_serialization():
    """
    Pipeline serialization for production deployment
    Save trained pipeline, load later for inference
    """
    print("\n" + "=" * 60)
    print("PART 5: Pipeline Serialization")
    print("=" * 60)
    
    # Train a pipeline
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    original_predictions = pipeline.predict(X_test)
    original_score = pipeline.score(X_test, y_test)
    
    # Save pipeline
    model_filename = 'pipeline_model.pkl'
    print(f"\n💾 Saving pipeline to {model_filename}...")
    joblib.dump(pipeline, model_filename)
    
    # Load pipeline
    print(f"📂 Loading pipeline from {model_filename}...")
    loaded_pipeline = joblib.load(model_filename)
    
    # Verify loaded pipeline produces identical results
    loaded_predictions = loaded_pipeline.predict(X_test)
    loaded_score = loaded_pipeline.score(X_test, y_test)
    
    print(f"\n✅ Verification:")
    print(f"  Original accuracy: {original_score:.4f}")
    print(f"  Loaded accuracy: {loaded_score:.4f}")
    print(f"  Predictions match: {np.array_equal(original_predictions, loaded_predictions)}")
    
    return loaded_pipeline


def production_pipeline_example():
    """
    Complete production-ready pipeline example
    Demonstrates best practices used in real ML systems
    """
    print("\n" + "=" * 60)
    print("PART 6: Production Pipeline Pattern")
    print("=" * 60)
    
    # Create realistic customer churn dataset
    np.random.seed(42)
    n_samples = 2000
    
    data = pd.DataFrame({
        'account_age_months': np.random.randint(1, 120, n_samples),
        'monthly_usage_gb': np.random.lognormal(3, 1, n_samples),
        'support_tickets': np.random.poisson(2, n_samples),
        'subscription_price': np.random.choice([9.99, 14.99, 19.99, 29.99], n_samples),
        'plan_type': np.random.choice(['Basic', 'Pro', 'Enterprise'], n_samples),
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples),
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer'], n_samples)
    })
    
    # Target: customer churn
    churn_probability = (
        (data['account_age_months'] < 12) * 0.3 +
        (data['support_tickets'] > 3) * 0.2 +
        (data['monthly_usage_gb'] < 5) * 0.2 +
        np.random.random(n_samples) * 0.3
    )
    y = (churn_probability > 0.5).astype(int)
    
    print(f"\n📊 Customer Churn Dataset")
    print(f"  Total customers: {n_samples}")
    print(f"  Churn rate: {y.mean()*100:.1f}%")
    print(f"  Features: {data.shape[1]}")
    
    # Define feature groups
    numerical_features = ['account_age_months', 'monthly_usage_gb', 'support_tickets', 'subscription_price']
    categorical_features = ['plan_type', 'device_type', 'payment_method']
    
    # Build production pipeline
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Full pipeline with optimized classifier
    production_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42
        ))
    ])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("\n🔧 Training production pipeline...")
    production_pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = production_pipeline.score(X_train, y_train)
    test_score = production_pipeline.score(X_test, y_test)
    
    print(f"\n📈 Performance Metrics:")
    print(f"  Training accuracy: {train_score:.4f}")
    print(f"  Test accuracy: {test_score:.4f}")
    print(f"  Generalization: {(test_score/train_score)*100:.1f}%")
    
    # Cross-validation for robust estimate
    cv_scores = cross_val_score(
        production_pipeline, data, y, cv=5, scoring='accuracy'
    )
    print(f"\n  5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Save for deployment
    print("\n💾 Saving production model...")
    joblib.dump(production_pipeline, 'churn_model_v1.pkl')
    print("  ✅ Model saved: churn_model_v1.pkl")
    print("  Ready for deployment!")
    
    return production_pipeline


def main():
    """Run all pipeline demonstrations"""
    print("\n" + "🚀" * 30)
    print("Day 73: Scikit-learn Pipelines")
    print("Production ML Workflow Automation")
    print("🚀" * 30)
    
    # Run demonstrations
    pipeline1 = demonstrate_basic_pipeline()
    pipeline2 = demonstrate_data_leakage_prevention()
    pipeline3 = demonstrate_column_transformer()
    pipeline4, cv_scores = demonstrate_cross_validation_with_pipeline()
    pipeline5 = demonstrate_serialization()
    pipeline6 = production_pipeline_example()
    
    print("\n" + "=" * 60)
    print("✅ All demonstrations complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Pipelines prevent data leakage by fitting transformers only on training data")
    print("2. ColumnTransformer handles mixed data types elegantly")
    print("3. Cross-validation with pipelines ensures proper fold isolation")
    print("4. Serialization makes deployment reproducible")
    print("5. Production pipelines chain preprocessing and models into single objects")
    print("\n💡 Next: Feature Engineering (Day 74)")


if __name__ == "__main__":
    main()
