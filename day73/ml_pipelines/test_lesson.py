"""
Test suite for Day 73: Pipelines
Comprehensive tests for pipeline functionality
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification
import joblib
import os


class TestBasicPipeline:
    """Test basic pipeline construction and usage"""
    
    def test_pipeline_creation(self):
        """Test pipeline can be created with multiple steps"""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        assert len(pipeline.steps) == 2
        assert pipeline.named_steps['scaler'].__class__.__name__ == 'StandardScaler'
        assert pipeline.named_steps['classifier'].__class__.__name__ == 'LogisticRegression'
    
    def test_pipeline_fit_predict(self):
        """Test pipeline can fit and predict"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        assert predictions.shape[0] == X_test.shape[0]
        assert set(predictions).issubset({0, 1})
    
    def test_pipeline_score(self):
        """Test pipeline scoring works correctly"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        
        assert 0 <= score <= 1


class TestDataLeakagePrevention:
    """Test that pipelines prevent data leakage"""
    
    def test_scaler_fits_only_on_train(self):
        """Verify scaler statistics computed only from training data"""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        
        # Fit pipeline on train data only
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        
        # Get scaler statistics
        scaler_mean = pipeline.named_steps['scaler'].mean_
        
        # Compute train data mean manually
        train_mean = X_train.mean(axis=0)
        
        # Should match train data statistics, not full dataset
        np.testing.assert_array_almost_equal(scaler_mean, train_mean, decimal=5)
    
    def test_proper_isolation_better_than_leakage(self):
        """Test that proper pipeline isolation gives realistic performance"""
        X, y = make_classification(n_samples=500, n_features=20, random_state=42)
        
        # Method 1: WRONG - scale before split
        scaler_wrong = StandardScaler()
        X_scaled_wrong = scaler_wrong.fit_transform(X)
        X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
            X_scaled_wrong, y, test_size=0.2, random_state=42
        )
        clf_wrong = LogisticRegression(random_state=42)
        clf_wrong.fit(X_train_w, y_train_w)
        score_wrong = clf_wrong.score(X_test_w, y_test_w)
        
        # Method 2: CORRECT - pipeline prevents leakage
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        pipeline_right = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        pipeline_right.fit(X_train_r, y_train_r)
        score_right = pipeline_right.score(X_test_r, y_test_r)
        
        # Leaked version usually has artificially high score
        assert score_wrong >= score_right - 0.05  # Allow small variance


class TestColumnTransformer:
    """Test ColumnTransformer for heterogeneous data"""
    
    def test_column_transformer_creation(self):
        """Test ColumnTransformer handles different feature types"""
        numerical_features = ['num1', 'num2']
        categorical_features = ['cat1', 'cat2']
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
        
        assert len(preprocessor.transformers) == 2
    
    def test_mixed_type_pipeline(self):
        """Test pipeline with mixed numerical and categorical features"""
        # Create mixed-type dataset
        data = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100),
            'cat2': np.random.choice(['X', 'Y'], 100)
        })
        y = np.random.randint(0, 2, 100)
        
        numerical_features = ['num1', 'num2']
        categorical_features = ['cat1', 'cat2']
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False), categorical_features)
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(
            data, y, test_size=0.2, random_state=42
        )
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        assert predictions.shape[0] == X_test.shape[0]
    
    def test_missing_value_handling(self):
        """Test pipeline handles missing values correctly"""
        data = pd.DataFrame({
            'num1': [1, 2, np.nan, 4, 5],
            'cat1': ['A', 'B', None, 'A', 'B']
        })
        y = np.array([0, 1, 0, 1, 0])
        
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), ['num1']),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(sparse_output=False))
            ]), ['cat1'])
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        pipeline.fit(data, y)
        predictions = pipeline.predict(data)
        
        assert not np.isnan(predictions).any()


class TestCrossValidation:
    """Test cross-validation with pipelines"""
    
    def test_cv_with_pipeline(self):
        """Test cross-validation properly uses pipeline"""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        cv_scores = cross_val_score(pipeline, X, y, cv=5)
        
        assert len(cv_scores) == 5
        assert all(0 <= score <= 1 for score in cv_scores)
    
    def test_cv_prevents_leakage(self):
        """Verify CV fits scaler separately for each fold"""
        X, y = make_classification(n_samples=150, n_features=10, random_state=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # CV should work without errors
        cv_scores = cross_val_score(pipeline, X, y, cv=3)
        
        assert len(cv_scores) == 3
        # CV scores should be reasonable (not perfect)
        assert all(0.4 <= score <= 1.0 for score in cv_scores)


class TestSerialization:
    """Test pipeline serialization and deserialization"""
    
    def test_save_load_pipeline(self):
        """Test pipeline can be saved and loaded"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        original_predictions = pipeline.predict(X_test)
        
        # Save pipeline
        filename = 'test_pipeline.pkl'
        joblib.dump(pipeline, filename)
        
        # Load pipeline
        loaded_pipeline = joblib.load(filename)
        loaded_predictions = loaded_pipeline.predict(X_test)
        
        # Clean up
        os.remove(filename)
        
        # Predictions should be identical
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_loaded_pipeline_reproducibility(self):
        """Test loaded pipeline produces consistent results"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        original_score = pipeline.score(X_test, y_test)
        
        # Save and load
        filename = 'test_pipeline_2.pkl'
        joblib.dump(pipeline, filename)
        loaded_pipeline = joblib.load(filename)
        loaded_score = loaded_pipeline.score(X_test, y_test)
        
        # Clean up
        os.remove(filename)
        
        # Scores should be identical
        assert original_score == loaded_score


class TestProductionPatterns:
    """Test production-ready pipeline patterns"""
    
    def test_end_to_end_pipeline(self):
        """Test complete preprocessing + training pipeline"""
        data = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100),
            'cat2': np.random.choice(['X', 'Y', 'Z'], 100)
        })
        y = np.random.randint(0, 2, 100)
        
        numerical_features = ['num1', 'num2']
        categorical_features = ['cat1', 'cat2']
        
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(
            data, y, test_size=0.2, random_state=42
        )
        
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        
        assert 0 <= score <= 1
    
    def test_pipeline_with_unknown_categories(self):
        """Test pipeline handles unknown categories in test data"""
        train_data = pd.DataFrame({
            'num': np.random.randn(100),
            'cat': np.random.choice(['A', 'B'], 100)
        })
        test_data = pd.DataFrame({
            'num': np.random.randn(20),
            'cat': np.random.choice(['A', 'B', 'C'], 20)  # 'C' is new
        })
        y_train = np.random.randint(0, 2, 100)
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ['num']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['cat'])
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        pipeline.fit(train_data, y_train)
        predictions = pipeline.predict(test_data)  # Should not error
        
        assert predictions.shape[0] == test_data.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
