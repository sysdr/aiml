"""
Day 76-84: End-to-End ML Pipeline
Building a Production-Ready ML System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """
    Validates input data for ML pipeline.
    Ensures data quality before processing.
    """
    
    def __init__(self, required_columns: List[str]):
        self.required_columns = required_columns
        self.validation_rules = {
            'Age': {'min': 0, 'max': 120},
            'Fare': {'min': 0},
            'SibSp': {'min': 0},
            'Parch': {'min': 0}
        }
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate dataframe meets requirements"""
        # Check for required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if 'Age' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['Age']):
                raise TypeError("Age must be numeric")
        
        if 'Fare' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['Fare']):
                raise TypeError("Fare must be numeric")
        
        # Validate ranges
        for col, rules in self.validation_rules.items():
            if col in df.columns:
                valid_data = df[col].dropna()
                if 'min' in rules and (valid_data < rules['min']).any():
                    raise ValueError(f"{col} contains values below minimum {rules['min']}")
                if 'max' in rules and (valid_data > rules['max']).any():
                    raise ValueError(f"{col} contains values above maximum {rules['max']}")
        
        return df
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality metrics"""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        return report


class FeatureTransformer:
    """
    Handles feature engineering and preprocessing.
    Learns transformations from training data, applies to new data.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputation_values = {}
        self.feature_names = []
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, target_col: str = 'Survived') -> 'FeatureTransformer':
        """Learn preprocessing parameters from training data"""
        df = df.copy()
        
        # Separate features and target
        if target_col in df.columns:
            df = df.drop(columns=[target_col])
        
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Learn imputation values (median for numerical, mode for categorical)
        for col in numerical_cols:
            self.imputation_values[col] = df[col].median()
        
        for col in categorical_cols:
            self.imputation_values[col] = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        
        # Fit scalers for numerical columns
        for col in numerical_cols:
            scaler = StandardScaler()
            filled_data = df[col].fillna(self.imputation_values[col])
            scaler.fit(filled_data.values.reshape(-1, 1))
            self.scalers[col] = scaler
        
        # Fit encoders for categorical columns
        for col in categorical_cols:
            encoder = LabelEncoder()
            filled_data = df[col].fillna(self.imputation_values[col])
            encoder.fit(filled_data)
            self.encoders[col] = encoder
        
        self.feature_names = numerical_cols + categorical_cols
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformations to data"""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        df = df.copy()
        
        # Apply imputation
        for col, value in self.imputation_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
        
        # Scale numerical features
        for col, scaler in self.scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[col].values.reshape(-1, 1))
        
        # Encode categorical features
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        
        return df[self.feature_names]
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Survived') -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(df, target_col)
        return self.transform(df)


class MLPipeline:
    """
    Complete end-to-end ML pipeline.
    Orchestrates data validation, transformation, training, and prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validator = DataValidator(config['required_columns'])
        self.transformer = FeatureTransformer()
        self.model = None
        self.metrics = {}
        self.is_trained = False
    
    def load_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples from {filepath}")
        
        # Validate data
        df = self.validator.validate(df)
        
        # Get data quality report
        quality_report = self.validator.get_data_quality_report(df)
        print(f"Data quality check: {quality_report['total_rows']} rows, {quality_report['duplicate_rows']} duplicates")
        
        return df, quality_report
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for training or prediction"""
        df = df.copy()
        
        # Extract target if present
        target = df[self.config['target_column']] if self.config['target_column'] in df.columns else None
        
        # Select feature columns
        feature_cols = [col for col in self.config['feature_columns'] if col in df.columns]
        df_features = df[feature_cols]
        
        # Transform features
        if is_training:
            df_transformed = self.transformer.fit_transform(df_features, self.config['target_column'])
        else:
            df_transformed = self.transformer.transform(df_features)
        
        return df_transformed, target
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train model with cross-validation"""
        print("\n=== Training ML Pipeline ===")
        
        # Prepare features
        X, y = self.prepare_features(df, is_training=True)
        print(f"Feature matrix shape: {X.shape}")
        
        # Split data
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        
        # Check if we can use stratified split (requires at least 2 samples per class)
        use_stratify = True
        if y is not None:
            class_counts = y.value_counts() if hasattr(y, 'value_counts') else pd.Series(y).value_counts()
            if len(class_counts) > 0 and class_counts.min() < 2:
                use_stratify = False
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if use_stratify else None
        )
        print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Initialize model
        model_params = self.config.get('model_params', {})
        self.model = RandomForestClassifier(**model_params)
        
        # Cross-validation (adjust folds for small datasets)
        cv_folds = self.config.get('cv_folds', 5)
        # For very small datasets, skip cross-validation or use simple validation
        if len(X_train) < 4:
            # Too small for CV, just train and use train score as CV score
            print("\nDataset too small for cross-validation, using training score...")
            self.model.fit(X_train, y_train)
            train_pred = self.model.predict(X_train)
            train_score = accuracy_score(y_train, train_pred)
            cv_scores = np.array([train_score])
            print(f"Training accuracy: {train_score:.4f}")
        else:
            # Ensure cv_folds doesn't exceed number of samples per class
            if y_train is not None:
                class_counts = pd.Series(y_train).value_counts()
                min_class_samples = class_counts.min() if len(class_counts) > 0 else len(X_train)
                max_folds = min(cv_folds, len(X_train), min_class_samples)
                cv_folds = max(2, min(cv_folds, max_folds))
            else:
                max_folds = min(cv_folds, len(X_train))
                cv_folds = max(2, min(cv_folds, max_folds))
            
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            print(f"\nCross-validation scores ({cv_folds} folds):")
            for i, score in enumerate(cv_scores, 1):
                print(f"  Fold {i}: {score:.4f}")
            print(f"  Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train final model (skip if already trained in small dataset case)
        if len(X_train) >= 4 or len(cv_scores) == 0:
            print("\nTraining final model...")
            self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        self.metrics = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'test_accuracy': float(accuracy_score(y_test, y_pred)),
            'test_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'test_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'test_f1': float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        print("\n=== Test Set Performance ===")
        print(f"Accuracy:  {self.metrics['test_accuracy']:.4f}")
        print(f"Precision: {self.metrics['test_precision']:.4f}")
        print(f"Recall:    {self.metrics['test_recall']:.4f}")
        print(f"F1 Score:  {self.metrics['test_f1']:.4f}")
        
        print("\nClassification Report:")
        # Only show classes that are present in test set
        unique_labels = sorted(set(list(y_test) + list(y_pred)))
        if len(unique_labels) == 2:
            print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived'], zero_division=0))
        else:
            # For single class or other cases, use labels parameter
            target_names = ['Not Survived' if 0 in unique_labels else '', 'Survived' if 1 in unique_labels else '']
            target_names = [name for name in target_names if name]
            print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names if len(target_names) == len(unique_labels) else None, zero_division=0))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.transformer.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.is_trained = True
        return self.metrics
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X, _ = self.prepare_features(df, is_training=False)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_single(self, passenger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict for a single passenger"""
        df = pd.DataFrame([passenger_data])
        predictions, probabilities = self.predict(df)
        
        result = {
            'survived': int(predictions[0]),
            'probability_not_survived': float(probabilities[0][0]),
            'probability_survived': float(probabilities[0][1]),
            'confidence': float(max(probabilities[0]))
        }
        return result
    
    def save_model(self, filepath: str):
        """Save complete pipeline"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_artifact = {
            'model': self.model,
            'transformer': self.transformer,
            'config': self.config,
            'metrics': self.metrics,
            'feature_names': self.transformer.feature_names
        }
        
        joblib.dump(model_artifact, filepath)
        print(f"\nModel saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MLPipeline':
        """Load complete pipeline"""
        model_artifact = joblib.load(filepath)
        
        pipeline = cls(model_artifact['config'])
        pipeline.model = model_artifact['model']
        pipeline.transformer = model_artifact['transformer']
        pipeline.metrics = model_artifact['metrics']
        pipeline.is_trained = True
        
        print(f"Model loaded from: {filepath}")
        return pipeline


def create_sample_data():
    """Create sample Titanic dataset for demonstration"""
    data = {
        'PassengerId': range(1, 892),
        'Survived': np.random.randint(0, 2, 891),
        'Pclass': np.random.choice([1, 2, 3], 891, p=[0.25, 0.25, 0.5]),
        'Sex': np.random.choice(['male', 'female'], 891, p=[0.65, 0.35]),
        'Age': np.random.normal(29, 14, 891).clip(0.42, 80),
        'SibSp': np.random.choice([0, 1, 2], 891, p=[0.7, 0.2, 0.1]),
        'Parch': np.random.choice([0, 1, 2], 891, p=[0.8, 0.15, 0.05]),
        'Fare': np.random.lognormal(2.5, 1.2, 891).clip(0, 512),
        'Embarked': np.random.choice(['S', 'C', 'Q'], 891, p=[0.72, 0.19, 0.09])
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.2), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], 'Age'] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], 'Embarked'] = np.nan
    
    return df


def main():
    """Main execution flow"""
    print("=" * 60)
    print("Day 76-84: End-to-End ML Pipeline")
    print("Building a Production-Ready ML System")
    print("=" * 60)
    
    # Configuration
    config = {
        'required_columns': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived'],
        'feature_columns': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
        'target_column': 'Survived',
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'model_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    }
    
    # Create sample data
    print("\nGenerating sample Titanic dataset...")
    df = create_sample_data()
    df.to_csv('data/titanic_train.csv', index=False)
    print(f"Dataset created: {len(df)} passengers")
    
    # Initialize pipeline
    pipeline = MLPipeline(config)
    
    # Load and validate data
    df, quality_report = pipeline.load_data('data/titanic_train.csv')
    
    # Train model
    metrics = pipeline.train(df)
    
    # Save model
    model_path = 'models/titanic_model_v1.pkl'
    pipeline.save_model(model_path)
    
    # Test single prediction
    print("\n=== Testing Single Prediction ===")
    test_passenger = {
        'Pclass': 3,
        'Sex': 'male',
        'Age': 22.0,
        'SibSp': 1,
        'Parch': 0,
        'Fare': 7.25,
        'Embarked': 'S'
    }
    
    result = pipeline.predict_single(test_passenger)
    print(f"Test passenger: {test_passenger}")
    print(f"Prediction: {'Survived' if result['survived'] == 1 else 'Not Survived'}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    # Test loading saved model
    print("\n=== Testing Model Loading ===")
    loaded_pipeline = MLPipeline.load_model(model_path)
    result2 = loaded_pipeline.predict_single(test_passenger)
    print(f"Loaded model prediction matches: {result == result2}")
    
    print("\n" + "=" * 60)
    print("Pipeline execution complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
