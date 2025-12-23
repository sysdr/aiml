"""
Day 47: Housing Price Prediction System
A production-ready regression model for real estate price estimation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)


class HousingDataGenerator:
    """Generate realistic housing dataset with controlled patterns"""
    
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        
    def generate(self):
        """Generate synthetic housing data with realistic distributions"""
        
        # Base features with realistic ranges
        sqft = np.random.normal(2000, 800, self.n_samples)
        sqft = np.clip(sqft, 500, 6000)  # Realistic bounds
        
        bedrooms = np.random.poisson(3, self.n_samples)
        bedrooms = np.clip(bedrooms, 1, 6)
        
        bathrooms = np.random.poisson(2, self.n_samples)
        bathrooms = np.clip(bathrooms, 1, 5)
        
        lot_size = np.random.normal(0.25, 0.15, self.n_samples)
        lot_size = np.clip(lot_size, 0.05, 2.0)  # acres
        
        age = np.random.exponential(15, self.n_samples)
        age = np.clip(age, 0, 100)
        
        garage = np.random.choice([0, 1, 2, 3], self.n_samples, p=[0.1, 0.3, 0.5, 0.1])
        
        # Price calculation with realistic relationships
        # Base price from square footage
        base_price = 100 * sqft
        
        # Bedroom/bathroom premiums
        bedroom_value = bedrooms * 15000
        bathroom_value = bathrooms * 20000
        
        # Lot size premium
        lot_value = lot_size * 50000
        
        # Age depreciation (non-linear)
        age_depreciation = age * 1000
        
        # Garage premium
        garage_value = garage * 25000
        
        # Calculate final price with noise
        price = (base_price + bedroom_value + bathroom_value + 
                lot_value + garage_value - age_depreciation)
        
        # Add realistic noise (±10%)
        noise = np.random.normal(0, price * 0.1, self.n_samples)
        price = price + noise
        
        # Ensure reasonable price range
        price = np.clip(price, 50000, 2000000)
        
        # Create DataFrame
        df = pd.DataFrame({
            "sqft": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "lot_size": lot_size,
            "age": age,
            "garage": garage,
            "price": price
        })
        
        # Introduce realistic missing values (10%)
        missing_mask = np.random.random(self.n_samples) < 0.1
        df.loc[missing_mask, "garage"] = np.nan
        
        missing_mask = np.random.random(self.n_samples) < 0.05
        df.loc[missing_mask, "age"] = np.nan
        
        return df


class HousingPricePredictor:
    """Production-ready housing price prediction system"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def load_data(self, filepath="housing_data.csv"):
        """Load housing data from CSV"""
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} properties from {filepath}")
            return df
        except FileNotFoundError:
            print(f"Error: {filepath} not found. Run with --generate-data first.")
            return None
    
    def analyze_data(self, df):
        """Perform exploratory data analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print("\nDataset Overview:")
        print(f"Total properties: {len(df)}")
        print(f"Features: {list(df.columns)}")
        
        print("\nFeature Statistics:")
        print(df.describe())
        
        # Missing values
        print("\nMissing Values:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            "Missing Count": missing,
            "Percentage": missing_pct
        })
        print(missing_df[missing_df["Missing Count"] > 0])
        
        # Correlation analysis
        print("\nFeature Correlations with Price:")
        correlations = df.corr()["price"].sort_values(ascending=False)
        print(correlations)
        
        # Create visualizations
        self._create_visualizations(df)
        
        print("\nVisualization saved: housing_analysis.png")
    
    def _create_visualizations(self, df):
        """Create comprehensive data visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Housing Data Analysis", fontsize=16, y=1.00)
        
        # Price distribution
        axes[0, 0].hist(df["price"], bins=50, edgecolor="black", alpha=0.7)
        axes[0, 0].set_title("Price Distribution")
        axes[0, 0].set_xlabel("Price ($)")
        axes[0, 0].set_ylabel("Frequency")
        
        # Square footage vs price
        axes[0, 1].scatter(df["sqft"], df["price"], alpha=0.5)
        axes[0, 1].set_title("Square Footage vs Price")
        axes[0, 1].set_xlabel("Square Feet")
        axes[0, 1].set_ylabel("Price ($)")
        
        # Bedrooms vs price
        axes[0, 2].boxplot([df[df["bedrooms"]==i]["price"].values 
                           for i in sorted(df["bedrooms"].unique())])
        axes[0, 2].set_title("Bedrooms vs Price")
        axes[0, 2].set_xlabel("Bedrooms")
        axes[0, 2].set_ylabel("Price ($)")
        
        # Correlation heatmap
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                   ax=axes[1, 0], cbar_kws={"label": "Correlation"})
        axes[1, 0].set_title("Feature Correlation Matrix")
        
        # Age vs price
        axes[1, 1].scatter(df["age"], df["price"], alpha=0.5)
        axes[1, 1].set_title("House Age vs Price")
        axes[1, 1].set_xlabel("Age (years)")
        axes[1, 1].set_ylabel("Price ($)")
        
        # Garage vs price
        axes[1, 2].boxplot([df[df["garage"]==i]["price"].values 
                           for i in sorted(df["garage"].dropna().unique())])
        axes[1, 2].set_title("Garage Spaces vs Price")
        axes[1, 2].set_xlabel("Garage Spaces")
        axes[1, 2].set_ylabel("Price ($)")
        
        plt.tight_layout()
        plt.savefig("housing_analysis.png", dpi=150, bbox_inches="tight")
        print("Analysis visualizations saved!")
    
    def engineer_features(self, df):
        """Feature engineering pipeline"""
        df_processed = df.copy()
        
        # Handle missing values
        # Strategy 1: Mean imputation for continuous features
        if df_processed["age"].isnull().any():
            mean_age = df_processed["age"].mean()
            df_processed["age_missing"] = df_processed["age"].isnull().astype(int)
            df_processed["age"].fillna(mean_age, inplace=True)
        else:
            df_processed["age_missing"] = 0
            
        # Strategy 2: Mode imputation for categorical features
        if df_processed["garage"].isnull().any():
            mode_garage = df_processed["garage"].mode()[0]
            df_processed["garage_missing"] = df_processed["garage"].isnull().astype(int)
            df_processed["garage"].fillna(mode_garage, inplace=True)
        else:
            df_processed["garage_missing"] = 0
        
        # Create derived features
        df_processed["price_per_sqft"] = df_processed["price"] / df_processed["sqft"]
        df_processed["total_rooms"] = df_processed["bedrooms"] + df_processed["bathrooms"]
        df_processed["is_new"] = (df_processed["age"] < 5).astype(int)
        
        print(f"\nFeature Engineering Complete:")
        print(f"Original features: {len(df.columns)}")
        print(f"Engineered features: {len(df_processed.columns)}")
        
        return df_processed
    
    def prepare_data(self, df):
        """Prepare data for model training"""
        # Separate features and target
        feature_cols = ["sqft", "bedrooms", "bathrooms", "lot_size", "age", 
                       "garage", "age_missing", "garage_missing", "total_rooms", "is_new"]
        
        X = df[feature_cols]
        y = df["price"]
        
        self.feature_names = feature_cols
        
        # Split data: 60% train, 20% validation, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
        )
        
        print(f"\nData Split:")
        print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train, y_train):
        """Train the regression model"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Display model coefficients
        print("\nModel Coefficients:")
        for feature, coef in zip(self.feature_names, self.model.coef_):
            print(f"{feature:20s}: ${coef:>12,.2f}")
        print(f"{'Intercept':20s}: ${self.model.intercept_:>12,.2f}")
        
        print("\nModel training complete!")
    
    def evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        results = {}
        
        for name, y_true, y_pred in [
            ("Training", y_train, y_train_pred),
            ("Validation", y_val, y_val_pred),
            ("Test", y_test, y_test_pred)
        ]:
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            results[name] = {
                "R²": r2,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape
            }
        
        # Display results
        print(f"\n{'Metric':<15} {'Training':>15} {'Validation':>15} {'Test':>15}")
        print("-" * 62)
        
        for metric in ["R²", "MAE", "RMSE", "MAPE"]:
            if metric == "R²":
                print(f"{metric:<15} {results['Training'][metric]:>15.4f} "
                     f"{results['Validation'][metric]:>15.4f} "
                     f"{results['Test'][metric]:>15.4f}")
            elif metric == "MAPE":
                print(f"{metric:<15} {results['Training'][metric]:>14.2f}% "
                     f"{results['Validation'][metric]:>14.2f}% "
                     f"{results['Test'][metric]:>14.2f}%")
            else:
                print(f"{metric:<15} ${results['Training'][metric]:>14,.0f} "
                     f"${results['Validation'][metric]:>14,.0f} "
                     f"${results['Test'][metric]:>14,.0f}")
        
        # Performance analysis
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        
        train_r2 = results["Training"]["R²"]
        test_r2 = results["Test"]["R²"]
        overfitting_gap = train_r2 - test_r2
        
        print(f"\nModel Quality:")
        if test_r2 > 0.75:
            print(f"✓ R² Score: {test_r2:.4f} (Good - explains {test_r2*100:.1f}% of variance)")
        else:
            print(f"⚠ R² Score: {test_r2:.4f} (Could be improved)")
        
        median_price = y_test.median()
        mae_pct = (results["Test"]["MAE"] / median_price) * 100
        
        if mae_pct < 10:
            print(f"✓ MAE: ${results['Test']['MAE']:,.0f} ({mae_pct:.1f}% of median price)")
        else:
            print(f"⚠ MAE: ${results['Test']['MAE']:,.0f} ({mae_pct:.1f}% of median price)")
        
        print(f"\nOverfitting Check:")
        if overfitting_gap < 0.05:
            print(f"✓ Train-Test Gap: {overfitting_gap:.4f} (No significant overfitting)")
        else:
            print(f"⚠ Train-Test Gap: {overfitting_gap:.4f} (Possible overfitting)")
        
        # Create prediction plots
        self._create_evaluation_plots(y_test, y_test_pred)
        
        return results
    
    def _create_evaluation_plots(self, y_true, y_pred):
        """Create prediction quality visualizations"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Predicted vs Actual
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    "r--", linewidth=2)
        axes[0].set_xlabel("Actual Price ($)")
        axes[0].set_ylabel("Predicted Price ($)")
        axes[0].set_title("Predicted vs Actual Prices")
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[1].set_xlabel("Predicted Price ($)")
        axes[1].set_ylabel("Residuals ($)")
        axes[1].set_title("Residual Plot")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
        print("\nEvaluation plots saved: model_evaluation.png")
    
    def predict(self, features_dict):
        """Make prediction for a single property"""
        if not self.is_fitted:
            raise ValueError("Model not trained. Run training first.")
        
        # Validate inputs
        required_features = ["sqft", "bedrooms", "bathrooms", "lot_size", "age", "garage"]
        for feature in required_features:
            if feature not in features_dict:
                raise ValueError(f"Missing required feature: {feature}")
            if features_dict[feature] < 0:
                raise ValueError(f"Invalid {feature}: must be non-negative")
        
        # Engineer features
        features_dict["age_missing"] = 0
        features_dict["garage_missing"] = 0
        features_dict["total_rooms"] = features_dict["bedrooms"] + features_dict["bathrooms"]
        features_dict["is_new"] = 1 if features_dict["age"] < 5 else 0
        
        # Create feature vector
        X = np.array([[features_dict[f] for f in self.feature_names]])
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        # Calculate confidence interval (±1 standard deviation from residuals)
        confidence_margin = 50000  # Simplified for demonstration
        
        print("\n" + "="*60)
        print("PRICE PREDICTION")
        print("="*60)
        print(f"\nProperty Details:")
        print(f"  Square Feet: {features_dict['sqft']:,.0f}")
        print(f"  Bedrooms: {features_dict['bedrooms']}")
        print(f"  Bathrooms: {features_dict['bathrooms']}")
        print(f"  Lot Size: {features_dict['lot_size']:.2f} acres")
        print(f"  Age: {features_dict['age']:.0f} years")
        print(f"  Garage Spaces: {features_dict['garage']}")
        
        print(f"\nPredicted Price: ${prediction:,.0f}")
        print(f"Confidence Interval: ${prediction-confidence_margin:,.0f} - ${prediction+confidence_margin:,.0f}")
        
        return prediction
    
    def save_model(self, filepath="housing_model.pkl"):
        """Save trained model to disk"""
        if not self.is_fitted:
            raise ValueError("No trained model to save")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to: {filepath}")
    
    def load_model(self, filepath="housing_model.pkl"):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.is_fitted = True
        print(f"Model loaded from: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Housing Price Prediction System")
    parser.add_argument("--generate-data", action="store_true",
                       help="Generate synthetic housing dataset")
    parser.add_argument("--analyze", action="store_true",
                       help="Perform exploratory data analysis")
    parser.add_argument("--train", action="store_true",
                       help="Train the prediction model")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate model performance")
    parser.add_argument("--predict", action="store_true",
                       help="Make a price prediction")
    parser.add_argument("--sqft", type=float, help="Square footage")
    parser.add_argument("--bedrooms", type=int, help="Number of bedrooms")
    parser.add_argument("--bathrooms", type=int, help="Number of bathrooms")
    parser.add_argument("--lot-size", type=float, default=0.25, help="Lot size in acres")
    parser.add_argument("--age", type=float, default=10, help="House age in years")
    parser.add_argument("--garage", type=int, default=2, help="Garage spaces")
    
    args = parser.parse_args()
    
    predictor = HousingPricePredictor()
    
    if args.generate_data:
        print("Generating housing dataset...")
        generator = HousingDataGenerator(n_samples=1000)
        df = generator.generate()
        df.to_csv("housing_data.csv", index=False)
        print(f"Dataset saved: housing_data.csv ({len(df)} properties)")
        print("\nSample data:")
        print(df.head())
        
    elif args.analyze:
        df = predictor.load_data()
        if df is not None:
            predictor.analyze_data(df)
            
    elif args.train or args.evaluate:
        df = predictor.load_data()
        if df is not None:
            # Feature engineering
            df_processed = predictor.engineer_features(df)
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = predictor.prepare_data(df_processed)
            
            # Train model
            predictor.train(X_train, y_train)
            
            if args.evaluate:
                # Evaluate
                predictor.evaluate(X_train, X_val, X_test, y_train, y_val, y_test)
            
            # Save model
            predictor.save_model()
            
    elif args.predict:
        if not all([args.sqft, args.bedrooms, args.bathrooms]):
            print("Error: --predict requires --sqft, --bedrooms, and --bathrooms")
            return
        
        # Try to load existing model
        if Path("housing_model.pkl").exists():
            predictor.load_model()
        else:
            print("No trained model found. Training new model...")
            df = predictor.load_data()
            if df is not None:
                df_processed = predictor.engineer_features(df)
                X_train, X_val, X_test, y_train, y_val, y_test = predictor.prepare_data(df_processed)
                predictor.train(X_train, y_train)
                predictor.save_model()
        
        # Make prediction
        features = {
            "sqft": args.sqft,
            "bedrooms": args.bedrooms,
            "bathrooms": args.bathrooms,
            "lot_size": args.lot_size,
            "age": args.age,
            "garage": args.garage
        }
        
        predictor.predict(features)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
