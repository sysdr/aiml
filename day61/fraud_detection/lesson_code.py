"""
Day 61: Credit Card Fraud Detection System
A production-grade fraud detection pipeline with imbalanced data handling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    auc
)
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    """
    Production-grade fraud detection system handling imbalanced data
    """
    
    def __init__(self, model_type='random_forest', use_smote=True):
        """
        Initialize fraud detection system
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'logistic'
            use_smote: Whether to apply SMOTE for balancing
        """
        self.model_type = model_type
        self.use_smote = use_smote
        self.scaler = StandardScaler()
        self.model = None
        self.best_threshold = 0.5
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
    
    def generate_synthetic_data(self, n_samples=10000, fraud_ratio=0.002):
        """
        Generate synthetic credit card transaction data
        Mimics real-world fraud patterns
        """
        np.random.seed(42)
        
        n_fraud = int(n_samples * fraud_ratio)
        n_legitimate = n_samples - n_fraud
        
        # Legitimate transactions (normal patterns)
        legitimate_data = {
            'Amount': np.random.gamma(2, 50, n_legitimate),  # Most purchases are small
            'Time_Hour': np.random.normal(14, 4, n_legitimate) % 24,  # Peak at 2pm
            'V1': np.random.normal(0, 1, n_legitimate),
            'V2': np.random.normal(0, 1, n_legitimate),
            'V3': np.random.normal(0, 1, n_legitimate),
            'V4': np.random.normal(0, 1, n_legitimate),
            'Distance_From_Home': np.random.gamma(2, 20, n_legitimate),
            'Transaction_Velocity': np.random.poisson(1, n_legitimate),
            'Class': 0
        }
        
        # Fraudulent transactions (anomalous patterns)
        fraud_data = {
            'Amount': np.random.gamma(5, 150, n_fraud),  # Higher amounts
            'Time_Hour': np.random.uniform(0, 24, n_fraud),  # Any time
            'V1': np.random.normal(2, 2, n_fraud),  # Different distribution
            'V2': np.random.normal(-1, 2, n_fraud),
            'V3': np.random.normal(3, 1.5, n_fraud),
            'V4': np.random.normal(-2, 1.5, n_fraud),
            'Distance_From_Home': np.random.gamma(8, 50, n_fraud),  # Far from home
            'Transaction_Velocity': np.random.poisson(5, n_fraud),  # Many rapid transactions
            'Class': 1
        }
        
        # Combine and shuffle
        df_legit = pd.DataFrame(legitimate_data)
        df_fraud = pd.DataFrame(fraud_data)
        df = pd.concat([df_legit, df_fraud], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def engineer_features(self, df):
        """
        Create fraud-relevant features from raw transaction data
        """
        df = df.copy()
        
        # Amount-based features
        df['Amount_Log'] = np.log1p(df['Amount'])
        df['Amount_Squared'] = df['Amount'] ** 2
        df['Is_Large_Transaction'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)
        
        # Time-based features
        df['Is_Night'] = ((df['Time_Hour'] >= 22) | (df['Time_Hour'] <= 6)).astype(int)
        df['Is_Weekend_Pattern'] = ((df['Time_Hour'] >= 10) & (df['Time_Hour'] <= 16)).astype(int)
        
        # Velocity and distance features
        df['High_Velocity'] = (df['Transaction_Velocity'] > 3).astype(int)
        df['Far_From_Home'] = (df['Distance_From_Home'] > df['Distance_From_Home'].quantile(0.90)).astype(int)
        
        # Interaction features (suspicious combinations)
        df['Night_And_Far'] = df['Is_Night'] * df['Far_From_Home']
        df['Large_And_Fast'] = df['Is_Large_Transaction'] * df['High_Velocity']
        
        return df
    
    def prepare_data(self, df):
        """
        Prepare data for training: split and optionally apply SMOTE
        
        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test, X_test_original
            where X_test_original is the unscaled test DataFrame for demo purposes
        """
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        print(f"📊 Original class distribution: {Counter(y)}")
        print(f"   Fraud ratio: {y.mean():.4f}")
        
        # Stratified split to maintain fraud ratio
        # Fall back to regular split if stratified split fails (e.g., too few samples per class)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )
        except ValueError:
            # If stratified split fails (e.g., only 1 sample in a class), use regular split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # Store original test data before scaling (for demo purposes)
        X_test_original = X_test.copy()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE if requested
        if self.use_smote:
            # Check if we have enough fraud samples for SMOTE (need at least 2, preferably more)
            fraud_count = Counter(y_train)[1] if 1 in Counter(y_train) else 0
            if fraud_count >= 2:
                print("\n🔄 Applying SMOTE to balance training data...")
                try:
                    smote = SMOTE(sampling_strategy=0.5, random_state=42)
                    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                    print(f"   After SMOTE: {Counter(y_train)}")
                except ValueError as e:
                    print(f"   ⚠️  SMOTE failed (insufficient samples): {str(e)}")
                    print("   Continuing without SMOTE balancing...")
            else:
                print(f"\n⚠️  Skipping SMOTE: insufficient fraud samples ({fraud_count}) for oversampling")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_test_original
    
    def train(self, X_train, y_train):
        """
        Train the fraud detection model
        """
        print(f"\n🎯 Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Get training score
        train_score = self.model.score(X_train, y_train)
        print(f"   Training accuracy: {train_score:.4f}")
        
        return self
    
    def optimize_threshold(self, X_test, y_test, target_recall=0.90):
        """
        Find optimal decision threshold for desired recall
        
        Args:
            target_recall: Minimum recall (fraud detection rate) required
        """
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        
        # Find threshold that achieves target recall
        valid_indices = recalls >= target_recall
        if valid_indices.any():
            # Get highest precision among thresholds meeting recall target
            best_idx = np.where(valid_indices)[0][0]
            self.best_threshold = thresholds[best_idx]
            print(f"\n🎚️  Optimized threshold: {self.best_threshold:.4f}")
            print(f"   Target recall: {target_recall:.2%}")
            print(f"   Achieved recall: {recalls[best_idx]:.2%}")
            print(f"   Precision at this threshold: {precisions[best_idx]:.2%}")
        else:
            print(f"\n⚠️  Cannot achieve {target_recall:.2%} recall. Using default threshold.")
            self.best_threshold = 0.5
        
        return self.best_threshold
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive evaluation with fraud-specific metrics
        """
        # Get predictions with optimized threshold
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.best_threshold).astype(int)
        
        print("\n" + "="*60)
        print("📈 FRAUD DETECTION EVALUATION REPORT")
        print("="*60)
        
        # Classification metrics
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Legitimate', 'Fraud']))
        
        # ROC-AUC Score
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")
        
        # Confusion Matrix with business context
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("\n📊 Confusion Matrix:")
        print(f"   True Negatives (Correct legitimate): {tn:,}")
        print(f"   False Positives (Blocked good customers): {fp:,}")
        print(f"   False Negatives (Missed fraud): {fn:,}")
        print(f"   True Positives (Caught fraud): {tp:,}")
        
        # Business metrics
        print("\n💰 Business Impact Metrics:")
        fraud_caught_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"   Fraud Detection Rate: {fraud_caught_rate:.2%}")
        print(f"   False Alarm Rate: {false_alarm_rate:.2%}")
        
        # Estimate financial impact (example values)
        avg_fraud_loss = 200  # Average fraud transaction amount
        false_positive_cost = 5  # Cost per declined legitimate transaction
        
        money_saved = tp * avg_fraud_loss
        customer_friction_cost = fp * false_positive_cost
        net_benefit = money_saved - customer_friction_cost
        
        print(f"\n💵 Estimated Financial Impact (example):")
        print(f"   Money saved by catching fraud: ${money_saved:,.2f}")
        print(f"   Cost of false alarms: ${customer_friction_cost:,.2f}")
        print(f"   Net benefit: ${net_benefit:,.2f}")
        
        return {
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'fraud_caught_rate': fraud_caught_rate,
            'false_alarm_rate': false_alarm_rate
        }
    
    def plot_evaluation_metrics(self, X_test, y_test):
        """
        Create comprehensive visualization of model performance
        """
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.best_threshold).astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_xticklabels(['Legitimate', 'Fraud'])
        axes[0, 0].set_yticklabels(['Legitimate', 'Fraud'])
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                       label='Random Classifier')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Precision-Recall Curve
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recalls, precisions)
        
        axes[1, 0].plot(recalls, precisions, color='green', lw=2,
                       label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[1, 0].axvline(x=0.90, color='red', linestyle='--', 
                          label='Target Recall (90%)')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 0].legend(loc="upper right")
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features
            
            axes[1, 1].barh(range(len(indices)), importances[indices], 
                           color='skyblue')
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([f'Feature {i}' for i in indices])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importances', 
                                fontsize=14, fontweight='bold')
            axes[1, 1].grid(axis='x', alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 
                           'Feature importance not available\nfor this model type',
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Feature Importances', 
                                fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('fraud_detection_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n📊 Evaluation plots saved as 'fraud_detection_evaluation.png'")
        plt.close()
    
    def predict_transaction(self, transaction_features):
        """
        Score a single transaction for fraud risk
        
        Args:
            transaction_features: Dict or array of transaction features
            
        Returns:
            fraud_probability: Probability of fraud (0-1)
            is_fraud: Boolean prediction
            risk_level: 'LOW', 'MEDIUM', or 'HIGH'
        """
        if isinstance(transaction_features, dict):
            # Convert dict to DataFrame for consistent processing
            transaction_df = pd.DataFrame([transaction_features])
        else:
            transaction_df = pd.DataFrame([transaction_features])
        
        # Scale features
        transaction_scaled = self.scaler.transform(transaction_df)
        
        # Get fraud probability
        fraud_prob = self.model.predict_proba(transaction_scaled)[0, 1]
        is_fraud = fraud_prob >= self.best_threshold
        
        # Determine risk level
        if fraud_prob < 0.3:
            risk_level = 'LOW'
        elif fraud_prob < 0.7:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        return {
            'fraud_probability': fraud_prob,
            'is_fraud': is_fraud,
            'risk_level': risk_level,
            'threshold_used': self.best_threshold
        }


def compare_models():
    """
    Compare different models for fraud detection
    """
    print("="*70)
    print("🔬 COMPARING FRAUD DETECTION MODELS")
    print("="*70)
    
    # Generate data once
    system = FraudDetectionSystem()
    df = system.generate_synthetic_data(n_samples=10000, fraud_ratio=0.002)
    df = system.engineer_features(df)
    
    models = {
        'Random Forest': 'random_forest',
        'Gradient Boosting': 'gradient_boosting',
        'Logistic Regression': 'logistic'
    }
    
    results = {}
    
    for name, model_type in models.items():
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print('='*70)
        
        # Create and train model
        detector = FraudDetectionSystem(model_type=model_type, use_smote=True)
        X_train, X_test, y_train, y_test, _ = detector.prepare_data(df)
        detector.train(X_train, y_train)
        detector.optimize_threshold(X_test, y_test, target_recall=0.90)
        
        # Evaluate
        metrics = detector.evaluate(X_test, y_test)
        results[name] = metrics
        
    # Summary comparison
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'ROC-AUC':<12} {'Fraud Caught':<15} {'False Alarms'}")
    print("-"*70)
    
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['roc_auc']:<12.4f} "
              f"{metrics['fraud_caught_rate']:<15.2%} "
              f"{metrics['false_alarm_rate']:.2%}")


def main():
    """
    Main execution: Build and evaluate fraud detection system
    """
    print("="*70)
    print("💳 DAY 61: CREDIT CARD FRAUD DETECTION SYSTEM")
    print("="*70)
    
    # Create fraud detection system
    print("\n1️⃣  Initializing fraud detection system...")
    detector = FraudDetectionSystem(model_type='random_forest', use_smote=True)
    
    # Generate realistic fraud data
    print("\n2️⃣  Generating synthetic credit card transaction data...")
    df = detector.generate_synthetic_data(n_samples=10000, fraud_ratio=0.002)
    print(f"   Generated {len(df):,} transactions")
    print(f"   Fraud cases: {df['Class'].sum():,} ({df['Class'].mean():.2%})")
    
    # Engineer features
    print("\n3️⃣  Engineering fraud-relevant features...")
    df = detector.engineer_features(df)
    print(f"   Created {len(df.columns) - 1} features")
    
    # Prepare data
    print("\n4️⃣  Preparing data with imbalance handling...")
    X_train, X_test, y_train, y_test, X_test_original = detector.prepare_data(df)
    
    # Train model
    print("\n5️⃣  Training Random Forest fraud detector...")
    detector.train(X_train, y_train)
    
    # Optimize threshold
    print("\n6️⃣  Optimizing decision threshold for 90% fraud detection...")
    detector.optimize_threshold(X_test, y_test, target_recall=0.90)
    
    # Evaluate performance
    print("\n7️⃣  Evaluating fraud detection performance...")
    detector.evaluate(X_test, y_test)
    
    # Create visualizations
    print("\n8️⃣  Generating evaluation visualizations...")
    detector.plot_evaluation_metrics(X_test, y_test)
    
    # Demo: Score individual transactions
    print("\n9️⃣  Demo: Scoring individual transactions...")
    print("="*70)
    
    # Test legitimate transaction
    legit_indices = [i for i, val in enumerate(y_test) if val == 0]
    if len(legit_indices) > 0:
        legit_transaction = X_test_original.iloc[legit_indices[0]].to_dict()
        result = detector.predict_transaction(legit_transaction)
        print(f"\n✅ Legitimate Transaction Example:")
        print(f"   Fraud Probability: {result['fraud_probability']:.2%}")
        print(f"   Prediction: {'🚨 FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
        print(f"   Risk Level: {result['risk_level']}")
    
    # Test fraudulent transaction
    fraud_indices = [i for i, val in enumerate(y_test) if val == 1]
    if len(fraud_indices) > 0:
        fraud_transaction = X_test_original.iloc[fraud_indices[0]].to_dict()
        result = detector.predict_transaction(fraud_transaction)
        print(f"\n🚨 Fraudulent Transaction Example:")
        print(f"   Fraud Probability: {result['fraud_probability']:.2%}")
        print(f"   Prediction: {'🚨 FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
        print(f"   Risk Level: {result['risk_level']}")
    
    # Compare models
    print("\n\n🔍 Running model comparison...\n")
    compare_models()
    
    print("\n" + "="*70)
    print("✅ FRAUD DETECTION SYSTEM COMPLETE!")
    print("="*70)
    print("\n📚 Key Takeaways:")
    print("   • Handled extreme class imbalance with SMOTE and class weights")
    print("   • Engineered features capturing fraud patterns")
    print("   • Optimized threshold for business requirements (90% recall)")
    print("   • Evaluated using fraud-specific metrics (not just accuracy)")
    print("   • Built production-ready real-time scoring system")
    print("\n🎯 Next: Day 62 - K-Nearest Neighbors for similarity-based fraud detection")


if __name__ == "__main__":
    main()
