"""
Day 65: SVMs with Scikit-learn - Production Fraud Detection System

This implementation demonstrates:
1. Feature scaling pipelines for SVM
2. Hyperparameter tuning with GridSearchCV
3. Model evaluation with classification metrics
4. Production deployment patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def generate_fraud_dataset(n_samples=10000, fraud_ratio=0.05):
    """
    Generate realistic credit card transaction dataset
    
    Features:
    - amount: Transaction amount ($)
    - hour: Hour of day (0-23)
    - distance_from_last: Miles from previous transaction
    - merchant_risk_score: Risk score of merchant (0-1)
    """
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # Legitimate transactions
    legit_amount = np.random.gamma(2, 50, n_legit)  # Typical purchases
    legit_hour = np.random.normal(14, 4, n_legit) % 24  # Daytime bias
    legit_distance = np.random.exponential(10, n_legit)  # Close to home
    legit_merchant = np.random.beta(2, 5, n_legit)  # Low-risk merchants
    
    # Fraudulent transactions
    fraud_amount = np.random.gamma(5, 150, n_fraud)  # Higher amounts
    fraud_hour = np.random.normal(2, 3, n_fraud) % 24  # Late night bias
    fraud_distance = np.random.gamma(3, 50, n_fraud)  # Far from home
    fraud_merchant = np.random.beta(5, 2, n_fraud)  # High-risk merchants
    
    # Combine datasets
    amounts = np.concatenate([legit_amount, fraud_amount])
    hours = np.concatenate([legit_hour, fraud_hour])
    distances = np.concatenate([legit_distance, fraud_distance])
    merchants = np.concatenate([legit_merchant, fraud_merchant])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'amount': amounts,
        'hour': hours,
        'distance_from_last': distances,
        'merchant_risk_score': merchants,
        'is_fraud': labels
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def demonstrate_scaling_impact():
    """Show why feature scaling is critical for SVMs"""
    print("\n" + "="*60)
    print("DEMONSTRATION: Impact of Feature Scaling on SVM")
    print("="*60)
    
    # Generate small dataset for demo
    df = generate_fraud_dataset(n_samples=1000)
    X = df[['amount', 'hour', 'distance_from_last', 'merchant_risk_score']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Without scaling
    print("\n1. SVM WITHOUT Scaling:")
    print("-" * 60)
    svm_no_scale = SVC(kernel='rbf', random_state=42)
    svm_no_scale.fit(X_train, y_train)
    score_no_scale = svm_no_scale.score(X_test, y_test)
    print(f"   Test Accuracy: {score_no_scale:.4f}")
    
    # With scaling
    print("\n2. SVM WITH Scaling:")
    print("-" * 60)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    score_with_scale = pipeline.score(X_test, y_test)
    print(f"   Test Accuracy: {score_with_scale:.4f}")
    
    improvement = (score_with_scale - score_no_scale) / score_no_scale * 100
    print(f"\n   Improvement: {improvement:.1f}%")
    print("\n   💡 Key Insight: Scaling is NON-NEGOTIABLE for SVMs!")
    print()


def train_fraud_detector():
    """Train production-ready fraud detection SVM"""
    print("\n" + "="*60)
    print("TRAINING: Production Fraud Detection System")
    print("="*60)
    
    # Generate realistic dataset
    print("\n📊 Generating transaction dataset...")
    df = generate_fraud_dataset(n_samples=10000, fraud_ratio=0.05)
    print(f"   Total transactions: {len(df):,}")
    print(f"   Fraudulent: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"   Legitimate: {(~df['is_fraud'].astype(bool)).sum():,}")
    
    # Prepare features and target
    X = df[['amount', 'hour', 'distance_from_last', 'merchant_risk_score']]
    y = df['is_fraud']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\n   Training set: {len(X_train):,}")
    print(f"   Test set: {len(X_test):,}")
    
    # Create pipeline with scaling
    print("\n🔧 Building SVM pipeline with preprocessing...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42, class_weight='balanced'))
    ])
    
    # Hyperparameter grid
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 0.01, 0.1],
        'svm__kernel': ['rbf', 'linear']
    }
    
    print(f"   Testing {len(param_grid['svm__C']) * len(param_grid['svm__gamma']) * len(param_grid['svm__kernel'])} configurations...")
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    print("\n🔍 Running GridSearchCV (5-fold cross-validation)...")
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print("\n✅ Training complete!")
    print(f"\n   Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"      {param}: {value}")
    print(f"\n   Best CV F1 Score: {grid_search.best_score_:.4f}")
    
    return best_model, X_train, X_test, y_train, y_test, df


def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    print("\n" + "="*60)
    print("EVALUATION: Model Performance Metrics")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print("\n📊 Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, 
                                target_names=['Legitimate', 'Fraud'],
                                digits=4))
    
    # Confusion matrix
    print("\n📈 Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"                 Legit  Fraud")
    print(f"   Actual Legit  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"   Actual Fraud  {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n   False Positives (Legit flagged as Fraud): {fp}")
    print(f"   False Negatives (Fraud missed): {fn}")
    print(f"   Precision: {precision:.4f} (of flagged fraud, % actually fraud)")
    print(f"   Recall: {recall:.4f} (of actual fraud, % caught)")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n   ROC-AUC Score: {roc_auc:.4f}")
    
    return y_pred, y_pred_proba


def visualize_results(model, X_test, y_test, y_pred_proba, df):
    """Create visualizations"""
    print("\n" + "="*60)
    print("VISUALIZATION: Model Insights")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature distributions by class
    ax = axes[0, 0]
    for feature in ['amount', 'distance_from_last']:
        for fraud_status in [0, 1]:
            data = df[df['is_fraud'] == fraud_status][feature]
            label = 'Fraud' if fraud_status == 1 else 'Legit'
            ax.hist(data, alpha=0.5, bins=30, label=f'{label} - {feature}')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Feature Distributions by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. ROC Curve
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ax.plot(fpr, tpr, linewidth=2, label=f'SVM (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Prediction probabilities
    ax = axes[1, 0]
    ax.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='Legitimate', color='blue')
    ax.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='Fraud', color='red')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    ax.set_xlabel('Fraud Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Probability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Confusion matrix heatmap
    ax = axes[1, 1]
    cm = confusion_matrix(y_test, (y_pred_proba >= 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('svm_fraud_detection_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization: svm_fraud_detection_results.png")
    plt.close()


def save_model_for_production(model):
    """Save model for production deployment"""
    print("\n" + "="*60)
    print("DEPLOYMENT: Saving Model for Production")
    print("="*60)
    
    # Save model
    model_filename = 'fraud_detector_svm_v1.pkl'
    joblib.dump(model, model_filename)
    print(f"\n✓ Model saved: {model_filename}")
    
    # Demo: Load and predict
    print("\n📦 Production Usage Example:")
    print("-" * 60)
    loaded_model = joblib.load(model_filename)
    
    # Sample transactions
    test_transactions = np.array([
        [50.0, 14, 5, 0.2],      # Small daytime purchase nearby
        [1200.0, 3, 250, 0.85],  # Large late-night purchase far away
        [25.0, 18, 2, 0.15],     # Small evening purchase nearby
        [800.0, 2, 180, 0.9]     # Large late-night purchase far away
    ])
    
    predictions = loaded_model.predict(test_transactions)
    probabilities = loaded_model.predict_proba(test_transactions)
    
    print("\n   Sample Transaction Predictions:")
    print("   " + "-" * 56)
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        trans = test_transactions[i]
        fraud_prob = prob[1] * 100
        status = "🚨 FRAUD" if pred == 1 else "✅ LEGIT"
        print(f"   Transaction {i+1}: ${trans[0]:.0f} at {trans[1]:.0f}:00, {trans[2]:.0f}mi away")
        print(f"      → {status} (Fraud probability: {fraud_prob:.1f}%)")
        print()


def demonstrate_kernel_comparison():
    """Compare different SVM kernels"""
    print("\n" + "="*60)
    print("COMPARISON: SVM Kernel Performance")
    print("="*60)
    
    # Generate dataset
    df = generate_fraud_dataset(n_samples=2000)
    X = df[['amount', 'hour', 'distance_from_last', 'merchant_risk_score']]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    kernels = ['linear', 'rbf', 'poly']
    results = {}
    
    print("\n⚙️  Training SVMs with different kernels...")
    print("-" * 60)
    
    for kernel in kernels:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel, random_state=42, class_weight='balanced'))
        ])
        
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        y_pred = pipeline.predict(X_test)
        
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred)
        
        results[kernel] = {'accuracy': score, 'f1': f1}
        print(f"   {kernel:8s} kernel → Accuracy: {score:.4f}, F1: {f1:.4f}")
    
    best_kernel = max(results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\n   🏆 Best kernel for this problem: {best_kernel}")
    print()


def main():
    """Main execution flow"""
    print("\n" + "="*70)
    print("   Day 65: SVMs with Scikit-learn - Fraud Detection System")
    print("="*70)
    
    # 1. Demonstrate scaling impact
    demonstrate_scaling_impact()
    
    # 2. Compare kernels
    demonstrate_kernel_comparison()
    
    # 3. Train production model
    model, X_train, X_test, y_train, y_test, df = train_fraud_detector()
    
    # 4. Evaluate
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # 5. Visualize
    visualize_results(model, X_test, y_test, y_pred_proba, df)
    
    # 6. Save for production
    save_model_for_production(model)
    
    print("\n" + "="*70)
    print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. ✓ Feature scaling is critical for SVM performance")
    print("  2. ✓ GridSearchCV finds optimal hyperparameters systematically")
    print("  3. ✓ Pipeline ensures preprocessing happens consistently")
    print("  4. ✓ Class imbalance handled with class_weight='balanced'")
    print("  5. ✓ Production deployment uses joblib for model persistence")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
