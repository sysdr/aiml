"""
Day 37: Introduction to AI, ML, and Deep Learning
Understanding the fundamentals of artificial intelligence
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AIMLIntroduction:
    """
    Introduction to AI, ML, and Deep Learning concepts
    """
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
        
    def traditional_programming_vs_ml(self):
        """
        Demonstrate the difference between traditional programming and machine learning
        """
        print("\n" + "="*60)
        print("TRADITIONAL PROGRAMMING vs MACHINE LEARNING")
        print("="*60)
        
        # Traditional Programming: Explicit rules
        def traditional_classifier(age, income):
            """Traditional: We write explicit rules"""
            if age > 30 and income > 50000:
                return "High Value Customer"
            elif age > 25:
                return "Medium Value Customer"
            else:
                return "Low Value Customer"
        
        # Machine Learning: Learn from data
        print("\n📝 Traditional Programming:")
        print("   We write explicit rules: if age > 30 and income > 50000...")
        print(f"   Example: {traditional_classifier(35, 60000)}")
        
        print("\n🤖 Machine Learning:")
        print("   Model learns patterns from data automatically")
        print("   No explicit rules - discovers relationships")
        
        # Generate sample data
        np.random.seed(42)
        ages = np.random.randint(18, 65, 100)
        incomes = np.random.randint(20000, 100000, 100)
        
        # Create labels based on pattern (simulating what ML would learn)
        labels = []
        for age, income in zip(ages, incomes):
            if age > 30 and income > 50000:
                labels.append(1)  # High value
            elif age > 25:
                labels.append(0.5)  # Medium value
            else:
                labels.append(0)  # Low value
        
        # Train ML model
        X = np.column_stack([ages, incomes])
        y = np.array(labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        print(f"   Model learned pattern with R² score: {r2:.3f}")
        print(f"   Prediction for age=35, income=60000: {model.predict([[35, 60000]])[0]:.3f}")
        
        return {
            'traditional': traditional_classifier(35, 60000),
            'ml_r2_score': r2,
            'ml_prediction': model.predict([[35, 60000]])[0]
        }
    
    def ai_ml_dl_relationship(self):
        """
        Explain the relationship between AI, ML, and Deep Learning
        """
        print("\n" + "="*60)
        print("AI, ML, AND DEEP LEARNING - THE RELATIONSHIP")
        print("="*60)
        
        print("\n🎯 Artificial Intelligence (AI):")
        print("   - Broadest concept: Machines performing tasks requiring intelligence")
        print("   - Includes: Expert systems, rule-based systems, ML, DL")
        print("   - Example: Chess-playing computer, voice assistant")
        
        print("\n🧠 Machine Learning (ML):")
        print("   - Subset of AI: Systems that learn from data")
        print("   - No explicit programming for every scenario")
        print("   - Types: Supervised, Unsupervised, Reinforcement Learning")
        print("   - Example: Email spam filter, recommendation system")
        
        print("\n🔬 Deep Learning (DL):")
        print("   - Subset of ML: Neural networks with multiple layers")
        print("   - Inspired by human brain structure")
        print("   - Excels at: Image recognition, natural language processing")
        print("   - Example: Face recognition, language translation")
        
        print("\n📊 Hierarchy:")
        print("   AI (Broadest)")
        print("   └── ML (Learning from data)")
        print("       └── DL (Deep neural networks)")
        
        return {
            'ai': 'Broadest - machines performing intelligent tasks',
            'ml': 'Subset of AI - learning from data',
            'dl': 'Subset of ML - deep neural networks'
        }
    
    def supervised_learning_demo(self):
        """
        Demonstrate supervised learning with a simple regression problem
        """
        print("\n" + "="*60)
        print("SUPERVISED LEARNING DEMONSTRATION")
        print("="*60)
        
        # Generate synthetic data: house prices based on size
        np.random.seed(42)
        house_sizes = np.random.uniform(500, 3000, 100)
        # Price = 100 * size + noise
        house_prices = 100 * house_sizes + np.random.normal(0, 20000, 100)
        house_prices = np.maximum(house_prices, 50000)  # Minimum price
        
        # Prepare data
        X = house_sizes.reshape(-1, 1)
        y = house_prices
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n📈 Model Performance:")
        print(f"   Mean Squared Error: ${mse:,.2f}")
        print(f"   R² Score: {r2:.3f} ({r2*100:.1f}% variance explained)")
        
        print(f"\n🔮 Predictions:")
        print(f"   House size 1000 sqft → Price: ${model.predict([[1000]])[0]:,.2f}")
        print(f"   House size 2000 sqft → Price: ${model.predict([[2000]])[0]:,.2f}")
        print(f"   House size 3000 sqft → Price: ${model.predict([[3000]])[0]:,.2f}")
        
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, alpha=0.6, label='Training Data', color='blue')
        plt.scatter(X_test, y_test, alpha=0.6, label='Test Data', color='green')
        
        # Plot regression line
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = model.predict(X_line)
        plt.plot(X_line, y_line, 'r-', linewidth=2, label='Learned Model')
        
        plt.xlabel('House Size (sqft)', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.title('Supervised Learning: House Price Prediction', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ml_prediction.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization to ml_prediction.png")
        plt.close()
        
        self.models['house_price'] = model
        self.metrics['house_price'] = {'mse': mse, 'r2': r2}
        
        return {
            'model': model,
            'mse': mse,
            'r2': r2,
            'predictions': {
                1000: model.predict([[1000]])[0],
                2000: model.predict([[2000]])[0],
                3000: model.predict([[3000]])[0]
            }
        }
    
    def learning_types_overview(self):
        """
        Overview of different types of machine learning
        """
        print("\n" + "="*60)
        print("TYPES OF MACHINE LEARNING")
        print("="*60)
        
        learning_types = {
            'Supervised Learning': {
                'description': 'Learn from labeled examples',
                'input': 'Features + Labels',
                'output': 'Predictions for new data',
                'examples': ['Email spam detection', 'House price prediction', 'Image classification']
            },
            'Unsupervised Learning': {
                'description': 'Find patterns in unlabeled data',
                'input': 'Features only (no labels)',
                'output': 'Hidden patterns, clusters, structure',
                'examples': ['Customer segmentation', 'Anomaly detection', 'Topic modeling']
            },
            'Reinforcement Learning': {
                'description': 'Learn through trial and error with rewards',
                'input': 'Actions + Rewards',
                'output': 'Optimal action policy',
                'examples': ['Game playing (Chess, Go)', 'Robot control', 'Trading algorithms']
            }
        }
        
        for ml_type, info in learning_types.items():
            print(f"\n📚 {ml_type}:")
            print(f"   Description: {info['description']}")
            print(f"   Input: {info['input']}")
            print(f"   Output: {info['output']}")
            print(f"   Examples: {', '.join(info['examples'])}")
        
        return learning_types
    
    def run_complete_introduction(self):
        """
        Run the complete introduction to AI, ML, and Deep Learning
        """
        print("\n" + "🎯 "*20)
        print("DAY 37: INTRODUCTION TO AI, ML, AND DEEP LEARNING")
        print("🎯 "*20 + "\n")
        
        results = {}
        
        # Run all sections
        results['traditional_vs_ml'] = self.traditional_programming_vs_ml()
        results['ai_ml_dl'] = self.ai_ml_dl_relationship()
        results['supervised_learning'] = self.supervised_learning_demo()
        results['learning_types'] = self.learning_types_overview()
        
        print("\n" + "="*60)
        print("✅ INTRODUCTION COMPLETE!")
        print("="*60)
        print("\n📊 Key Takeaways:")
        print("   1. AI is the broadest field - machines performing intelligent tasks")
        print("   2. ML is a subset of AI - systems that learn from data")
        print("   3. DL is a subset of ML - deep neural networks")
        print("   4. Supervised learning uses labeled data to make predictions")
        print("   5. Models learn patterns automatically, no explicit rules needed")
        print("\n🚀 You're now ready to dive deeper into machine learning!")
        
        return results


def main():
    """Main execution"""
    intro = AIMLIntroduction()
    results = intro.run_complete_introduction()
    return results


if __name__ == "__main__":
    main()
