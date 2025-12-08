"""
Day 39 Dashboard: Supervised vs. Unsupervised Learning
Interactive web interface for exploring ML concepts
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score

# Page configuration
st.set_page_config(
    page_title="Day 39: Supervised vs. Unsupervised Learning",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Day 39: Supervised vs. Unsupervised Learning")
st.markdown("### Understanding AI's Two Fundamental Approaches")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["Overview", "Supervised Learning", "Unsupervised Learning", "Comparison", "Interactive Demo"]
)

if page == "Overview":
    st.header("📚 Overview")
    st.markdown("""
    This lesson demonstrates the fundamental difference between two main approaches in machine learning:
    
    ### 🎯 Supervised Learning
    - **With a teacher**: Uses labeled data (input + correct answers)
    - **Goal**: Learn to predict outputs from inputs
    - **Example**: Spam email classification
    
    ### 🔍 Unsupervised Learning
    - **Without a teacher**: Uses only input data (no labels)
    - **Goal**: Discover hidden patterns and structures
    - **Example**: Customer segmentation
    
    ### Key Difference
    The API difference is simple but crucial:
    - **Supervised**: `model.fit(X, y)` ← Requires labels
    - **Unsupervised**: `model.fit(X)` ← No labels needed
    """)

elif page == "Supervised Learning":
    st.header("📧 Supervised Learning: Email Spam Classification")
    st.markdown("""
    **Key Insight**: We KNOW the correct answers (spam/not spam) for training data.
    """)
    
    if st.button("Run Supervised Learning Demo"):
        with st.spinner("Training model..."):
            # Generate synthetic email data
            np.random.seed(42)
            n_samples = 1000
            
            # Spam emails: high counts for promotional words
            spam_emails = np.random.poisson(lam=[8, 6, 5, 1, 1], size=(n_samples//2, 5))
            # Normal emails: high counts for work-related words
            normal_emails = np.random.poisson(lam=[1, 1, 0.5, 7, 6], size=(n_samples//2, 5))
            
            X = np.vstack([spam_emails, normal_emails])
            y = np.array([1] * (n_samples//2) + [0] * (n_samples//2))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = MultinomialNB()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", len(X_train))
            with col2:
                st.metric("Test Samples", len(X_test))
            with col3:
                st.metric("Accuracy", f"{accuracy:.2%}")
            
            st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
            
            # Show example predictions
            st.subheader("Example Predictions")
            predictions_df = pd.DataFrame({
                'Email': range(1, min(11, len(y_test) + 1)),
                'True Label': ['Spam' if y == 1 else 'Not Spam' for y in y_test[:10]],
                'Predicted Label': ['Spam' if y == 1 else 'Not Spam' for y in y_pred[:10]],
                'Correct': ['✓' if y_test[i] == y_pred[i] else '✗' for i in range(min(10, len(y_test)))]
            })
            st.dataframe(predictions_df, use_container_width=True)
            
            st.info("💡 **Key Insight**: We can measure 'correctness' because we have true labels for comparison!")

elif page == "Unsupervised Learning":
    st.header("👥 Unsupervised Learning: Customer Segmentation")
    st.markdown("""
    **Key Insight**: We DON'T have labels. Algorithm discovers patterns independently.
    """)
    
    n_clusters = st.slider("Number of Clusters", 2, 5, 3)
    
    if st.button("Run Unsupervised Learning Demo"):
        with st.spinner("Discovering customer segments..."):
            # Generate customer data
            np.random.seed(42)
            n_customers = 300
            
            # Three natural customer groups
            budget = np.random.normal(loc=[20, 15], scale=[5, 3], size=(n_customers//3, 2))
            premium = np.random.normal(loc=[80, 5], scale=[10, 2], size=(n_customers//3, 2))
            regular = np.random.normal(loc=[50, 10], scale=[8, 3], size=(n_customers//3, 2))
            
            X = np.vstack([budget, premium, regular])
            
            # Fit KMeans
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = model.fit_predict(X)
            
            # Calculate silhouette score
            silhouette = silhouette_score(X, clusters)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Customers", len(X))
            with col2:
                st.metric("Silhouette Score", f"{silhouette:.3f}")
            
            st.info("💡 **Note**: We can't measure 'accuracy' - there's no ground truth! Instead, we use silhouette score to measure cluster quality.")
            
            # Analyze segments
            st.subheader("Discovered Customer Segments")
            segments_data = []
            for cluster_id in range(n_clusters):
                cluster_data = X[clusters == cluster_id]
                avg_purchase = cluster_data[:, 0].mean()
                avg_frequency = cluster_data[:, 1].mean()
                n_customers = len(cluster_data)
                
                if avg_purchase < 35:
                    segment_type = "Budget Shoppers"
                elif avg_purchase > 65:
                    segment_type = "Premium Shoppers"
                else:
                    segment_type = "Regular Shoppers"
                
                segments_data.append({
                    'Segment': cluster_id + 1,
                    'Type': segment_type,
                    'Customers': n_customers,
                    'Avg Purchase ($)': f"${avg_purchase:.2f}",
                    'Purchases/Month': f"{avg_frequency:.1f}"
                })
            
            segments_df = pd.DataFrame(segments_data)
            st.dataframe(segments_df, use_container_width=True)
            
            # Visualization
            st.subheader("Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
            
            for cluster_id in range(n_clusters):
                cluster_points = X[clusters == cluster_id]
                ax.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    c=[colors[cluster_id]],
                    label=f'Segment {cluster_id + 1}',
                    alpha=0.6,
                    s=50
                )
            
            # Plot cluster centers
            centers = model.cluster_centers_
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c='black',
                marker='X',
                s=200,
                label='Cluster Centers',
                edgecolors='white',
                linewidths=2
            )
            
            ax.set_xlabel('Average Purchase Amount ($)', fontsize=12)
            ax.set_ylabel('Purchases per Month', fontsize=12)
            ax.set_title('Customer Segmentation (Unsupervised Learning)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

elif page == "Comparison":
    st.header("⚖️ Supervised vs. Unsupervised Learning")
    
    comparison_data = {
        'Aspect': [
            'Training Data',
            'Algorithm Goal',
            'Evaluation',
            'Use Case Examples',
            'Cost to Implement',
            'When to Use'
        ],
        'Supervised Learning': [
            'Requires labeled data (X, y)',
            'Learn mapping: input → output',
            'Can measure accuracy/error',
            'Spam detection, image recognition',
            'High (need labeled data)',
            'When you have correct answers'
        ],
        'Unsupervised Learning': [
            'Only needs input data (X)',
            'Discover hidden patterns',
            'No ground truth to compare',
            'Customer segmentation, anomaly detection',
            'Lower (no labeling needed)',
            'When exploring data structure'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.subheader("🌍 Real-World Examples")
    st.markdown("""
    ### Netflix
    - **Supervised**: Predict ratings based on past ratings
    - **Unsupervised**: Discover similar movies for recommendations
    
    ### Tesla
    - **Supervised**: Detect objects (cars, pedestrians, signs)
    - **Unsupervised**: Discover rare driving scenarios
    
    ### Gmail
    - **Supervised**: Classify known spam patterns
    - **Unsupervised**: Detect emerging spam tactics
    """)
    
    st.subheader("🎯 Key Takeaways")
    st.markdown("""
    1. **API Difference**:
       - Supervised: `model.fit(X, y)` ← Requires labels
       - Unsupervised: `model.fit(X)` ← No labels needed
    
    2. **Evaluation Difference**:
       - Supervised: Can measure "correctness" (accuracy, error)
       - Unsupervised: Evaluate "usefulness" (silhouette, business value)
    
    3. **Data Requirement**:
       - Supervised: Need expensive labeled data
       - Unsupervised: Work with abundant unlabeled data
    
    4. **Production Reality**:
       - Most AI systems combine both approaches
       - Start with unsupervised to explore, add supervised for precision
    """)

elif page == "Interactive Demo":
    st.header("🎮 Interactive Demo")
    st.markdown("Compare both approaches side by side!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📧 Supervised Learning")
        if st.button("Run Spam Classifier", key="supervised"):
            with st.spinner("Training..."):
                np.random.seed(42)
                spam_emails = np.random.poisson(lam=[8, 6, 5, 1, 1], size=(500, 5))
                normal_emails = np.random.poisson(lam=[1, 1, 0.5, 7, 6], size=(500, 5))
                X = np.vstack([spam_emails, normal_emails])
                y = np.array([1] * 500 + [0] * 500)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = MultinomialNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                st.success(f"Accuracy: {accuracy:.2%}")
                st.code("model.fit(X, y)  # Labels provided!")
    
    with col2:
        st.subheader("👥 Unsupervised Learning")
        if st.button("Run Customer Segmentation", key="unsupervised"):
            with st.spinner("Discovering patterns..."):
                np.random.seed(42)
                budget = np.random.normal(loc=[20, 15], scale=[5, 3], size=(100, 2))
                premium = np.random.normal(loc=[80, 5], scale=[10, 2], size=(100, 2))
                regular = np.random.normal(loc=[50, 10], scale=[8, 3], size=(100, 2))
                X = np.vstack([budget, premium, regular])
                
                model = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = model.fit_predict(X)
                silhouette = silhouette_score(X, clusters)
                
                st.success(f"Silhouette Score: {silhouette:.3f}")
                st.code("model.fit(X)  # No labels needed!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Day 39** of 180-Day AI/ML Course")
st.sidebar.markdown("Next: Day 40 - Regression vs. Classification")

