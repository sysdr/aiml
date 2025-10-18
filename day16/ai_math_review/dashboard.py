"""
AI/ML Math Review Dashboard
Interactive web dashboard for the math review project
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
from sympy import symbols, diff, solve

# Page configuration
st.set_page_config(
    page_title="AI/ML Math Review Dashboard",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .problem-box {
        background-color: #fff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product

def create_gradient_descent_plot():
    """Create interactive gradient descent visualization"""
    def loss_function(x):
        return x**2 - 4*x + 5
    
    def gradient(x):
        return 2*x - 4
    
    # Generate data for plotting
    x_range = np.linspace(-1, 5, 100)
    y_range = [loss_function(x) for x in x_range]
    
    # Gradient descent steps
    x = 0.0
    learning_rate = 0.1
    history_x = [x]
    history_y = [loss_function(x)]
    
    for step in range(10):
        grad = gradient(x)
        x = x - learning_rate * grad
        history_x.append(x)
        history_y.append(loss_function(x))
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add loss function curve
    fig.add_trace(go.Scatter(
        x=x_range, y=y_range,
        mode='lines',
        name='Loss Function',
        line=dict(color='blue', width=3)
    ))
    
    # Add gradient descent path
    fig.add_trace(go.Scatter(
        x=history_x, y=history_y,
        mode='markers+lines',
        name='Gradient Descent Path',
        line=dict(color='red', width=2),
        marker=dict(size=8, color='red')
    ))
    
    # Add minimum point
    fig.add_trace(go.Scatter(
        x=[2], y=[1],
        mode='markers',
        name='Global Minimum',
        marker=dict(size=15, color='green', symbol='star')
    ))
    
    fig.update_layout(
        title='Gradient Descent Optimization',
        xaxis_title='Parameter Value (x)',
        yaxis_title='Loss Function Value',
        hovermode='closest',
        height=500
    )
    
    return fig, history_x, history_y

def create_similarity_heatmap():
    """Create user similarity heatmap"""
    users = ['User A', 'User B', 'User C']
    user_vectors = [
        [4.2, 3.8, 2.1, 4.9, 1.3],
        [4.1, 3.9, 2.3, 4.7, 1.1], 
        [1.2, 4.8, 4.9, 1.3, 0.8]
    ]
    
    similarity_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            similarity_matrix[i][j] = cosine_similarity(
                np.array(user_vectors[i]), 
                np.array(user_vectors[j])
            )
    
    fig = px.imshow(
        similarity_matrix,
        x=users,
        y=users,
        color_continuous_scale='viridis',
        title='User Similarity Matrix',
        aspect='auto'
    )
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            fig.add_annotation(
                x=j, y=i,
                text=f'{similarity_matrix[i, j]:.3f}',
                showarrow=False,
                font=dict(color='white', size=12)
            )
    
    fig.update_layout(height=400)
    return fig, similarity_matrix

def create_neural_network_diagram():
    """Create neural network visualization"""
    fig = go.Figure()
    
    # Input layer nodes
    input_x = [0] * 3
    input_y = [0, 1, 2]
    fig.add_trace(go.Scatter(
        x=input_x, y=input_y,
        mode='markers',
        marker=dict(size=20, color='lightblue'),
        name='Input Layer',
        text=['Feature 1', 'Feature 2', 'Feature 3'],
        textposition='middle right'
    ))
    
    # Output layer nodes
    output_x = [2] * 2
    output_y = [0.5, 1.5]
    fig.add_trace(go.Scatter(
        x=output_x, y=output_y,
        mode='markers',
        marker=dict(size=20, color='lightgreen'),
        name='Output Layer',
        text=['Neuron 1', 'Neuron 2'],
        textposition='middle left'
    ))
    
    # Add connections
    for i in range(3):
        for j in range(2):
            fig.add_trace(go.Scatter(
                x=[input_x[i], output_x[j]],
                y=[input_y[i], output_y[j]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title='Neural Network Architecture',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        showlegend=True
    )
    
    return fig

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🧮 AI/ML Math Review Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    section = st.sidebar.selectbox(
        "Choose a section:",
        ["📈 Overview", "🔢 Vector Operations", "🏗️ Matrix Operations", 
         "📐 Calculus", "🚀 Advanced Applications", "📊 Visualizations"]
    )
    
    # Overview Section
    if section == "📈 Overview":
        st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Problems Covered", "8", "Complete")
        
        with col2:
            st.metric("Sections", "4", "All Topics")
        
        with col3:
            st.metric("Concepts", "12+", "Math Foundations")
        
        with col4:
            st.metric("Visualizations", "3", "Interactive")
        
        st.markdown("""
        ### 🎯 Learning Objectives Achieved
        
        This comprehensive math review covers the essential mathematical foundations for AI and Machine Learning:
        
        - **Vector Operations**: User similarity, feature enhancement
        - **Matrix Operations**: Neural networks, data transformations  
        - **Calculus**: Optimization, gradient descent, partial derivatives
        - **Advanced Topics**: PCA, eigenvalues, backpropagation
        
        ### 💡 Key Insights
        
        - Mathematical concepts directly power modern AI applications
        - Each problem demonstrates real-world AI/ML use cases
        - Interactive visualizations help understand complex concepts
        - Foundation knowledge prepares you for advanced ML topics
        """)
    
    # Vector Operations Section
    elif section == "🔢 Vector Operations":
        st.markdown('<h2 class="section-header">Vector Operations for AI</h2>', unsafe_allow_html=True)
        
        st.markdown("### Problem 1: User Preference Similarity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Scenario**: Netflix-style recommendation system
            
            **User Preferences** (ratings for genres):
            - Action, Comedy, Drama, Sci-fi, Horror
            """)
            
            user_a = np.array([4.2, 3.8, 2.1, 4.9, 1.3])
            user_b = np.array([4.1, 3.9, 2.3, 4.7, 1.1])
            user_c = np.array([1.2, 4.8, 4.9, 1.3, 0.8])
            
            st.write("**User A:**", user_a)
            st.write("**User B:**", user_b)
            st.write("**User C:**", user_c)
        
        with col2:
            sim_ab = cosine_similarity(user_a, user_b)
            sim_ac = cosine_similarity(user_a, user_c)
            
            st.metric("Similarity A-B", f"{sim_ab:.3f}", "Very High")
            st.metric("Similarity A-C", f"{sim_ac:.3f}", "Moderate")
            
            if sim_ab > sim_ac:
                st.success("✅ Users A and B are more similar - better for recommendations!")
        
        # Similarity heatmap
        fig_heatmap, sim_matrix = create_similarity_heatmap()
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("### Problem 2: Feature Vector Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Image Enhancement Example**")
            image_features = np.array([0.8, 0.2, 0.9, 0.1])
            adjustment = np.array([0.1, -0.05, 0.05, -0.02])
            
            st.write("**Original Features:**", image_features)
            st.write("**Enhancement Vector:**", adjustment)
        
        with col2:
            enhanced_features = image_features + adjustment
            st.write("**Enhanced Features:**", enhanced_features)
            st.info("🖼️ This is how AI enhances images!")
    
    # Matrix Operations Section
    elif section == "🏗️ Matrix Operations":
        st.markdown('<h2 class="section-header">Matrix Operations for Neural Networks</h2>', unsafe_allow_html=True)
        
        st.markdown("### Problem 3: Neural Network Layer Forward Pass")
        
        # Neural network diagram
        fig_nn = create_neural_network_diagram()
        st.plotly_chart(fig_nn, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Forward Pass Calculation**")
            X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
            W = np.array([[0.1, 0.5], [0.2, 0.3], [0.4, 0.1]])
            b = np.array([0.1, 0.2])
            
            st.write("**Input X:**")
            st.dataframe(pd.DataFrame(X, columns=['Feature 1', 'Feature 2', 'Feature 3']))
            
            st.write("**Weights W:**")
            st.dataframe(pd.DataFrame(W, columns=['Neuron 1', 'Neuron 2']))
        
        with col2:
            Y = np.dot(X, W) + b
            st.write("**Bias b:**", b)
            st.write("**Output Y = XW + b:**")
            st.dataframe(pd.DataFrame(Y, columns=['Neuron 1', 'Neuron 2']))
            
            st.success("🧠 This is how neural networks transform data!")
        
        st.markdown("### Problem 4: Image Rotation Matrix")
        
        col1, col2 = st.columns(2)
        
        with col1:
            angle = st.slider("Rotation Angle (degrees)", 0, 360, 45)
            angle_rad = np.radians(angle)
            
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            
            st.write("**Rotation Matrix:**")
            st.dataframe(pd.DataFrame(rotation_matrix))
        
        with col2:
            point = np.array([1, 1])
            rotated_point = np.dot(rotation_matrix, point)
            
            st.write("**Original Point:**", point)
            st.write("**Rotated Point:**", rotated_point)
            st.info("🔄 This is how AI rotates images for data augmentation!")
    
    # Calculus Section
    elif section == "📐 Calculus":
        st.markdown('<h2 class="section-header">Calculus for AI Optimization</h2>', unsafe_allow_html=True)
        
        st.markdown("### Problem 5: Gradient Descent Optimization")
        
        fig_gd, history_x, history_y = create_gradient_descent_plot()
        st.plotly_chart(fig_gd, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Gradient Descent Steps**")
            df_steps = pd.DataFrame({
                'Step': range(1, len(history_x)),
                'X Value': [f"{x:.3f}" for x in history_x[1:]],
                'Loss': [f"{y:.3f}" for y in history_y[1:]]
            })
            st.dataframe(df_steps)
        
        with col2:
            st.metric("Final X Value", f"{history_x[-1]:.3f}", "Target: 2.0")
            st.metric("Final Loss", f"{history_y[-1]:.3f}", "Minimum: 1.0")
            st.success("📉 This is how AI models learn by minimizing loss!")
        
        st.markdown("### Problem 6: Partial Derivatives for Multi-Parameter Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Symbolic Computation**")
            x, y = symbols('x y')
            f = x**2 + 2*y**2 + x*y - 4*x - 2*y + 5
            
            st.latex(f"f(x,y) = {sp.latex(f)}")
            
            df_dx = diff(f, x)
            df_dy = diff(f, y)
            
            st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(df_dx)}")
            st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(df_dy)}")
        
        with col2:
            critical_points = solve([df_dx, df_dy], [x, y])
            st.markdown("**Critical Points**")
            st.write(critical_points)
            st.success("🎯 This is how AI finds optimal parameters!")
    
    # Advanced Applications Section
    elif section == "🚀 Advanced Applications":
        st.markdown('<h2 class="section-header">Advanced AI Math Applications</h2>', unsafe_allow_html=True)
        
        st.markdown("### Problem 7: Eigenvalues and Data Compression Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            np.random.seed(42)
            data = np.random.randn(50, 2)
            data[:, 1] = data[:, 0] + 0.5 * np.random.randn(50)
            
            cov_matrix = np.cov(data.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            st.write("**Covariance Matrix:**")
            st.dataframe(pd.DataFrame(cov_matrix))
        
        with col2:
            st.write("**Eigenvalues:**", eigenvalues)
            st.write("**Data Shape:**", data.shape)
            
            # Create PCA visualization
            fig_pca = px.scatter(
                x=data[:, 0], y=data[:, 1],
                title='Sample Data for PCA',
                labels={'x': 'Feature 1', 'y': 'Feature 2'}
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            
            st.info("📊 This is how AI compresses data while keeping important info!")
        
        st.markdown("### Problem 8: Chain Rule for Neural Network Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Forward Pass**")
            x_val, y_val = 2.0, 3.0
            z_val = x_val * y_val
            loss_val = z_val**2
            
            st.write(f"x = {x_val}")
            st.write(f"y = {y_val}")
            st.write(f"z = x × y = {z_val}")
            st.write(f"loss = z² = {loss_val}")
        
        with col2:
            st.markdown("**Backward Pass (Chain Rule)**")
            dloss_dz = 2 * z_val
            dz_dx = y_val
            dz_dy = x_val
            
            dloss_dx = dloss_dz * dz_dx
            dloss_dy = dloss_dz * dz_dy
            
            st.write(f"∂loss/∂z = 2z = {dloss_dz}")
            st.write(f"∂z/∂x = y = {dz_dx}")
            st.write(f"∂z/∂y = x = {dz_dy}")
            st.write(f"∂loss/∂x = {dloss_dx}")
            st.write(f"∂loss/∂y = {dloss_dy}")
            
            st.success("⛓️ This is backpropagation - how neural networks learn!")
    
    # Visualizations Section
    elif section == "📊 Visualizations":
        st.markdown('<h2 class="section-header">Interactive Visualizations</h2>', unsafe_allow_html=True)
        
        st.markdown("### All Visualizations Combined")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Gradient Descent", "User Similarity", "Neural Network"])
        
        with tab1:
            fig_gd, _, _ = create_gradient_descent_plot()
            st.plotly_chart(fig_gd, use_container_width=True)
            st.markdown("**Interactive gradient descent optimization showing how AI models learn.**")
        
        with tab2:
            fig_heatmap, _ = create_similarity_heatmap()
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.markdown("**User similarity matrix for recommendation systems.**")
        
        with tab3:
            fig_nn = create_neural_network_diagram()
            st.plotly_chart(fig_nn, use_container_width=True)
            st.markdown("**Neural network architecture visualization.**")
        
        # Summary metrics
        st.markdown("### 📈 Performance Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Problems Solved", "8/8", "100%")
        
        with col2:
            st.metric("Concepts Mastered", "12+", "Complete")
        
        with col3:
            st.metric("Visualizations", "3", "Interactive")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🧮 AI/ML Math Review Dashboard | Built with Streamlit</p>
        <p>Master the mathematical foundations of Artificial Intelligence and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
