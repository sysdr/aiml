"""
Day 36: EDA Dashboard
Interactive Streamlit dashboard for e-commerce data analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="EDA Dashboard - E-Commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load e-commerce data, generate if not exists"""
    data_path = Path("eda_output/ecommerce_data.csv")
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        # Generate data if it doesn't exist
        from lesson_code import generate_ecommerce_dataset
        df = generate_ecommerce_dataset(n_samples=100000)
        output_dir = Path("eda_output")
        output_dir.mkdir(exist_ok=True)
        df.to_csv(data_path, index=False)
        return df

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 E-Commerce EDA Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        df_filtered = df[
            (df['timestamp'].dt.date >= date_range[0]) & 
            (df['timestamp'].dt.date <= date_range[1])
        ]
    else:
        df_filtered = df.copy()
    
    # Device filter
    devices = st.sidebar.multiselect(
        "Device Type",
        options=df['device_type'].unique(),
        default=df['device_type'].unique()
    )
    df_filtered = df_filtered[df_filtered['device_type'].isin(devices)]
    
    # Traffic source filter
    sources = st.sidebar.multiselect(
        "Traffic Source",
        options=df['traffic_source'].unique(),
        default=df['traffic_source'].unique()
    )
    df_filtered = df_filtered[df_filtered['traffic_source'].isin(sources)]
    
    # Customer segment filter
    segments = st.sidebar.multiselect(
        "Customer Segment",
        options=df['customer_segment'].unique(),
        default=df['customer_segment'].unique()
    )
    df_filtered = df_filtered[df_filtered['customer_segment'].isin(segments)]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📈 Showing {len(df_filtered):,} of {len(df):,} records")
    
    # Key Metrics
    st.header("📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_sessions = len(df_filtered)
        st.metric("Total Sessions", f"{total_sessions:,}")
    
    with col2:
        conversion_rate = (df_filtered['purchased'].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    
    with col3:
        total_revenue = df_filtered['revenue'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col4:
        avg_revenue = df_filtered[df_filtered['purchased']]['revenue'].mean() if df_filtered['purchased'].sum() > 0 else 0
        st.metric("Avg Order Value", f"${avg_revenue:.2f}")
    
    with col5:
        avg_pages = df_filtered['pages_viewed'].mean()
        st.metric("Avg Pages/Visit", f"{avg_pages:.1f}")
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "💰 Revenue Analysis", 
        "👥 User Behavior", 
        "🔗 Correlations", 
        "📋 Data Quality"
    ])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Summary")
            st.dataframe(df_filtered.describe(), use_container_width=True)
        
        with col2:
            st.subheader("Data Types & Info")
            info_dict = {
                'Column': df_filtered.dtypes.index.tolist(),
                'Data Type': df_filtered.dtypes.values.astype(str).tolist(),
                'Non-Null Count': df_filtered.count().values.tolist(),
                'Null Count': df_filtered.isnull().sum().values.tolist()
            }
            info_df = pd.DataFrame(info_dict)
            st.dataframe(info_df, use_container_width=True)
        
        st.subheader("Sample Data")
        st.dataframe(df_filtered.head(100), use_container_width=True)
    
    with tab2:
        st.header("💰 Revenue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue Over Time")
            revenue_daily = df_filtered.groupby(df_filtered['timestamp'].dt.date)['revenue'].sum()
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(revenue_daily.index, revenue_daily.values, linewidth=2, color='#1f77b4')
            ax.fill_between(revenue_daily.index, revenue_daily.values, alpha=0.3)
            ax.set_title('Daily Revenue Trend', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Revenue ($)')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Revenue by Traffic Source")
            revenue_by_source = df_filtered.groupby('traffic_source')['revenue'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = sns.color_palette("husl", len(revenue_by_source))
            ax.barh(revenue_by_source.index, revenue_by_source.values, color=colors)
            ax.set_title('Total Revenue by Traffic Source', fontsize=14, fontweight='bold')
            ax.set_xlabel('Revenue ($)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Revenue by Device Type")
            revenue_by_device = df_filtered.groupby('device_type')['revenue'].sum()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(revenue_by_device.values, labels=revenue_by_device.index, autopct='%1.1f%%', 
                   startangle=90, colors=sns.color_palette("husl", len(revenue_by_device)))
            ax.set_title('Revenue Distribution by Device', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col4:
            st.subheader("Revenue by Customer Segment")
            revenue_by_segment = df_filtered.groupby('customer_segment')['revenue'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(revenue_by_segment.index, revenue_by_segment.values, color=sns.color_palette("husl", len(revenue_by_segment)))
            ax.set_title('Revenue by Customer Segment', fontsize=14, fontweight='bold')
            ax.set_xlabel('Customer Segment')
            ax.set_ylabel('Revenue ($)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    with tab3:
        st.header("👥 User Behavior Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pages Viewed Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_filtered['pages_viewed'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(df_filtered['pages_viewed'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df_filtered["pages_viewed"].mean():.1f}')
            ax.axvline(df_filtered['pages_viewed'].median(), color='green', linestyle='--', 
                      label=f'Median: {df_filtered["pages_viewed"].median():.1f}')
            ax.set_title('Distribution of Pages Viewed', fontsize=14, fontweight='bold')
            ax.set_xlabel('Pages Viewed')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Time on Site Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_filtered['time_on_site_seconds'] / 60, bins=50, color='coral', edgecolor='black', alpha=0.7)
            ax.axvline(df_filtered['time_on_site_seconds'].mean() / 60, color='red', linestyle='--', 
                      label=f'Mean: {df_filtered["time_on_site_seconds"].mean() / 60:.1f} min')
            ax.set_title('Distribution of Time on Site', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("User Age Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_filtered['user_age'], bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
            ax.axvline(df_filtered['user_age'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df_filtered["user_age"].mean():.1f}')
            ax.set_title('User Age Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Age')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col4:
            st.subheader("Conversion by Segment")
            conversion_by_segment = df_filtered.groupby('customer_segment')['purchased'].mean() * 100
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['green' if x > conversion_by_segment.mean() else 'red' for x in conversion_by_segment.values]
            ax.bar(conversion_by_segment.index, conversion_by_segment.values, color=colors, alpha=0.7)
            ax.axhline(conversion_by_segment.mean(), color='blue', linestyle='--', 
                      label=f'Average: {conversion_by_segment.mean():.1f}%')
            ax.set_title('Conversion Rate by Customer Segment', fontsize=14, fontweight='bold')
            ax.set_xlabel('Customer Segment')
            ax.set_ylabel('Conversion Rate (%)')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.subheader("Traffic Source Performance")
        traffic_metrics = df_filtered.groupby('traffic_source').agg({
            'purchased': ['sum', 'mean'],
            'revenue': 'sum',
            'pages_viewed': 'mean',
            'time_on_site_seconds': 'mean'
        }).round(2)
        traffic_metrics.columns = ['Total Purchases', 'Conversion Rate', 'Total Revenue', 'Avg Pages', 'Avg Time (sec)']
        traffic_metrics['Conversion Rate'] = traffic_metrics['Conversion Rate'] * 100
        st.dataframe(traffic_metrics, use_container_width=True)
    
    with tab4:
        st.header("🔗 Correlation Analysis")
        
        # Select only numerical columns
        numerical_cols = df_filtered.select_dtypes(include=[np.number]).columns
        corr_data = df_filtered[numerical_cols]
        
        if len(corr_data.columns) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(12, 10))
            corr_matrix = corr_data.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True, 
                       linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.subheader("Correlation Table")
            st.dataframe(corr_matrix, use_container_width=True)
            
            # Strong correlations
            st.subheader("Strong Correlations (|r| > 0.7)")
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corrs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': f"{corr_val:.3f}"
                        })
            
            if strong_corrs:
                st.dataframe(pd.DataFrame(strong_corrs), use_container_width=True)
            else:
                st.info("No strong correlations found (|r| > 0.7)")
        else:
            st.warning("Insufficient numerical columns for correlation analysis")
    
    with tab5:
        st.header("📋 Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Values")
            missing = df_filtered.isnull().sum()
            missing_pct = (missing / len(df_filtered) * 100).round(2)
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing %': missing_pct.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
            
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(missing_df['Column'], missing_df['Missing %'], color='coral')
                ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
                ax.set_xlabel('Missing Percentage (%)')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.success("✅ No missing values found!")
        
        with col2:
            st.subheader("Data Completeness")
            total_cells = df_filtered.shape[0] * df_filtered.shape[1]
            missing_cells = df_filtered.isnull().sum().sum()
            completeness = (1 - missing_cells / total_cells) * 100
            
            st.metric("Overall Completeness", f"{completeness:.2f}%")
            st.metric("Total Cells", f"{total_cells:,}")
            st.metric("Missing Cells", f"{missing_cells:,}")
        
        st.subheader("Outlier Detection (3×IQR Method)")
        numerical_cols = df_filtered.select_dtypes(include=[np.number]).columns
        outliers_summary = []
        
        for col in numerical_cols:
            Q1 = df_filtered[col].quantile(0.25)
            Q3 = df_filtered[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_count = ((df_filtered[col] < lower_bound) | (df_filtered[col] > upper_bound)).sum()
            outlier_pct = (outlier_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
            
            if outlier_count > 0:
                outliers_summary.append({
                    'Column': col,
                    'Outlier Count': outlier_count,
                    'Outlier %': f"{outlier_pct:.2f}%",
                    'Normal Range': f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                })
        
        if outliers_summary:
            st.dataframe(pd.DataFrame(outliers_summary), use_container_width=True)
        else:
            st.success("✅ No significant outliers detected!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📊 EDA Dashboard | Day 36: Exploratory Data Analysis Project</p>
        <p>Generated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

