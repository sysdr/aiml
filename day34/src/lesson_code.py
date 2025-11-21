"""
Day 34: DataFrame Indexing, Slicing, and Filtering
Building a Smart Content Recommendation Filter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set random seed for reproducibility
np.random.seed(42)

class ContentRecommendationEngine:
    """
    A production-style content recommendation filter that demonstrates
    real-world DataFrame operations used in streaming platforms.
    """
    
    def __init__(self, n_content_items: int = 1000):
        """Initialize with synthetic content engagement data."""
        self.df = self._generate_content_data(n_content_items)
        print(f"📊 Initialized with {len(self.df)} content items")
    
    def _generate_content_data(self, n: int) -> pd.DataFrame:
        """Generate realistic content engagement data."""
        data = {
            'content_id': range(1, n + 1),
            'title': [f'Content_{i}' for i in range(1, n + 1)],
            'views': np.random.randint(100, 100000, n),
            'completion_rate': np.random.uniform(0.1, 1.0, n),
            'rating': np.random.uniform(1.0, 5.0, n),
            'category': np.random.choice(
                ['tech', 'entertainment', 'education', 'sports'], n
            ),
            'upload_date': pd.date_range('2024-01-01', periods=n, freq='H'),
            'likes': np.random.randint(10, 10000, n),
            'shares': np.random.randint(5, 5000, n),
            'watch_time_minutes': np.random.randint(5, 120, n)
        }
        
        df = pd.DataFrame(data)
        
        # Set content_id as index for efficient lookups
        df.set_index('content_id', inplace=True)
        
        return df
    
    def demo_indexing(self) -> None:
        """Demonstrate three types of indexing operations."""
        print("\n" + "="*60)
        print("1. INDEXING: Direct Access to Data")
        print("="*60)
        
        # loc - label-based indexing
        print("\n🎯 Using .loc[] (label-based):")
        content_42 = self.df.loc[42]
        print(f"Content ID 42: {content_42['title']}")
        print(f"  Views: {content_42['views']:,}")
        print(f"  Rating: {content_42['rating']:.2f}")
        
        # iloc - position-based indexing
        print("\n🎯 Using .iloc[] (position-based):")
        first_10 = self.df.iloc[0:10]
        print(f"First 10 content items:")
        print(first_10[['title', 'views', 'rating']].to_string())
        
        # at - fast scalar access
        print("\n🎯 Using .at[] (fast scalar access):")
        rating = self.df.at[100, 'rating']
        print(f"Content 100 rating: {rating:.2f}")
        
        # Performance comparison
        import time
        
        # Time loc access
        start = time.time()
        for _ in range(1000):
            _ = self.df.loc[500, 'rating']
        loc_time = time.time() - start
        
        # Time at access
        start = time.time()
        for _ in range(1000):
            _ = self.df.at[500, 'rating']
        at_time = time.time() - start
        
        print(f"\n⚡ Performance comparison (1000 lookups):")
        print(f"  .loc[]: {loc_time*1000:.2f}ms")
        print(f"  .at[]:  {at_time*1000:.2f}ms")
        print(f"  Speedup: {loc_time/at_time:.1f}x faster")
    
    def demo_slicing(self) -> None:
        """Demonstrate data slicing operations."""
        print("\n" + "="*60)
        print("2. SLICING: Extracting Data Segments")
        print("="*60)
        
        # Column slicing
        print("\n📊 Column selection (feature engineering):")
        engagement_features = self.df[['views', 'likes', 'shares', 'completion_rate']]
        print(engagement_features.head())
        
        # Row slicing by position
        print("\n📊 Row slicing (recent content):")
        recent_content = self.df.iloc[-50:]
        print(f"Last 50 uploads: {len(recent_content)} items")
        print(f"Average views: {recent_content['views'].mean():,.0f}")
        
        # Row slicing by range
        print("\n📊 Range slicing (validation set):")
        validation_set = self.df.iloc[800:900]
        print(f"Validation set size: {len(validation_set)}")
        
        # Multiple column slicing
        print("\n📊 Multi-column slice (model features):")
        model_features = self.df[['views', 'completion_rate', 'rating', 'watch_time_minutes']]
        print(f"Feature matrix shape: {model_features.shape}")
    
    def demo_filtering(self) -> None:
        """Demonstrate boolean filtering operations."""
        print("\n" + "="*60)
        print("3. FILTERING: Conditional Data Selection")
        print("="*60)
        
        # Single condition
        print("\n🔍 Single condition filter:")
        popular_content = self.df[self.df['views'] > 50000]
        print(f"Popular content (>50K views): {len(popular_content)} items")
        print(f"Percentage: {len(popular_content)/len(self.df)*100:.1f}%")
        
        # Multiple conditions (AND)
        print("\n🔍 Multiple conditions (AND):")
        high_quality = self.df[
            (self.df['completion_rate'] > 0.7) &
            (self.df['rating'] > 4.0) &
            (self.df['views'] > 5000)
        ]
        print(f"High-quality content: {len(high_quality)} items")
        print(f"Average rating: {high_quality['rating'].mean():.2f}")
        
        # Multiple conditions (OR)
        print("\n🔍 Multiple conditions (OR):")
        viral_or_engaging = self.df[
            (self.df['shares'] > 1000) |
            (self.df['completion_rate'] > 0.9)
        ]
        print(f"Viral or highly engaging: {len(viral_or_engaging)} items")
        
        # Category filtering
        print("\n🔍 Category-based filtering:")
        for category in self.df['category'].unique():
            cat_content = self.df[self.df['category'] == category]
            avg_views = cat_content['views'].mean()
            print(f"  {category:15s}: {len(cat_content):3d} items, avg {avg_views:7,.0f} views")
    
    def build_recommendation_filter(self, 
                                   user_category: str = 'tech',
                                   min_quality: float = 0.7) -> pd.DataFrame:
        """
        Build a production-style recommendation filter.
        
        This mimics how Netflix/YouTube select content for users.
        """
        print("\n" + "="*60)
        print("4. PRODUCTION FILTER: Building Recommendations")
        print("="*60)
        
        print(f"\n🎬 Building recommendations for:")
        print(f"  Category: {user_category}")
        print(f"  Min quality: {min_quality}")
        
        # Step 1: Filter by category preference
        category_content = self.df[self.df['category'] == user_category]
        print(f"\n  Step 1 - Category filter: {len(category_content)} items")
        
        # Step 2: Apply quality threshold
        quality_content = category_content[
            category_content['completion_rate'] > min_quality
        ]
        print(f"  Step 2 - Quality filter: {len(quality_content)} items")
        
        # Step 3: Sort by engagement score (composite metric)
        quality_content = quality_content.copy()
        quality_content['engagement_score'] = (
            quality_content['completion_rate'] * 0.4 +
            (quality_content['rating'] / 5.0) * 0.3 +
            (quality_content['likes'] / quality_content['likes'].max()) * 0.3
        )
        
        recommendations = quality_content.sort_values(
            'engagement_score', 
            ascending=False
        ).head(10)
        
        print(f"  Step 3 - Top 10 ranked by engagement")
        
        print("\n📋 Recommended Content:")
        print(recommendations[['title', 'views', 'rating', 'completion_rate', 'engagement_score']].to_string())
        
        return recommendations
    
    def analyze_data_quality(self) -> Dict:
        """Analyze dataset quality using filtering operations."""
        print("\n" + "="*60)
        print("5. DATA QUALITY ANALYSIS")
        print("="*60)
        
        total = len(self.df)
        
        # Find low-quality content
        low_quality = self.df[
            (self.df['completion_rate'] < 0.3) &
            (self.df['rating'] < 2.5)
        ]
        
        # Find high-engagement content
        high_engagement = self.df[
            (self.df['completion_rate'] > 0.8) &
            (self.df['views'] > 10000)
        ]
        
        # Find underperforming content
        underperforming = self.df[
            (self.df['rating'] > 4.0) &
            (self.df['views'] < 1000)
        ]
        
        results = {
            'total': total,
            'low_quality': len(low_quality),
            'high_engagement': len(high_engagement),
            'underperforming': len(underperforming)
        }
        
        print(f"\n📊 Dataset Quality Metrics:")
        print(f"  Total content: {total}")
        print(f"  Low quality (remove): {results['low_quality']} ({results['low_quality']/total*100:.1f}%)")
        print(f"  High engagement (promote): {results['high_engagement']} ({results['high_engagement']/total*100:.1f}%)")
        print(f"  Underperforming (needs promotion): {results['underperforming']} ({results['underperforming']/total*100:.1f}%)")
        
        return results
    
    def visualize_filtering_impact(self) -> None:
        """Visualize the impact of different filtering strategies."""
        print("\n" + "="*60)
        print("6. VISUALIZATION: Filtering Impact")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Impact of DataFrame Filtering on Content Selection', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Views distribution
        axes[0, 0].hist(self.df['views'], bins=50, alpha=0.7, color='blue', label='All Content')
        high_quality = self.df[
            (self.df['completion_rate'] > 0.7) & 
            (self.df['rating'] > 4.0)
        ]
        axes[0, 0].hist(high_quality['views'], bins=50, alpha=0.7, color='green', label='High Quality')
        axes[0, 0].set_xlabel('Views')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Views Distribution: All vs High Quality')
        axes[0, 0].legend()
        
        # Plot 2: Category breakdown
        category_counts = self.df['category'].value_counts()
        axes[0, 1].bar(category_counts.index, category_counts.values, color='coral')
        axes[0, 1].set_xlabel('Category')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Content Distribution by Category')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Completion rate vs Rating
        axes[1, 0].scatter(self.df['completion_rate'], self.df['rating'], 
                          alpha=0.3, s=10, color='purple')
        axes[1, 0].axvline(x=0.7, color='r', linestyle='--', label='Quality Threshold')
        axes[1, 0].axhline(y=4.0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Completion Rate')
        axes[1, 0].set_ylabel('Rating')
        axes[1, 0].set_title('Quality Filter: Completion Rate vs Rating')
        axes[1, 0].legend()
        
        # Plot 4: Filtering funnel
        filter_stages = ['Total', 'Views>5K', 'Complete>0.7', 'Rating>4.0', 'All Filters']
        counts = [
            len(self.df),
            len(self.df[self.df['views'] > 5000]),
            len(self.df[self.df['completion_rate'] > 0.7]),
            len(self.df[self.df['rating'] > 4.0]),
            len(self.df[(self.df['views'] > 5000) & 
                       (self.df['completion_rate'] > 0.7) & 
                       (self.df['rating'] > 4.0)])
        ]
        axes[1, 1].bar(filter_stages, counts, color=['gray', 'blue', 'green', 'orange', 'red'])
        axes[1, 1].set_ylabel('Content Count')
        axes[1, 1].set_title('Filtering Funnel: Progressive Data Reduction')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('filtering_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'filtering_analysis.png'")
        plt.close()


def main():
    """Run the complete Day 34 lesson."""
    print("\n" + "="*70)
    print("  DAY 34: DATAFRAME INDEXING, SLICING, AND FILTERING")
    print("  Building Production-Grade Data Selection Systems")
    print("="*70)
    
    # Initialize recommendation engine
    engine = ContentRecommendationEngine(n_content_items=1000)
    
    # Demonstrate core concepts
    engine.demo_indexing()
    engine.demo_slicing()
    engine.demo_filtering()
    
    # Build production filter
    recommendations = engine.build_recommendation_filter(
        user_category='tech',
        min_quality=0.7
    )
    
    # Analyze data quality
    quality_metrics = engine.analyze_data_quality()
    
    # Create visualizations
    engine.visualize_filtering_impact()
    
    print("\n" + "="*70)
    print("✅ Day 34 Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. .loc[] for label-based indexing (readable)")
    print("  2. .iloc[] for position-based indexing (flexible)")
    print("  3. .at[] for fast scalar access (performance)")
    print("  4. Boolean filtering for data quality control")
    print("  5. Combined operations for production-grade systems")
    print("\n💡 These operations power every recommendation system at scale!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
