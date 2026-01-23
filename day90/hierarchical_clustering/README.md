# Day 90: Hierarchical Clustering

Production-ready hierarchical clustering implementation for building multi-level content taxonomies.

## Overview

This lesson implements hierarchical clustering algorithms used in production systems like:
- Netflix's genre taxonomy (organizing 250M+ subscribers' content)
- Spotify's music clustering (6,000+ micro-genres)
- Google Scholar's research paper organization (200M+ papers)

## Features

- Multiple linkage methods (single, complete, average, Ward)
- Dendrogram visualization
- Multi-level taxonomy generation
- Production-ready API design
- Comprehensive test suite (15 tests)

## Quick Start

```bash
# Setup environment
bash setup_venv.sh
source venv/bin/activate

# Run tests
pytest test_lesson.py -v

# Run examples
python lesson_code.py
```

## Expected Output

```
Comparing Linkage Methods:
------------------------------------------------------------

SINGLE Linkage:
  Cluster sizes: [29, 30, 31]
  Number of clusters: 3

COMPLETE Linkage:
  Cluster sizes: [30, 30, 30]
  Number of clusters: 3

AVERAGE Linkage:
  Cluster sizes: [30, 30, 30]
  Number of clusters: 3

WARD Linkage:
  Cluster sizes: [30, 30, 30]
  Number of clusters: 3

============================================================
Content Taxonomy Example: Movie Genre Clustering
============================================================

Processing 100 movies with 50-dimensional embeddings...

Taxonomy Structure:
  Level 1: 2 clusters
    Cluster 0: 51 movies
    Cluster 1: 49 movies
  Level 2: 4 clusters
    Cluster 0: 22 movies
    Cluster 1: 29 movies
    Cluster 2: 26 movies
    Cluster 3: 23 movies
  Level 3: 8 clusters
    Cluster 0: 11 movies
    Cluster 1: 11 movies
    Cluster 2: 15 movies
    Cluster 3: 14 movies
    Cluster 4: 11 movies
    Cluster 5: 15 movies
    Cluster 6: 13 movies
    Cluster 7: 10 movies

Taxonomy saved to movie_taxonomy.json
Dendrogram saved to movie_dendrogram.png
```

## Usage Examples

### Basic Clustering

```python
from lesson_code import HierarchicalClusterer
import numpy as np

# Load your data
X = np.random.randn(100, 50)  # 100 items, 50 features

# Cluster with Ward linkage
clusterer = HierarchicalClusterer(linkage_method='ward', n_clusters=8)
labels = clusterer.fit_predict(X)

# Visualize dendrogram
clusterer.plot_dendrogram(save_path='dendrogram.png')
```

### Building Content Taxonomy

```python
from lesson_code import ContentTaxonomyBuilder
import numpy as np

# Content embeddings
embeddings = np.random.randn(200, 100)
item_ids = [f"item_{i}" for i in range(200)]

# Build multi-level taxonomy
builder = ContentTaxonomyBuilder(linkage_method='ward')
taxonomy = builder.build_taxonomy(embeddings, item_ids, n_levels=3)

# Access specific cluster
items = builder.get_cluster_at_level(level=2, cluster_id=3)
print(f"Cluster contains {len(items)} items")

# Save taxonomy
builder.save_taxonomy('taxonomy.json')
```

## Files Generated

- `movie_taxonomy.json` - Multi-level hierarchy structure
- `movie_dendrogram.png` - Visualization of clustering tree

## Key Concepts

### Linkage Methods

- **Single**: Minimum distance between clusters (creates chain-like clusters)
- **Complete**: Maximum distance between clusters (creates compact clusters)
- **Average**: Average distance between all pairs (balanced approach)
- **Ward**: Minimizes variance when merging (most common in production)

### When to Use Hierarchical Clustering

Use hierarchical clustering when:
- You don't know the optimal number of clusters
- You need multi-resolution clustering
- Your data has natural hierarchical structure
- You want to visualize cluster relationships

Use K-means when:
- You know the number of clusters
- You need fast clustering on large datasets
- You only need flat clustering

## Performance

- Dataset: 100 items, 50 dimensions
- Clustering time: <1 second
- Test suite: 15 tests in ~2 seconds
- Memory: O(n²) for distance matrix

## Next Steps

Tomorrow (Day 91): Principal Component Analysis (PCA) for dimensionality reduction
- Reduce 1000D embeddings to 50D
- Combine PCA + hierarchical clustering
- Build complete taxonomy generation pipeline

## References

- Netflix Tech Blog: Genre Taxonomy
- Spotify Research: Music Clustering
- Scikit-learn Documentation: Hierarchical Clustering
