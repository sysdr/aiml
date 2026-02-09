# Movie Recommender System (Days 106-112)

Production-ready hybrid recommendation system combining collaborative and content-based filtering.

## 🎯 Project Overview

This 7-day project implements a movie recommendation engine mirroring the architecture used by Netflix, Spotify, and YouTube. The system combines collaborative filtering (learning from user behavior patterns) with content-based filtering (analyzing item features) to generate personalized recommendations.

### What We Built

- **Data Pipeline**: Loads and preprocesses MovieLens 100K dataset (100,000 ratings, 1,682 movies, 943 users)
- **Collaborative Filtering**: Matrix factorization using SVD to learn latent user/item factors
- **Content-Based Filtering**: Feature similarity using genre embeddings and release year
- **Hybrid System**: Adaptive blending that adjusts weights based on user experience
- **Evaluation Framework**: Comprehensive metrics (RMSE, Precision@K, Recall@K, Coverage, Diversity)
- **Production Architecture**: Modular design ready for scale

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip
- 2GB free disk space (for dataset)

### Installation

```bash
# 1. Make install script executable
chmod +x install.sh

# 2. Run setup (creates venv, installs dependencies, downloads dataset)
./install.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Run the recommender system
python main.py
```

## 🧪 Running Tests

```bash
# Run all tests with coverage
pytest -v tests/ --cov=. --cov-report=term-missing

# Run specific test class
pytest -v tests/test_recommender.py::TestCollaborativeFilter
```

## 📊 Architecture

```
User Request
    ↓
Data Pipeline
    ↓
    ├─→ Collaborative Filter (SVD) ──→ Collaborative Score
    └─→ Content-Based Filter ──→ Content Score
              ↓
         Hybrid Layer (α-blending)
              ↓
         Top-N Ranking
              ↓
         Recommendations
```

### Key Components

**Collaborative Filtering** (`models/collaborative_filtering.py`)
- Matrix factorization using SVD
- Learns 50 latent factors per user/item
- Handles cold-start by returning global mean

**Content-Based Filtering** (`models/content_based.py`)
- Cosine similarity between item features
- Enables instant recommendations for new items

**Hybrid Recommender** (`models/hybrid_recommender.py`)
- Adaptive blending: α ∈ [0.2, 0.8]
- New users: 20% collaborative, 80% content
- Established users: 80% collaborative, 20% content

## 🌟 Real-World Connections

### Netflix
- Processes 200M recommendations daily
- 80% of content watched comes from recommendations
- Uses hybrid approach similar to this implementation

### Spotify
- Combines collaborative filtering with deep audio analysis
- Enables instant recommendations for new releases

### YouTube
- Two-stage architecture: Candidate Generation → Ranking
- Diversity injection prevents filter bubbles

## 📈 Performance Characteristics

**Training Time**:
- Collaborative model (SVD): ~2-3 seconds
- Content model: ~1 second
- Total training: <5 seconds

**Prediction Latency**:
- Single prediction: <1ms
- Batch (100 predictions): ~10ms
- Top-10 recommendations: ~15ms

## 🔧 Customization

### Adjusting Blend Weights

```python
hybrid = HybridRecommender(
    collab_model,
    content_model,
    min_alpha=0.3,  # More collaborative for new users
    max_alpha=0.9   # Even more collaborative for established
)
```

### Changing Latent Factors

```python
# More factors = more nuanced preferences but slower
collab_model = CollaborativeFilter(n_factors=100)

# Fewer factors = faster but less accurate
collab_model = CollaborativeFilter(n_factors=20)
```

## 📚 Next Steps

1. **Deep Learning**: Replace SVD with neural collaborative filtering
2. **Reinforcement Learning**: Optimize for long-term engagement
3. **Context-Aware**: Incorporate time, device, mood
4. **Distributed Systems**: Scale to millions of users with Spark
5. **Online Learning**: Update models in real-time from streaming data

## 🐛 Troubleshooting

**Dataset download fails**:
```bash
# Manual download
cd data
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

**Import errors**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Tests fail**:
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run with verbose output
pytest -vv tests/
```

## 📄 License

Educational project for 180-Day AI/ML Course.
MovieLens dataset © GroupLens Research Group.

---

**Built for Days 106-112 of the 180-Day AI/ML Course**

Next lesson: Day 113 - Gradient Boosting Machines

