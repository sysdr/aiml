# Day 103: Recommender Systems Theory

## Overview
Implementation of the three core recommender system architectures: Collaborative Filtering, Content-Based Filtering, and Hybrid approaches. This lesson demonstrates the mathematical foundations and production patterns behind Netflix, Amazon, and Spotify's recommendation engines.

## Quick Start

### Setup (First Time)
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Run Main Demo
```bash
python lesson_code.py
```

Expected output shows:
- Collaborative filtering recommendations using user similarity
- Content-based recommendations using item features
- Hybrid recommendations combining both approaches
- Performance comparisons and production insights

### Run Tests
```bash
pytest test_lesson.py -v
```

Expected: 25 tests passed

## What You'll Learn

### 1. Collaborative Filtering
- User-item interaction matrix construction
- Similarity computation using cosine distance
- Weighted score aggregation from similar users
- Handles implicit and explicit feedback

### 2. Content-Based Filtering
- Item feature extraction and representation
- User preference profile construction
- Feature-based similarity matching
- Cold-start item recommendations

### 3. Hybrid Systems
- Multi-method score combination
- Normalized score aggregation
- Production ensemble patterns
- Trade-offs between approaches

## Architecture

```
User Request
    ↓
Hybrid Recommender
    ├── Collaborative Engine (60% weight)
    │   ├── Find similar users (cosine similarity)
    │   ├── Aggregate ratings (weighted average)
    │   └── Return top candidates
    │
    ├── Content Engine (40% weight)
    │   ├── Match user profile to item features
    │   ├── Compute feature similarity
    │   └── Return top candidates
    │
    └── Score Fusion
        ├── Normalize scores [0,1]
        ├── Weight and combine
        └── Re-rank final recommendations
```

## Production Patterns

**Netflix**: Hybrid approach with 200+ models
- Stage 1: CF for candidate generation (<20ms)
- Stage 2: Deep learning for ranking (<50ms)
- Stage 3: Business logic (diversity, freshness)

**Amazon**: Cascading recommenders
- Item-to-item CF for related products
- Session-based for real-time
- Content features for cold-start

**Spotify**: Audio features + collaborative
- Deep learning on audio spectrograms
- User listening history patterns
- Playlist co-occurrence signals

## Key Metrics

**Sparsity**: 99%+ in production (users interact with <1% of items)
**Latency**: 50-100ms end-to-end for top-K recommendations
**Throughput**: 100M+ requests/day for large platforms
**Accuracy**: Measured via A/B testing, not offline metrics

## Common Pitfalls

1. **Popularity Bias**: CF recommends popular items more
   - Solution: Penalize popular items in ranking

2. **Cold Start**: No recommendations for new users/items
   - Solution: Use content features, trending items

3. **Filter Bubble**: Only recommending similar content
   - Solution: Add exploration (epsilon-greedy), diversity constraints

4. **Scalability**: User-user similarity doesn't scale
   - Solution: Use item-item CF or matrix factorization

## Tomorrow's Lesson

Day 104 implements collaborative filtering from scratch:
- Matrix factorization with gradient descent
- Implicit feedback handling
- Efficient similarity computation
- Cold-start strategies

## Resources

- Research paper: "Amazon.com Recommendations: Item-to-Item Collaborative Filtering"
- Netflix Prize: Lessons learned from competition
- RecSys conference proceedings

## Troubleshooting

**Low recommendation quality**: Increase dataset size or adjust similarity thresholds
**Slow performance**: Use approximate nearest neighbors (FAISS) for production
**Memory errors**: Implement sparse matrix representations

---
Ready for Day 104: Collaborative Filtering implementation!
