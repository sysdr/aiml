# Day 114: XGBoost and LightGBM - Production Fraud Detection

Production-grade implementation of advanced gradient boosting frameworks for high-scale fraud detection systems.

## Overview

This lesson demonstrates:
- XGBoost and LightGBM implementation for fraud detection
- Performance benchmarking and optimization
- Feature importance analysis
- Production deployment patterns

## Quick Start

### 1. Setup Environment

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run Main Implementation

```bash
python lesson_code.py
```

Expected output:
- Training metrics for both models
- Inference performance benchmarks
- Feature importance analysis
- ROC curves and visualizations

### 3. Run Tests

```bash
pytest test_lesson.py -v
```

Expected: 25+ tests passing

## Project Structure

```
day_114_xgboost_lightgbm/
├── lesson_code.py          # Main implementation
├── test_lesson.py          # Comprehensive test suite
├── setup.sh                # Environment setup
├── requirements.txt        # Dependencies
├── creditcard.csv          # Fraud detection dataset (generated)
└── README.md               # This file
```

## Key Concepts

### XGBoost Optimizations
- Sparsity-aware split finding
- Weighted quantile sketching
- Cache-aware parallelization
- Histogram-based tree building

### LightGBM Innovations
- Gradient-based One-Side Sampling (GOSS)
- Leaf-wise tree growth
- Categorical feature support
- Memory-efficient training

### Production Patterns
- Class imbalance handling
- Early stopping for efficiency
- Feature importance extraction
- Inference performance benchmarking

## Expected Results

### Training Performance
- XGBoost: ~5-15 seconds
- LightGBM: ~3-10 seconds (typically 2-3x faster)

### Model Accuracy
- ROC-AUC: >0.95 on fraud detection
- F1 Score: >0.85 with proper threshold tuning

### Inference Speed
- XGBoost: ~100K predictions/second
- LightGBM: ~150K predictions/second

## Real-World Applications

### Fraud Detection (PayPal, Stripe)
- Real-time transaction scoring
- Adaptive fraud pattern learning
- Explainable predictions for compliance

### Dynamic Pricing (Uber, Airbnb)
- Sub-second price predictions
- Multi-feature optimization
- High-frequency model updates

### Ad Click Prediction (Microsoft Bing)
- Billions of predictions daily
- Feature importance for campaign optimization
- Memory-efficient training on massive datasets

## Common Issues

### Out of Memory
- Reduce `num_leaves` (LightGBM) or `max_depth` (XGBoost)
- Enable `subsample` and `colsample_bytree`
- Use `tree_method='hist'` for XGBoost

### Slow Training
- Reduce `n_estimators` initially
- Enable early stopping
- Use GPU training if available (`tree_method='gpu_hist'`)

### Poor Performance on Imbalanced Data
- Tune `scale_pos_weight` (XGBoost) or `is_unbalance` (LightGBM)
- Adjust prediction threshold
- Use stratified sampling

## Next Steps

Tomorrow (Day 115): **Bias-Variance Tradeoff**
- Diagnosing model errors
- Underfitting vs overfitting
- Regularization strategies
- Learning curves analysis

## Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [LightGBM Paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
