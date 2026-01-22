# Day 89: Customer Segmentation Project

Production-grade customer segmentation system using K-means clustering for behavioral analysis and segment profiling.

## Overview

This implementation demonstrates how companies like Netflix, Spotify, and Amazon segment millions of customers into distinct groups for personalized experiences. The system handles feature engineering, optimal cluster selection, segment profiling, and real-time predictions.

## Features

- **Intelligent Cluster Selection**: Automatically determines optimal number of segments using Elbow Method and Silhouette Analysis
- **Feature Engineering**: Creates composite behavioral metrics from raw customer data
- **Segment Profiling**: Generates detailed statistical profiles for each customer segment
- **Confidence Scoring**: Provides prediction confidence based on distance to cluster centers
- **Model Persistence**: Saves and loads trained models for production deployment
- **Comprehensive Testing**: 25+ tests covering edge cases and production scenarios

## Quick Start

### Setup

```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Run Demonstration

```bash
python lesson_code.py
```

Expected output:
- Segment profiles for customer groups
- Predictions on new customers with confidence scores
- Visualization of segments in feature space
- Saved model artifacts

### Run Tests

```bash
pytest test_lesson.py -v
```

All tests should pass, covering:
- Data generation and validation
- Feature engineering edge cases
- Model training and prediction
- Segment profiling
- Model persistence
- Error handling

## System Architecture

```
Customer Data → Feature Engineering → Scaling → Cluster Selection
                                                       ↓
User Interface ← Segment Assignment ← Model Training ← Optimal K
                      ↓
              Confidence Scoring
```

## Key Components

### CustomerSegmentationEngine

Main class handling the complete segmentation pipeline:

```python
engine = CustomerSegmentationEngine(min_clusters=2, max_clusters=10)
engine.fit(customer_data)
labels, confidence = engine.predict(new_customers)
```

### Feature Engineering

Automatically creates behavioral metrics:
- `value_score`: Monetary value per transaction
- `engagement_level`: Frequency × recency interaction
- `ltv_proxy`: Customer lifetime value estimate

### Segment Profiling

Generates comprehensive segment statistics:
- Mean, median, std dev for all features
- Segment sizes and distributions
- Distinguishing characteristics

## Real-World Applications

**Netflix**: Segments 230M+ subscribers into taste clusters for content recommendation
**Spotify**: Creates thousands of listening behavior segments for playlist curation
**Amazon**: Groups customers by purchase patterns for product recommendations
**Uber**: Segments riders (commuters vs. explorers) for targeted features

## Production Considerations

This implementation includes:
- Missing value handling
- Feature scaling consistency
- Outlier robustness
- Prediction confidence scoring
- Model versioning via persistence
- Comprehensive error handling

## File Structure

```
customer_segmentation_project/
├── setup.sh                        # Environment setup
├── requirements.txt                # Python dependencies
├── lesson_code.py                  # Main implementation
├── test_lesson.py                  # Test suite
├── README.md                       # This file
└── customer_segmentation_model.pkl # Trained model (generated)
```

## Next Steps

Tomorrow (Day 90): Hierarchical Clustering
- Learn alternative clustering approach
- Build dendrograms for multi-resolution segmentation
- Compare K-means vs. hierarchical methods

## Resources

- Scikit-learn K-means documentation
- Customer segmentation best practices
- Production ML system design patterns

## Requirements

- Python 3.11+
- See requirements.txt for package versions
