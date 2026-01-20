# Day 87: K-Means with Scikit-learn

Production-ready K-Means clustering implementation using scikit-learn for customer segmentation and unsupervised learning applications.

## Overview

This lesson demonstrates how to implement K-Means clustering using scikit-learn's industrial-strength implementation. You'll build a complete customer segmentation system following production patterns used at companies like Spotify, Amazon, and Netflix.

## What You'll Learn

- Production K-Means implementation with scikit-learn
- Feature scaling and preprocessing for clustering
- Model persistence and deployment patterns
- Cluster analysis and interpretation
- Testing strategies for unsupervised learning

## Prerequisites

- Python 3.11+
- Understanding of K-Means theory (Day 86)
- Basic NumPy and pandas knowledge

## Quick Start

### 1. Setup Environment

```bash
# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Run Main Implementation

```bash
python lesson_code.py
```

Expected output:
- Training logs with metrics
- Segment analysis table
- Visualization saved to `customer_segments.png`
- Saved model in `models/customer_segmentation_v1.pkl`

### 3. Run Tests

```bash
# Run all tests
pytest test_lesson.py -v

# Run specific test class
pytest test_lesson.py::TestCustomerSegmentation -v

# Run with coverage
pytest test_lesson.py --cov=lesson_code --cov-report=term-missing
```

## Project Structure

```
day_87/
├── setup.sh                 # Environment setup script
├── requirements.txt         # Python dependencies
├── lesson_code.py          # Main implementation
├── test_lesson.py          # Comprehensive test suite
├── README.md               # This file
├── models/                 # Saved models (created at runtime)
└── customer_segments.png   # Visualization (created at runtime)
```

## Key Components

### CustomerSegmentation Class

Production-ready segmentation system with:
- Automatic feature scaling
- Model training and prediction
- Cluster analysis
- Model persistence

```python
from lesson_code import CustomerSegmentation

# Train segmentation model
model = CustomerSegmentation(n_clusters=5)
labels = model.fit_predict(customer_data)

# Save for production
model.save('models/segments_v1.pkl')

# Load and predict
loaded = CustomerSegmentation.load('models/segments_v1.pkl')
new_segments = loaded.predict(new_customers)
```

### Data Generation

Synthetic customer data for demonstration:

```python
from lesson_code import generate_synthetic_customers

X, y, feature_names = generate_synthetic_customers(
    n_samples=1000,
    n_features=4,
    n_clusters=5
)
```

## Running Without Docker

All code runs natively in Python virtual environment:

```bash
# Setup
./setup.sh
source venv/bin/activate

# Run
python lesson_code.py

# Test
pytest test_lesson.py -v
```

## Expected Results

### Training Output
```
Step 1: Generating synthetic customer data...
✅ Generated 1000 customers with 4 features

Step 2: Training customer segmentation model...
✅ Model trained successfully
   Inertia: 3847.23
   Iterations: 12
```

### Segment Analysis
```
   Segment  Size Percentage  purchase_frequency  avg_order_value  ...
 Segment 0   198     19.8%                4.23             102.45  ...
 Segment 1   203     20.3%                2.15              67.89  ...
 Segment 2   195     19.5%                8.76             187.23  ...
```

### Test Results
```
test_lesson.py::TestCustomerSegmentation::test_model_initialization PASSED
test_lesson.py::TestCustomerSegmentation::test_fit_basic PASSED
test_lesson.py::TestCustomerSegmentation::test_predict_basic PASSED
...
========================= 25 passed in 2.43s =========================
```

## Production Insights

### Feature Scaling
K-Means uses Euclidean distance, making feature scaling critical:

```python
# Without scaling: income ($20k-$200k) dominates age (20-80)
# With scaling: both features contribute equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Model Persistence
Train once (offline), predict many times (online):

```python
# Nightly batch job
model.fit(historical_data)
model.save('models/segments_v1.pkl')

# Real-time API
loaded = CustomerSegmentation.load('models/segments_v1.pkl')
segment = loaded.predict(new_customer)
```

### Initialization Strategy
k-means++ reduces poor local optima by 70%:

```python
# Always use k-means++ in production
model = CustomerSegmentation(init='k-means++')
```

## Real-World Applications

- **Spotify**: Clustering songs by audio features for recommendations
- **Amazon**: Product clustering for "customers also bought"
- **Uber**: Geographic zone creation for driver-rider matching
- **Netflix**: User segmentation for personalized content
- **Pinterest**: Visual similarity clustering for "More Like This"

## Troubleshooting

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Visualization Not Generated
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# Force Agg backend if needed
export MPLBACKEND=Agg
```

### Test Failures
```bash
# Run single test for debugging
pytest test_lesson.py::test_full_pipeline -v -s

# Check Python version
python --version  # Should be 3.11+
```

## Next Steps

- **Day 88**: Learn how to choose optimal number of clusters using elbow method and silhouette analysis
- Experiment with different numbers of clusters
- Try clustering on your own datasets
- Explore advanced initialization strategies

## Resources

- [Scikit-learn K-Means Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [K-Means++ Paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
- [Production ML Patterns](https://ml-ops.org/)

## License

Part of 180-Day AI/ML Course - For educational purposes
