# Day 47: Housing Price Prediction Project

A production-ready housing price prediction system using multiple linear regression.

## Overview

This project implements a complete ML pipeline for predicting real estate prices, demonstrating:
- Realistic data generation and EDA
- Feature engineering and missing data handling
- Model training with proper train/val/test splits
- Comprehensive evaluation metrics
- Prediction API with input validation
- Model persistence and deployment patterns

## Quick Start

### 1. Setup Environment

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Generate and Explore Data

```bash
# Generate housing dataset (1000 properties)
python lesson_code.py --generate-data

# Perform exploratory data analysis
python lesson_code.py --analyze
```

### 3. Train and Evaluate Model

```bash
# Train model with automatic evaluation
python lesson_code.py --train --evaluate
```

### 4. Make Predictions

```bash
# Predict price for a specific property
python lesson_code.py --predict     --sqft 2000     --bedrooms 3     --bathrooms 2     --lot-size 0.25     --age 10     --garage 2
```

### 5. Run Tests

```bash
# Run comprehensive test suite
pytest test_lesson.py -v

# With coverage report
pytest test_lesson.py -v --cov=lesson_code --cov-report=html
```

## Project Structure

```
.
├── lesson_code.py          # Main implementation
├── test_lesson.py          # Test suite
├── requirements.txt        # Python dependencies
├── setup.sh               # Environment setup script
├── README.md              # This file
├── housing_data.csv       # Generated dataset (after --generate-data)
├── housing_model.pkl      # Trained model (after --train)
└── *.png                  # Generated visualizations
```

## Requirements

- Python 3.11+
- See `requirements.txt` for dependencies

## License

Educational use for 180-Day AI/ML Course
