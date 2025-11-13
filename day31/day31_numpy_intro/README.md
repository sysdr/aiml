# Day 31: Introduction to NumPy

## Overview
Learn NumPy fundamentals for AI/ML through building a production-style data preprocessing pipeline. This lesson demonstrates why NumPy is the foundation of every major AI framework.

## What You'll Learn
- NumPy arrays and why they're 10-100x faster than Python lists
- Vectorized operations for processing millions of numbers instantly
- Broadcasting for efficient operations on different-shaped arrays
- Real-world data preprocessing patterns used in production AI systems

## Quick Start

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### 2. Run Main Lesson
```bash
python lesson_code.py
```

Expected output: Performance comparison showing NumPy's speed, feature extraction from 10,000 images, and batch processing demonstration.

### 3. Run Tests
```bash
pytest test_lesson.py -v
```

All tests should pass, confirming your understanding of NumPy fundamentals.

## Key Concepts Covered

### 1. Arrays (The Foundation)
- Creating arrays from data
- Array properties (shape, dtype, size)
- Why arrays are faster than lists

### 2. Vectorization (The Speed Secret)
- Processing millions of elements simultaneously
- Eliminating loops for performance
- Real-world performance comparisons

### 3. Broadcasting (The Power Multiplier)
- Operating on different-shaped arrays
- Avoiding explicit loops and copies
- How neural networks use broadcasting

### 4. Memory Efficiency (The Scale Enabler)
- Contiguous memory storage
- 10x less RAM than Python lists
- Saving/loading preprocessed data

## Real-World Applications

This lesson's patterns appear in:
- **Tesla Autopilot**: Processing camera frames for computer vision
- **OpenAI GPT**: Converting text to token embeddings
- **Netflix**: Computing user-item similarities for recommendations
- **Spotify**: Processing audio features for music analysis

## Files Generated
- `setup.sh` - Environment setup script
- `lesson_code.py` - Complete preprocessing pipeline implementation
- `test_lesson.py` - Test suite validating your NumPy knowledge
- `requirements.txt` - Python dependencies
- `features.npy` - Sample preprocessed data (generated when you run the code)

## Performance Benchmarks
On a typical laptop, you should see:
- 10,000 image normalization: ~0.05 seconds
- Feature extraction: ~0.5 seconds
- NumPy 100x+ faster than Python lists
- Batch processing 1000 images: ~0.3 seconds

## Next Steps
Tomorrow (Day 32): NumPy Array Manipulation and Vectorization
- Advanced indexing techniques
- Fancy indexing and boolean masking
- Multidimensional array operations
- Setting up for machine learning algorithms

## Troubleshooting

**Import errors**: Make sure virtual environment is activated
```bash
source venv/bin/activate
```

**Slow performance**: Ensure NumPy is using optimized BLAS libraries
```python
import numpy as np
print(np.__config__.show())
```

**Memory issues**: Reduce num_images in generate_synthetic_images() call

## Additional Resources
- NumPy Official Documentation: https://numpy.org/doc/
- NumPy for Beginners: https://numpy.org/doc/stable/user/absolute_beginners.html
- Real Python NumPy Tutorial: https://realpython.com/numpy-tutorial/

---
**Time to complete**: 2-3 hours
**Difficulty**: Beginner
**Prerequisites**: Basic Python knowledge (Day 1-29 completed)
