# Day 32: NumPy Array Manipulation and Vectorization

## Overview

Learn production-grade array manipulation techniques used at Tesla, Google, and OpenAI. This lesson covers reshaping, vectorization, broadcasting, and advanced indexing—the foundations that make real-time AI systems possible.

## Quick Start

```bash
# Setup environment
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate

# Run lesson
python lesson_code.py

# Launch interactive dashboard
streamlit run dashboard.py

# Run tests
pytest test_lesson.py -v
```

## What You'll Learn

1. **Image Batch Processing** - The exact preprocessing pipeline used for ResNet/VGG
2. **Weight Initialization** - Xavier and Kaiming methods that prevent training failures
3. **Vectorization** - Achieve 50-200x speedups over Python loops
4. **Advanced Indexing** - Top-k selection, filtering, gather/scatter operations

## Key Files

- `lesson_code.py` - Complete implementation with production patterns
- `test_lesson.py` - Comprehensive tests verifying correctness
- `requirements.txt` - Dependencies (NumPy, pytest)

## Key Concepts

### Vectorization
```python
# Slow (loop)
for i in range(len(data)):
    result[i] = data[i] * 2

# Fast (vectorized) - 100x faster
result = data * 2
```

### Broadcasting
```python
# ImageNet normalization across batch
images = (images - mean) / std  # mean shape (3,) broadcasts to (batch, h, w, 3)
```

### Reshaping
```python
# Image to neural network input
image.reshape(-1)  # Flatten
# ViT patch extraction
image.reshape(14, 16, 14, 16, 3)
```

## Production Applications

- **Tesla Autopilot**: Processes 2,500 images/second using these techniques
- **GPT models**: Top-k token sampling uses advanced indexing
- **Vision Transformers**: Patch extraction via reshape operations

## Expected Output

```
1. IMAGE BATCH PROCESSING
   Input: (32, 224, 224, 3) uint8 [0-255]
   Output: (32, 3, 224, 224) float32 [-2.5 to 2.5]

2. WEIGHT INITIALIZATION
   Layer 1: (784, 256), mean≈0, std≈0.05

3. VECTORIZATION PERFORMANCE
   Normalization speedup: ~80x
   Matrix multiply speedup: ~150x

4. ADVANCED INDEXING
   Top-k, filtering, scatter-add demonstrations
```

## Next Steps

Tomorrow (Day 33): Introduction to Pandas - building on NumPy for real-world messy data handling.
