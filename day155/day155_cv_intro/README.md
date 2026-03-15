# Day 155: Introduction to Computer Vision

## What You Will Build

A production-grade image preprocessing pipeline that converts raw images into
normalized tensors ready for convolutional neural networks. This is the exact
foundation used in systems like Google Lens, Tesla Autopilot, and Waymo.

---

## Quick Start (without Docker)

```bash
# 1. Setup environment
bash setup.sh
source venv/bin/activate

# 2. Run the demo
python lesson_code.py

# 3. Run all tests
pytest test_lesson.py -v
```

Expected test output: 25 passed

---

## Quick Start (with Docker)

```bash
# Build the image
docker build -t day155-cv .

# Run demo
docker run --rm day155-cv python lesson_code.py

# Run tests
docker run --rm day155-cv pytest test_lesson.py -v
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python - << 'EOF'
import urllib.request
urllib.request.urlretrieve(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    "sample.png"
)
EOF
CMD ["python", "lesson_code.py"]
```

---

## Key Concepts

| Concept | Why It Matters |
|---------|---------------|
| BGR vs RGB | OpenCV loads BGR; PyTorch expects RGB. Wrong order = silent accuracy loss. |
| Tensor shape [C, H, W] | PyTorch uses channel-first convention. NumPy/PIL use [H, W, C]. |
| Normalize to [0,1] then ImageNet stats | Pretrained models expect this exact distribution. |
| Batch dimension | Models process [N, C, H, W]. Always stack into batches. |

---

## Files

| File | Purpose |
|------|---------|
| `lesson_code.py` | Core preprocessing pipeline with 6 functions |
| `test_lesson.py` | 25 pytest tests across 5 test classes |
| `requirements.txt` | Pinned dependencies |
| `setup.sh` | Environment setup + sample image download |
