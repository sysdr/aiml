# Day 116: Hyperparameter Tuning Theory
**180-Day AI/ML Course — Module 3, Week 17-18**

## What You'll Learn
- Difference between model **parameters** (learned) and **hyperparameters** (set by you)
- Three search strategies: **Grid Search**, **Random Search**, **Bayesian Optimization**
- Cross-validation integrity — why the test set is touched exactly once
- A reusable `HyperparameterTuner` class for production ML pipelines

---

## Quick Start (5 minutes)

### Option A: Local Python

```bash
# 1. Setup
bash setup.sh
source .venv/bin/activate

# 2. Run full demo
python lesson_code.py

# 3. Run tests
pytest test_lesson.py -v

# 4. Interactive
jupyter notebook
```

Expected output:
```
Day 116: Hyperparameter Tuning Theory
Dataset split: Train 3400 | Val 850 | Test 750
── Baseline (Default Hyperparameters) ──
  Validation AUC : 0.8312
── Grid Search ──
  Best CV AUC    : 0.8491
── Random Search ──
  Best CV AUC    : 0.8563
── Bayesian Optimization (Optuna/TPE) ──
  Best CV AUC    : 0.8617
── RESULTS SUMMARY ──
  Final TEST SET AUC : 0.8594  ← reported once
  AUC lift vs baseline : +0.0282
```

### Option B: Docker

```bash
docker run --rm -v $(pwd):/workspace -w /workspace \
  python:3.11-slim bash -c "pip install -r requirements.txt -q && python lesson_code.py"
```

To run tests in Docker:
```bash
docker run --rm -v $(pwd):/workspace -w /workspace \
  python:3.11-slim bash -c "pip install -r requirements.txt pytest -q && pytest test_lesson.py -v"
```

---

## File Structure

```
day116_hyperparameter_tuning/
├── lesson_code.py          # Main implementation — 5 experiments
├── test_lesson.py          # 15 pytest tests covering all concepts
├── requirements.txt        # Python dependencies
├── setup.sh                # Automated environment setup
└── README.md               # This file
```

---

## Key Concepts Quick Reference

| Concept | What it is | Example |
|---------|-----------|---------|
| Parameter | Learned during training | GBM tree split values |
| Hyperparameter | Set before training | `learning_rate`, `max_depth` |
| Grid Search | All combos exhaustively | 24 fixed configs |
| Random Search | Sampled randomly | 30 random configs |
| Bayesian Optim | Surrogate-guided | 30 smart configs |
| CV Integrity | Test set = final only | AUC reported once |

---

## Running Individual Tests

```bash
# Test only concept understanding
pytest test_lesson.py::TestHyperparameterConcepts -v

# Test only Bayesian Optimization
pytest test_lesson.py::TestBayesianOptimization -v

# Test only the HyperparameterTuner class
pytest test_lesson.py::TestHyperparameterTuner -v

# Full integration test
pytest test_lesson.py::TestEndToEndPipeline -v
```

---

## Troubleshooting

**`optuna` import error**: `pip install optuna>=3.6.0`

**Tests slow**: Add `-x` flag to stop at first failure: `pytest test_lesson.py -v -x`

**Memory error on 5000 samples**: Edit `load_dataset(n_samples=2000)` in `lesson_code.py`

---

## What's Next

**Days 117–126 Project**: Take your fraud detection or recommendation model from earlier in the course and run a full Optuna tuning campaign. You will set up an Optuna dashboard, define a production-grade search space, run 50+ trials, and document the AUC lift.

```bash
# Preview: launch Optuna dashboard
optuna-dashboard sqlite:///optuna_study.db
```

---

*Previous: Day 115 — Bias-Variance Tradeoff*
*Next: Days 117–126 — Project: Improve a Previous Model with Hyperparameter Tuning*
