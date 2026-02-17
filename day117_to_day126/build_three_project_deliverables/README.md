# Day 117–126: Improve a Model with Hyperparameter Tuning

## What You Will Build

A production-grade hyperparameter tuning pipeline that takes an XGBoost fraud
detection model from a 0.74 F1 baseline to 0.83+ using Bayesian optimization
with Optuna's TPE sampler and MedianPruner.

## Quick Start (local)

```bash
# 1. Generate all files (you've already done this)
# chmod +x generate_lesson_files.sh && ./generate_lesson_files.sh

# 2. Set up environment
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate

# 3. Run the full tuning pipeline (~5-8 min on CPU)
python lesson_code.py

# 4. Run the test suite
pytest test_lesson.py -v
```

### Expected output (lesson_code.py)

```
[DATA] 10,000 samples  |  fraud rate: 3.0%
[SPLIT]  Train: 8,000  |  Test: 2,000
[BASELINE] Training with default hyperparameters...
  F1: 0.7412  Precision: 0.7834  Recall: 0.7031  AUC-ROC: 0.9108
[OPTUNA] Starting search: 50 trials × 5-fold CV = 250 model fits
  Best CV F1: 0.8270
[TUNED] Retraining on full training set with best params...
  F1: 0.8301  Precision: 0.8512  Recall: 0.8102  AUC-ROC: 0.9476
[REPORT] Saved → outputs/tuning_report.md
[PLOT]   Saved → outputs/optimization_history.png
TUNING COMPLETE
  Baseline F1 : 0.7412
  Tuned F1    : 0.8301
  Improvement : +0.0889  (+12.0%)
```

## Docker (no local Python required)

```bash
docker run --rm -v "$(pwd)":/app -w /app python:3.11-slim bash -c \
  "pip install -r requirements.txt -q && python lesson_code.py"
```

## Output Files

| File                                    | Description                              |
|-----------------------------------------|------------------------------------------|
| `outputs/fraud_detector_baseline.pkl`   | Serialised default-params model          |
| `outputs/fraud_detector_tuned_v2.pkl`   | Serialised Optuna-optimised model        |
| `outputs/tuning_report.md`              | Markdown comparison table + best params  |
| `outputs/optimization_history.png`      | Trial progress + parameter importance    |

## Key Concepts Demonstrated

- **Bayesian Optimization (TPE)** — learns from every trial to guide next sample
- **MedianPruner** — kills weak trials early, saving ~30% compute
- **Stratified CV** — preserves class imbalance ratio across all folds
- **scale_pos_weight** — compensates for 3% fraud rate without oversampling
- **Experiment tracking** — every trial logged to Optuna Study object
- **Model serialisation** — joblib round-trip with prediction identity test

## Tuning Parameters Reference

| Parameter          | Range         | Why Tune It                              |
|--------------------|---------------|------------------------------------------|
| `learning_rate`    | 1e-4 – 0.3   | Single biggest lever (log scale)         |
| `n_estimators`     | 100 – 800    | Pairs with learning rate                 |
| `max_depth`        | 3 – 10       | Controls overfitting vs underfitting     |
| `subsample`        | 0.5 – 1.0    | Row subsampling → variance reduction     |
| `colsample_bytree` | 0.5 – 1.0    | Feature subsampling → decorrelation      |

## Increase Search Depth

Edit `lesson_code.py` line 28:
```python
N_TRIALS = 200   # Production runs typically use 500–2000
```

## Next Lesson

Day 127 — The Perceptron and its Limitations (Module 4: Deep Learning)
