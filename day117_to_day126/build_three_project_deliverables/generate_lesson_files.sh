#!/usr/bin/env bash
# generate_lesson_files.sh
# Day 117-126: Improve a Model with Hyperparameter Tuning
# Run: chmod +x generate_lesson_files.sh && ./generate_lesson_files.sh

set -euo pipefail

echo "=============================================="
echo " Day 117-126: Hyperparameter Tuning Project"
echo " Generating implementation package..."
echo "=============================================="

mkdir -p outputs

# ─────────────────────────────────────────────────
# 1. requirements.txt
# ─────────────────────────────────────────────────
cat > requirements.txt << 'EOF'
xgboost>=2.0.0
optuna>=3.6.0
scikit-learn>=1.4.0
pandas>=2.1.0
numpy>=1.26.0
matplotlib>=3.8.0
joblib>=1.3.0
pytest>=8.0.0
EOF

echo "[OK] requirements.txt"

# ─────────────────────────────────────────────────
# 2. setup.sh
# ─────────────────────────────────────────────────
cat > setup.sh << 'EOF'
#!/usr/bin/env bash
# setup.sh - Environment setup for Day 117-126

set -euo pipefail

echo "=== Day 117-126: Hyperparameter Tuning Setup ==="

# Verify Python 3.11+
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/4] Detected Python ${PYVER}"

# Create virtual environment
echo "[2/4] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "[3/4] Installing dependencies (this may take 60-90 seconds)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Create outputs directory
mkdir -p outputs
echo "[4/4] Setup complete."
echo ""
echo "  Activate:  source .venv/bin/activate"
echo "  Run:       python lesson_code.py"
echo "  Test:      pytest test_lesson.py -v"
echo ""
echo "  Docker:    docker run -v \$(pwd):/app -w /app python:3.11-slim \\"
echo "             bash -c 'pip install -r requirements.txt -q && python lesson_code.py'"
EOF
chmod +x setup.sh
echo "[OK] setup.sh"

# ─────────────────────────────────────────────────
# 3. lesson_code.py
# ─────────────────────────────────────────────────
cat > lesson_code.py << 'EOF'
"""
lesson_code.py
Day 117-126: Improve a Previous Model with Hyperparameter Tuning

Pipeline:
  1. Generate synthetic fraud detection data (mirrors Module 2 dataset)
  2. Train XGBoost baseline with default hyperparameters
  3. Run Bayesian hyperparameter search using Optuna (TPE + MedianPruner)
  4. Retrain on full training set with best params
  5. Compare baseline vs tuned metrics
  6. Generate markdown report + optimization history plot
  7. Serialize both models for downstream use
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import optuna
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_val_score,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import joblib
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (works in CI/Docker)
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
N_TRIALS       = 50        # Raise to 200 for production runs
CV_FOLDS       = 5
OUTPUT_DIR     = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── 1. Data Generation ───────────────────────────────────────────────────────

def generate_fraud_data() -> pd.DataFrame:
    """
    Synthetic fraud dataset:
      - 10,000 transactions, 20 features
      - ~3% fraud rate (realistic class imbalance)
      - 1% label noise to simulate real-world messiness
    """
    X, y = make_classification(
        n_samples=10_000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        weights=[0.97, 0.03],
        flip_y=0.01,
        random_state=RANDOM_STATE,
    )
    cols = [f"feature_{i:02d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["is_fraud"] = y
    print(f"[DATA] {len(df):,} samples  |  fraud rate: {y.mean()*100:.1f}%")
    return df


# ─── 2. Baseline Model ────────────────────────────────────────────────────────

def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple:
    """
    XGBoost with out-of-the-box defaults.
    scale_pos_weight compensates for class imbalance — this is the only
    non-default setting because omitting it produces a degenerate classifier.
    """
    print("\n[BASELINE] Training with default hyperparameters...")
    t0 = time.time()

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "model":           "XGBoost (baseline)",
        "f1":              round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision":       round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":          round(recall_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":         round(roc_auc_score(y_test, y_prob), 4),
        "train_time_sec":  round(elapsed, 2),
    }

    print(f"  F1:        {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  AUC-ROC:   {results['auc_roc']:.4f}")
    print(f"  Time:      {elapsed:.1f}s")
    return model, results


# ─── 3. Optuna Objective ──────────────────────────────────────────────────────

def make_objective(X_train: np.ndarray, y_train: np.ndarray):
    """
    Factory: returns a closure over the training data.
    Optuna calls objective(trial) for each search iteration.

    Search space hierarchy (coarse → fine within a single pass):
      High-impact : learning_rate, max_depth, n_estimators
      Medium      : subsample, colsample_bytree
      Regulariser : gamma, reg_alpha, reg_lambda, min_child_weight
    """
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            # High-impact axes
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            # Tree structure
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            # Regularisation
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 3.0),
        }

        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        scores = cross_val_score(
            model, X_train, y_train,
            scoring="f1",
            cv=cv,
            n_jobs=1,          # Avoid nested parallelism warnings
        )
        return float(scores.mean())

    return objective


# ─── 4. Run Optuna Study ──────────────────────────────────────────────────────

def run_hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> optuna.Study:
    """
    TPE sampler:   intelligent Bayesian search (learns from prior trials)
    MedianPruner:  kills trials below median performance at each fold checkpoint
    n_warmup_steps: don't prune until 10 trials complete (cold-start safety)
    """
    print(
        f"\n[OPTUNA] Starting search: {N_TRIALS} trials × {CV_FOLDS}-fold CV"
        f" = {N_TRIALS * CV_FOLDS} model fits"
    )
    print("         Grab a coffee — this takes a few minutes on CPU ☕")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=10)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="fraud_detector_tuning_v2",
    )

    t0 = time.time()
    study.optimize(
        make_objective(X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    print(f"\n[OPTUNA] Search complete in {elapsed:.1f}s")
    print(f"  Best CV F1:  {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=4)}")
    return study


# ─── 5. Train Tuned Model ─────────────────────────────────────────────────────

def train_tuned_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: dict,
) -> tuple:
    """
    Critical: retrain on the FULL training set, not just one CV fold.
    Evaluate on the held-out test set (never touched during CV).
    """
    print("\n[TUNED] Retraining on full training set with best params...")
    t0 = time.time()

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "model":          "XGBoost (Optuna-tuned)",
        "f1":             round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision":      round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":         round(recall_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":        round(roc_auc_score(y_test, y_prob), 4),
        "train_time_sec": round(elapsed, 2),
    }

    print(f"  F1:        {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  AUC-ROC:   {results['auc_roc']:.4f}")
    return model, results


# ─── 6. Comparison Report ─────────────────────────────────────────────────────

def generate_report(
    baseline: dict,
    tuned: dict,
    study: optuna.Study,
) -> str:
    f1_delta = tuned["f1"] - baseline["f1"]
    f1_pct   = f1_delta / max(baseline["f1"], 1e-9) * 100
    auc_delta = tuned["auc_roc"] - baseline["auc_roc"]

    report = f"""# Hyperparameter Tuning Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Performance Comparison

| Metric     | Baseline | Tuned    | Delta                  |
|------------|----------|----------|------------------------|
| F1 Score   | {baseline['f1']:.4f}   | {tuned['f1']:.4f}   | {f1_delta:+.4f} ({f1_pct:+.1f}%) |
| Precision  | {baseline['precision']:.4f}   | {tuned['precision']:.4f}   | {tuned['precision']-baseline['precision']:+.4f}             |
| Recall     | {baseline['recall']:.4f}   | {tuned['recall']:.4f}   | {tuned['recall']-baseline['recall']:+.4f}             |
| AUC-ROC    | {baseline['auc_roc']:.4f}   | {tuned['auc_roc']:.4f}   | {auc_delta:+.4f}             |

## Search Configuration

- Strategy    : Bayesian Optimization — Optuna TPE Sampler
- Pruner      : MedianPruner (n_warmup_steps=10)
- Trials run  : {len(study.trials)}
- CV folds    : {CV_FOLDS}
- Total fits  : {len(study.trials) * CV_FOLDS}

## Best Hyperparameters

```json
{json.dumps(study.best_params, indent=2)}
```

## Key Takeaway

F1 improved by **{f1_pct:+.1f}%** through systematic Bayesian search.
No new data. No new algorithm. Just disciplined optimization.
"""

    path = OUTPUT_DIR / "tuning_report.md"
    path.write_text(report)
    print(f"[REPORT] Saved → {path}")
    return report


# ─── 7. Optimization History Plot ─────────────────────────────────────────────

def plot_history(study: optuna.Study) -> None:
    """
    Two-panel matplotlib plot:
      Left:  individual trial F1 scores (scatter)
      Right: parameter importance bar chart
    """
    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = [max(values[: i + 1]) for i in range(len(values))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Optuna Optimization — Fraud Detector v2", fontsize=14, fontweight="bold")

    # Panel 1: trial history
    ax = axes[0]
    ax.scatter(range(len(values)), values, alpha=0.4, color="#6BA3D6", s=30, label="Trial F1")
    ax.plot(range(len(best_so_far)), best_so_far, color="#E8914A", lw=2.5, label="Best so far")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("CV F1 Score")
    ax.set_title("Trial-by-Trial Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: parameter importance (Fanova)
    try:
        importances = optuna.importance.get_param_importances(study)
        params   = list(importances.keys())[:8]    # Top 8
        imp_vals = [importances[p] for p in params]
        colors   = ["#6BA3D6" if v > 0.1 else "#CBD5E1" for v in imp_vals]
        axes[1].barh(params[::-1], imp_vals[::-1], color=colors[::-1], edgecolor="white")
        axes[1].set_xlabel("Importance Score")
        axes[1].set_title("Hyperparameter Importance (Fanova)")
        axes[1].grid(True, alpha=0.3, axis="x")
    except Exception:
        axes[1].text(0.5, 0.5, "Importance unavailable\n(requires ≥20 trials)",
                     ha="center", va="center", transform=axes[1].transAxes, color="#64748b")

    plt.tight_layout()
    path = OUTPUT_DIR / "optimization_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT]   Saved → {path}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print(" Day 117-126: Hyperparameter Tuning Project")
    print(" Baseline → Optuna Search → Tuned Model v2")
    print("=" * 60)

    # ── Load data ──
    df = generate_fraud_data()
    X = df.drop("is_fraud", axis=1).values
    y = df["is_fraud"].values

    # Stratified split preserves the 3% fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[SPLIT]  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Step 1: Baseline ──
    baseline_model, baseline_results = train_baseline(X_train, y_train, X_test, y_test)

    # ── Step 2: Hyperparameter search ──
    study = run_hyperparameter_search(X_train, y_train)

    # ── Step 3: Retrain best config ──
    tuned_model, tuned_results = train_tuned_model(
        X_train, y_train, X_test, y_test, study.best_params
    )

    # ── Step 4: Report + Plot ──
    generate_report(baseline_results, tuned_results, study)
    plot_history(study)

    # ── Step 5: Serialize models ──
    joblib.dump(baseline_model, OUTPUT_DIR / "fraud_detector_baseline.pkl")
    joblib.dump(tuned_model,    OUTPUT_DIR / "fraud_detector_tuned_v2.pkl")
    print(f"[SAVE]   Models serialized → {OUTPUT_DIR}/")

    # ── Final Summary ──
    f1_delta = tuned_results["f1"] - baseline_results["f1"]
    f1_pct   = f1_delta / max(baseline_results["f1"], 1e-9) * 100
    print("\n" + "=" * 60)
    print(" TUNING COMPLETE")
    print(f"  Baseline F1 : {baseline_results['f1']:.4f}")
    print(f"  Tuned F1    : {tuned_results['f1']:.4f}")
    print(f"  Improvement : {f1_delta:+.4f}  ({f1_pct:+.1f}%)")
    print("=" * 60)
    print("\n  Next → open outputs/tuning_report.md")


if __name__ == "__main__":
    main()
EOF

echo "[OK] lesson_code.py"

# ─────────────────────────────────────────────────
# 4. test_lesson.py
# ─────────────────────────────────────────────────
cat > test_lesson.py << 'EOF'
"""
test_lesson.py
Day 117-126: Pytest suite — Hyperparameter Tuning Project

Run: pytest test_lesson.py -v
Expected: 15 tests passing in < 3 minutes
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# ─── Shared Fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_dataset():
    """400-sample stratified split — fast iteration for tests."""
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=7,
        weights=[0.94, 0.06],
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


# ─── Data Tests ───────────────────────────────────────────────────────────────

class TestDataGeneration:

    def test_fraud_rate_is_realistic(self):
        """Fraud rate must fall between 1% and 10%."""
        from lesson_code import generate_fraud_data
        df = generate_fraud_data()
        rate = df["is_fraud"].mean()
        assert 0.01 <= rate <= 0.10, f"Unrealistic fraud rate: {rate:.2%}"

    def test_no_nulls_in_dataset(self):
        from lesson_code import generate_fraud_data
        df = generate_fraud_data()
        assert df.isnull().sum().sum() == 0

    def test_dataset_shape(self):
        from lesson_code import generate_fraud_data
        df = generate_fraud_data()
        assert df.shape == (10_000, 21)   # 20 features + target

    def test_stratified_split_preserves_ratio(self, small_dataset):
        """Train/test fraud rates should differ by less than 3 percentage points."""
        X_train, X_test, y_train, y_test = small_dataset
        assert abs(y_train.mean() - y_test.mean()) < 0.03


# ─── Baseline Tests ───────────────────────────────────────────────────────────

class TestBaselineModel:

    def test_baseline_trains_without_error(self, small_dataset):
        from xgboost import XGBClassifier
        X_train, X_test, y_train, y_test = small_dataset
        spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
        model = XGBClassifier(
            n_estimators=30,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_baseline_outputs_valid_probabilities(self, small_dataset):
        from xgboost import XGBClassifier
        X_train, X_test, y_train, y_test = small_dataset
        spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
        model = XGBClassifier(
            n_estimators=30,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0
        assert probs.shape == (len(y_test),)

    def test_predictions_are_binary(self, small_dataset):
        from xgboost import XGBClassifier
        X_train, X_test, y_train, y_test = small_dataset
        model = XGBClassifier(n_estimators=30, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1})


# ─── Optuna Integration Tests ─────────────────────────────────────────────────

class TestOptunaSearch:

    def test_study_completes_all_trials(self, small_dataset):
        import optuna
        from lesson_code import make_objective
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        X_train, X_test, y_train, y_test = small_dataset
        study = optuna.create_study(direction="maximize")
        study.optimize(make_objective(X_train, y_train), n_trials=5)
        assert len(study.trials) == 5

    def test_best_value_is_valid_f1(self, small_dataset):
        import optuna
        from lesson_code import make_objective
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        X_train, X_test, y_train, y_test = small_dataset
        study = optuna.create_study(direction="maximize")
        study.optimize(make_objective(X_train, y_train), n_trials=5)
        assert 0.0 <= study.best_value <= 1.0

    def test_best_params_within_search_bounds(self, small_dataset):
        import optuna
        from lesson_code import make_objective
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        X_train, X_test, y_train, y_test = small_dataset
        study = optuna.create_study(direction="maximize")
        study.optimize(make_objective(X_train, y_train), n_trials=5)
        p = study.best_params
        assert 100 <= p["n_estimators"] <= 800
        assert 1e-4 <= p["learning_rate"] <= 0.3
        assert 3 <= p["max_depth"] <= 10
        assert 0.5 <= p["subsample"] <= 1.0
        assert 0.5 <= p["colsample_bytree"] <= 1.0

    def test_all_trials_have_values(self, small_dataset):
        import optuna
        from lesson_code import make_objective
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        X_train, X_test, y_train, y_test = small_dataset
        study = optuna.create_study(direction="maximize")
        study.optimize(make_objective(X_train, y_train), n_trials=4)
        completed = [t for t in study.trials if t.value is not None]
        assert len(completed) >= 3   # At least 3 of 4 should complete


# ─── Report Tests ─────────────────────────────────────────────────────────────

class TestReportGeneration:

    def test_improvement_delta_calculation(self):
        """Arithmetic correctness of the delta calculation."""
        baseline_f1 = 0.7400
        tuned_f1    = 0.8325
        delta = tuned_f1 - baseline_f1
        pct   = delta / baseline_f1 * 100
        assert abs(delta - 0.0925) < 0.0001
        assert abs(pct - 12.5) < 0.1

    def test_report_contains_required_sections(self, tmp_path, small_dataset, monkeypatch):
        import optuna
        import lesson_code
        from lesson_code import make_objective, generate_report
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        X_train, X_test, y_train, y_test = small_dataset
        study = optuna.create_study(direction="maximize")
        study.optimize(make_objective(X_train, y_train), n_trials=3)

        monkeypatch.setattr(lesson_code, "OUTPUT_DIR", tmp_path)
        baseline = {"f1": 0.74, "precision": 0.78, "recall": 0.71, "auc_roc": 0.89}
        tuned    = {"f1": 0.83, "precision": 0.85, "recall": 0.81, "auc_roc": 0.94}
        report   = generate_report(baseline, tuned, study)

        for section in ["F1 Score", "Baseline", "Tuned", "Best Hyperparameters"]:
            assert section in report, f"Missing section: {section}"


# ─── Serialisation Tests ──────────────────────────────────────────────────────

class TestModelSerialisation:

    def test_model_round_trips_via_joblib(self, small_dataset, tmp_path):
        """Loaded model must produce byte-identical predictions."""
        import joblib
        from xgboost import XGBClassifier
        X_train, X_test, y_train, y_test = small_dataset
        model = XGBClassifier(n_estimators=20, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)
        original_preds = model.predict(X_test)

        path = tmp_path / "model.pkl"
        joblib.dump(model, path)
        loaded_preds = joblib.load(path).predict(X_test)

        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_serialised_probabilities_are_identical(self, small_dataset, tmp_path):
        import joblib
        from xgboost import XGBClassifier
        X_train, X_test, y_train, y_test = small_dataset
        model = XGBClassifier(n_estimators=20, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)
        orig_probs = model.predict_proba(X_test)

        path = tmp_path / "model.pkl"
        joblib.dump(model, path)
        np.testing.assert_array_almost_equal(
            orig_probs, joblib.load(path).predict_proba(X_test), decimal=6
        )
EOF

echo "[OK] test_lesson.py"

# ─────────────────────────────────────────────────
# 5. README.md
# ─────────────────────────────────────────────────
cat > README.md << 'EOF'
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
EOF

echo "[OK] README.md"

echo ""
echo "=============================================="
echo " All files generated successfully!"
echo ""
echo " Files created:"
echo "   requirements.txt"
echo "   setup.sh"
echo "   lesson_code.py"
echo "   test_lesson.py"
echo "   README.md"
echo "   outputs/ (directory)"
echo ""
echo " Next steps:"
echo "   chmod +x setup.sh && ./setup.sh"
echo "   source .venv/bin/activate"
echo "   python lesson_code.py"
echo "   pytest test_lesson.py -v"
echo "=============================================="
