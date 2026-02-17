"""
Day 116: Hyperparameter Tuning Theory
======================================
Demonstrates Grid Search, Random Search, and Bayesian Optimization (Optuna)
on a binary classification task, comparing search strategies by efficiency.

Concepts covered:
  - Parameters vs Hyperparameters
  - Grid Search exhaustive scan
  - Random Search stochastic sampling
  - Bayesian Optimization with Optuna (surrogate model + acquisition function)
  - Cross-validation integrity (train/val/test split discipline)
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# 0. DATASET — Synthetic binary classification (credit default proxy)
# =============================================================================
def load_dataset(n_samples=5000, random_state=42):
    """
    Generate synthetic binary classification data mimicking credit scoring:
    20 features (15 informative, 5 redundant), moderate class imbalance.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.75, 0.25],   # 75% negative class — realistic imbalance
        random_state=random_state
    )
    return X, y


# =============================================================================
# 1. BASELINE — Default hyperparameters
# =============================================================================
def run_baseline(X_train, X_val, y_train, y_val):
    """Train GBM with sklearn defaults. Establish the performance floor."""
    print("\n── Baseline (Default Hyperparameters) ──────────────────────────")
    t0 = time.time()
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"  Default params : n_estimators=100, lr=0.1, max_depth=3")
    print(f"  Validation AUC : {val_auc:.4f}")
    print(f"  Training time  : {elapsed:.2f}s")
    return val_auc, model.get_params()


# =============================================================================
# 2. GRID SEARCH — Exhaustive over small, focused grid
# =============================================================================
def run_grid_search(X_train, y_train, cv):
    """
    Grid Search is the brute-force sweep. Every combination is evaluated.
    Use only when the search space is small and you have a compute budget.
    """
    print("\n── Grid Search ─────────────────────────────────────────────────")
    param_grid = {
        "n_estimators": [100, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
    }
    # Total combos: 2 × 3 × 2 × 2 = 24
    total_fits = 2 * 3 * 2 * 2 * cv.n_splits
    print(f"  Grid size      : 24 combos × {cv.n_splits} folds = {total_fits} fits")

    t0 = time.time()
    gs = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=False
    )
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Best CV AUC    : {gs.best_score_:.4f}")
    print(f"  Best params    : {gs.best_params_}")
    print(f"  Search time    : {elapsed:.1f}s")

    # Return results dataframe for visualization
    results_df = pd.DataFrame(gs.cv_results_)
    return gs.best_estimator_, gs.best_score_, results_df


# =============================================================================
# 3. RANDOM SEARCH — Stochastic sampling over continuous + discrete ranges
# =============================================================================
def run_random_search(X_train, y_train, cv, n_iter=30):
    """
    Random Search samples the space randomly. For high-dimensional spaces,
    it finds better configurations than Grid Search with the same budget.
    Key insight: any hyperparameter that barely matters gets fewer evaluations.
    """
    from scipy.stats import uniform, randint

    print("\n── Random Search ───────────────────────────────────────────────")
    param_dist = {
        "n_estimators": randint(50, 600),           # integer range
        "learning_rate": uniform(0.01, 0.29),        # continuous: 0.01 → 0.30
        "max_depth": randint(2, 9),
        "subsample": uniform(0.5, 0.5),              # continuous: 0.5 → 1.0
        "min_samples_leaf": randint(1, 25),
    }
    print(f"  Trials         : {n_iter} (vs 24 for Grid Search)")

    t0 = time.time()
    rs = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        return_train_score=False
    )
    rs.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Best CV AUC    : {rs.best_score_:.4f}")
    print(f"  Best params    : {rs.best_params_}")
    print(f"  Search time    : {elapsed:.1f}s")

    results_df = pd.DataFrame(rs.cv_results_)
    return rs.best_estimator_, rs.best_score_, results_df


# =============================================================================
# 4. BAYESIAN OPTIMIZATION — Optuna with TPE sampler
# =============================================================================
def run_bayesian_optuna(X_train, y_train, cv, n_trials=30):
    """
    Bayesian Optimization with Tree-structured Parzen Estimator (TPE).
    
    Each trial informs the surrogate model. The acquisition function decides
    the next configuration by balancing exploitation vs exploration.
    This is the backbone of Google Vizier, Meta Ax, AWS SageMaker tuning.
    """
    print("\n── Bayesian Optimization (Optuna/TPE) ──────────────────────────")
    trial_records = []

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 600),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "max_depth":       trial.suggest_int("max_depth", 2, 8),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 25),
        }
        model = GradientBoostingClassifier(random_state=42, **params)
        scores = cross_val_score(model, X_train, y_train,
                                  cv=cv, scoring="roc_auc", n_jobs=-1)
        auc = scores.mean()
        trial_records.append({
            "trial": trial.number,
            "auc": auc,
            "learning_rate": params["learning_rate"],
            "max_depth": params["max_depth"],
            "n_estimators": params["n_estimators"],
        })
        return auc

    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_params
    best_auc = study.best_value
    print(f"  Best CV AUC    : {best_auc:.4f}")
    print(f"  Best params    : {best}")
    print(f"  Search time    : {elapsed:.1f}s")

    trial_df = pd.DataFrame(trial_records)
    return study, best_auc, trial_df


# =============================================================================
# 5. VISUALIZATION — Search space exploration comparison
# =============================================================================
def visualize_results(optuna_df, output_path="tuning_results.png"):
    """
    Plot Bayesian Optimization trial progression and search space coverage.
    Shows how TPE converges on good hyperparameter regions over time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Day 116 — Hyperparameter Tuning: Bayesian Search Analysis",
                 fontsize=13, fontweight="bold")

    # Left: AUC progression over trials
    ax = axes[0]
    ax.plot(optuna_df["trial"], optuna_df["auc"],
            alpha=0.5, color="#6FA8DC", linewidth=1.2, label="Trial AUC")
    running_best = optuna_df["auc"].cummax()
    ax.plot(optuna_df["trial"], running_best,
            color="#E69138", linewidth=2.2, label="Running Best")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("CV AUC (ROC)")
    ax.set_title("Convergence: AUC Over Trials")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(optuna_df["auc"].min() - 0.01, 1.0)

    # Right: Search space scatter — learning_rate vs max_depth, colored by AUC
    ax = axes[1]
    scatter = ax.scatter(
        optuna_df["learning_rate"],
        optuna_df["max_depth"],
        c=optuna_df["auc"],
        cmap="YlOrRd",
        s=70, alpha=0.8, edgecolors="grey", linewidths=0.4
    )
    plt.colorbar(scatter, ax=ax, label="CV AUC")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_xscale("log")
    ax.set_ylabel("Max Depth")
    ax.set_title("Search Space: LR vs Depth (color = AUC)")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  [SAVED] {output_path}")
    plt.close()


# =============================================================================
# 6. PRODUCTION HYPERPARAMETER TUNER — Reusable wrapper class
# =============================================================================
class HyperparameterTuner:
    """
    Production-grade tuner wrapper.
    
    Encapsulates the tuning loop used in ML pipelines at scale.
    Supports three backends: grid, random, bayesian.
    
    Usage:
        tuner = HyperparameterTuner(estimator, search_space, method="bayesian")
        tuner.fit(X_train, y_train)
        print(tuner.best_params_)
        predictions = tuner.predict_proba(X_test)
    """

    METHODS = ("grid", "random", "bayesian")

    def __init__(self, estimator, search_space: dict,
                 method: str = "bayesian",
                 n_trials: int = 30,
                 cv_folds: int = 5,
                 scoring: str = "roc_auc",
                 random_state: int = 42):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.estimator = estimator
        self.search_space = search_space
        self.method = method
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.trial_history_ = None

    def fit(self, X, y):
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                              random_state=self.random_state)

        if self.method == "grid":
            searcher = GridSearchCV(self.estimator, self.search_space,
                                     scoring=self.scoring, cv=cv, n_jobs=-1)
            searcher.fit(X, y)
            self.best_estimator_ = searcher.best_estimator_
            self.best_params_ = searcher.best_params_
            self.best_score_ = searcher.best_score_

        elif self.method == "random":
            searcher = RandomizedSearchCV(self.estimator, self.search_space,
                                           n_iter=self.n_trials,
                                           scoring=self.scoring, cv=cv,
                                           n_jobs=-1,
                                           random_state=self.random_state)
            searcher.fit(X, y)
            self.best_estimator_ = searcher.best_estimator_
            self.best_params_ = searcher.best_params_
            self.best_score_ = searcher.best_score_

        elif self.method == "bayesian":
            records = []

            def objective(trial):
                params = {}
                for key, val in self.search_space.items():
                    if isinstance(val, list):
                        params[key] = trial.suggest_categorical(key, val)
                    elif isinstance(val, tuple) and len(val) == 2:
                        if isinstance(val[0], int):
                            params[key] = trial.suggest_int(key, val[0], val[1])
                        else:
                            params[key] = trial.suggest_float(
                                key, val[0], val[1],
                                log=(key == "learning_rate")
                            )
                est = self.estimator.__class__(
                    random_state=self.random_state, **params)
                scores = cross_val_score(est, X, y, cv=cv,
                                          scoring=self.scoring, n_jobs=-1)
                records.append({"params": params, "score": scores.mean()})
                return scores.mean()

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state)
            )
            study.optimize(objective, n_trials=self.n_trials,
                            show_progress_bar=False)

            self.best_params_ = study.best_params
            self.best_score_ = study.best_value
            self.trial_history_ = pd.DataFrame(records)

            best_est = self.estimator.__class__(
                random_state=self.random_state, **self.best_params_)
            best_est.fit(X, y)
            self.best_estimator_ = best_est

        return self

    def predict_proba(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError("Call fit() before predict_proba()")
        return self.best_estimator_.predict_proba(X)

    def summary(self):
        print(f"\n── HyperparameterTuner Summary ────────────────────────────────")
        print(f"  Method      : {self.method}")
        print(f"  Best Score  : {self.best_score_:.4f}")
        print(f"  Best Params : {self.best_params_}")


# =============================================================================
# MAIN — Full demonstration pipeline
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Day 116: Hyperparameter Tuning Theory")
    print("=" * 60)

    # --- Data ---
    X, y = load_dataset(n_samples=5000)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\nDataset split:")
    print(f"  Train  : {X_train.shape[0]} samples")
    print(f"  Val    : {X_val.shape[0]} samples")
    print(f"  Test   : {X_test.shape[0]} samples  ← touched only at the end")

    # --- Experiments ---
    baseline_auc, _ = run_baseline(X_train, X_val, y_train, y_val)
    gs_model, gs_auc, _ = run_grid_search(X_train, y_train, cv)
    rs_model, rs_auc, _ = run_random_search(X_train, y_train, cv, n_iter=30)
    study, bo_auc, optuna_df = run_bayesian_optuna(X_train, y_train, cv, n_trials=30)

    # --- Visualization ---
    visualize_results(optuna_df)

    # --- Final test-set evaluation on best model (Bayesian) ---
    # Retrain on full train+val with best params
    best_params = study.best_params
    final_model = GradientBoostingClassifier(random_state=42, **best_params)
    final_model.fit(X_trainval, y_trainval)
    test_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Baseline (default)         : {baseline_auc:.4f}  (val AUC)")
    print(f"  Grid Search best CV AUC    : {gs_auc:.4f}")
    print(f"  Random Search best CV AUC  : {rs_auc:.4f}")
    print(f"  Bayesian Optim best CV AUC : {bo_auc:.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Final TEST SET AUC         : {test_auc:.4f}  ← reported once")
    print(f"  AUC lift vs baseline       : +{(test_auc - baseline_auc):.4f}")

    # --- HyperparameterTuner demo ---
    print("\n── HyperparameterTuner Class Demo ──────────────────────────────")
    search_space = {
        "n_estimators": (100, 400),
        "learning_rate": (0.01, 0.2),
        "max_depth": (3, 7),
        "subsample": (0.7, 1.0),
    }
    tuner = HyperparameterTuner(
        estimator=GradientBoostingClassifier(random_state=42),
        search_space=search_space,
        method="bayesian",
        n_trials=20
    )
    tuner.fit(X_train, y_train)
    tuner.summary()

    print("\n[DONE] All Day 116 experiments complete.")
