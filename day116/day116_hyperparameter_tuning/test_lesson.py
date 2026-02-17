"""
Day 116: Hyperparameter Tuning Theory — Test Suite
====================================================
Tests verify conceptual understanding and implementation correctness:
  - Hyperparameter vs parameter distinction
  - Search strategy output contracts
  - HyperparameterTuner class interface
  - Cross-validation integrity (no test set leakage)
  - Bayesian trial convergence property
"""

import pytest
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score

from lesson_code import (
    load_dataset,
    HyperparameterTuner,
    run_baseline,
    run_grid_search,
    run_random_search,
    run_bayesian_optuna,
)


@pytest.fixture(scope="module")
def dataset():
    """Shared small dataset for all tests — 1000 samples for speed."""
    X, y = load_dataset(n_samples=1000, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    return X_train, X_test, y_train, y_test, cv


# ── Concept Tests ──────────────────────────────────────────────────────────

class TestHyperparameterConcepts:
    """Verify understanding of parameters vs hyperparameters."""

    def test_parameters_learned_during_training(self):
        """GBM estimators are parameters: they exist only after fit()."""
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        X, y = make_classification(n_samples=200, random_state=42)
        assert not hasattr(model, "estimators_"), \
            "estimators_ should not exist before fit()"
        model.fit(X, y)
        assert hasattr(model, "estimators_"), \
            "estimators_ (learned parameters) must exist after fit()"
        assert model.n_estimators_ == 10

    def test_hyperparameters_are_constructor_arguments(self):
        """Hyperparameters are set before training and don't change during it."""
        params = {"n_estimators": 77, "learning_rate": 0.07, "max_depth": 4}
        model = GradientBoostingClassifier(**params, random_state=42)
        X, y = make_classification(n_samples=200, random_state=42)
        model.fit(X, y)
        # Hyperparameters are unchanged by training
        assert model.n_estimators == 77
        assert model.learning_rate == 0.07
        assert model.max_depth == 4

    def test_default_hyperparameters_exist(self):
        """Scikit-learn defaults are the baseline we should always beat."""
        model = GradientBoostingClassifier()
        assert model.learning_rate == 0.1
        assert model.max_depth == 3
        assert model.subsample == 1.0


# ── Data Integrity Tests ───────────────────────────────────────────────────

class TestCrossValidationIntegrity:
    """Verify proper train/val/test split discipline."""

    def test_test_set_not_used_in_cv(self, dataset):
        """Cross-validation must not have access to test labels."""
        X_train, X_test, y_train, y_test, cv = dataset
        gs = GridSearchCV(
            GradientBoostingClassifier(n_estimators=10, random_state=42),
            {"max_depth": [2, 3]},
            cv=cv, scoring="roc_auc"
        )
        gs.fit(X_train, y_train)
        # Test set is untouched — evaluate independently
        test_auc = roc_auc_score(
            y_test, gs.predict_proba(X_test)[:, 1]
        )
        # Both scores should be realistic (above 0.5 for non-trivial data)
        assert gs.best_score_ > 0.5
        assert test_auc > 0.5

    def test_cv_folds_are_stratified(self, dataset):
        """StratifiedKFold preserves class ratio in each fold."""
        X_train, _, y_train, _, _ = dataset
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        global_ratio = y_train.mean()
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            fold_ratio = y_train[val_idx].mean()
            assert abs(fold_ratio - global_ratio) < 0.08, \
                f"Fold {fold_idx} class imbalance diverged: {fold_ratio:.2f} vs {global_ratio:.2f}"


# ── Grid Search Tests ─────────────────────────────────────────────────────

class TestGridSearch:
    def test_grid_search_explores_all_combinations(self, dataset):
        """GridSearchCV must evaluate every combination in param_grid."""
        X_train, _, y_train, _, cv = dataset
        param_grid = {"max_depth": [2, 4], "learning_rate": [0.05, 0.1]}
        expected_combos = 4  # 2 × 2
        gs = GridSearchCV(
            GradientBoostingClassifier(n_estimators=10, random_state=42),
            param_grid, cv=cv, scoring="roc_auc"
        )
        gs.fit(X_train, y_train)
        assert len(gs.cv_results_["params"]) == expected_combos

    def test_grid_search_best_params_in_grid(self, dataset):
        """Best params must come from the defined grid, not outside it."""
        X_train, _, y_train, _, cv = dataset
        param_grid = {"max_depth": [2, 4, 6]}
        gs = GridSearchCV(
            GradientBoostingClassifier(n_estimators=10, random_state=42),
            param_grid, cv=cv
        )
        gs.fit(X_train, y_train)
        assert gs.best_params_["max_depth"] in [2, 4, 6]

    def test_grid_search_returns_best_score_above_baseline(self, dataset):
        """Grid Search should find params with AUC above random guessing."""
        X_train, _, y_train, _, cv = dataset
        param_grid = {"max_depth": [3, 5], "learning_rate": [0.05, 0.15]}
        gs = GridSearchCV(
            GradientBoostingClassifier(n_estimators=20, random_state=42),
            param_grid, cv=cv, scoring="roc_auc"
        )
        gs.fit(X_train, y_train)
        assert gs.best_score_ > 0.65, \
            f"Expected AUC > 0.65, got {gs.best_score_:.4f}"


# ── Random Search Tests ───────────────────────────────────────────────────

class TestRandomSearch:
    def test_random_search_n_iter_controls_trial_count(self, dataset):
        """n_iter must match the number of evaluated configurations."""
        from scipy.stats import randint, uniform
        X_train, _, y_train, _, cv = dataset
        n_iter = 8
        rs = RandomizedSearchCV(
            GradientBoostingClassifier(n_estimators=10, random_state=42),
            {"max_depth": randint(2, 7), "learning_rate": uniform(0.01, 0.2)},
            n_iter=n_iter, cv=cv, scoring="roc_auc", random_state=42
        )
        rs.fit(X_train, y_train)
        assert len(rs.cv_results_["params"]) == n_iter

    def test_random_search_samples_continuous_range(self, dataset):
        """Random Search must produce non-grid, continuous values."""
        from scipy.stats import uniform
        X_train, _, y_train, _, cv = dataset
        rs = RandomizedSearchCV(
            GradientBoostingClassifier(n_estimators=10, random_state=42),
            {"learning_rate": uniform(0.01, 0.29)},
            n_iter=15, cv=cv, scoring="roc_auc", random_state=42
        )
        rs.fit(X_train, y_train)
        lrs = [p["learning_rate"] for p in rs.cv_results_["params"]]
        unique_lrs = set(lrs)
        assert len(unique_lrs) > 5, "Expected diverse continuous samples"


# ── Bayesian Optimization Tests ───────────────────────────────────────────

class TestBayesianOptimization:
    def test_optuna_returns_correct_trial_count(self, dataset):
        """Optuna study must run exactly n_trials trials."""
        from sklearn.model_selection import cross_val_score
        X_train, _, y_train, _, cv = dataset
        n_trials = 6

        study = optuna.create_study(direction="maximize")

        def objective(trial):
            lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            depth = trial.suggest_int("max_depth", 2, 5)
            model = GradientBoostingClassifier(
                n_estimators=20, learning_rate=lr,
                max_depth=depth, random_state=42
            )
            return cross_val_score(
                model, X_train, y_train, cv=cv, scoring="roc_auc"
            ).mean()

        study.optimize(objective, n_trials=n_trials)
        assert len(study.trials) == n_trials

    def test_bayesian_best_trial_is_best_auc(self, dataset):
        """study.best_value must equal the maximum AUC across all trials."""
        from sklearn.model_selection import cross_val_score
        X_train, _, y_train, _, cv = dataset

        study = optuna.create_study(direction="maximize")
        all_scores = []

        def objective(trial):
            lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            model = GradientBoostingClassifier(
                n_estimators=15, learning_rate=lr, random_state=42
            )
            score = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="roc_auc"
            ).mean()
            all_scores.append(score)
            return score

        study.optimize(objective, n_trials=8)
        assert abs(study.best_value - max(all_scores)) < 1e-8

    def test_bayesian_beats_random_guessing(self, dataset):
        """Bayesian Optimization must converge above 0.65 AUC in 15 trials."""
        X_train, _, y_train, _, cv = dataset
        _, best_auc, _ = run_bayesian_optuna(X_train, y_train, cv, n_trials=15)
        assert best_auc > 0.65, \
            f"Bayesian AUC {best_auc:.4f} should exceed 0.65"


# ── HyperparameterTuner Class Tests ───────────────────────────────────────

class TestHyperparameterTuner:
    def test_tuner_raises_on_invalid_method(self):
        """Unsupported method should raise ValueError immediately."""
        with pytest.raises(ValueError, match="method must be one of"):
            HyperparameterTuner(
                GradientBoostingClassifier(), {}, method="genetic"
            )

    def test_tuner_predict_before_fit_raises(self):
        """predict_proba before fit() must raise RuntimeError."""
        tuner = HyperparameterTuner(
            GradientBoostingClassifier(), {}, method="bayesian"
        )
        X = np.random.rand(10, 3)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            tuner.predict_proba(X)

    def test_tuner_bayesian_fit_populates_best_params(self, dataset):
        """After fit(), best_params_ and best_score_ must be populated."""
        X_train, _, y_train, _, _ = dataset
        search_space = {
            "n_estimators": (50, 150),
            "learning_rate": (0.05, 0.2),
            "max_depth": (3, 5),
        }
        tuner = HyperparameterTuner(
            GradientBoostingClassifier(random_state=42),
            search_space,
            method="bayesian",
            n_trials=5,
            cv_folds=3
        )
        tuner.fit(X_train, y_train)
        assert tuner.best_params_ is not None
        assert isinstance(tuner.best_score_, float)
        assert tuner.best_score_ > 0.5
        assert "n_estimators" in tuner.best_params_

    def test_tuner_grid_method_works(self, dataset):
        """Grid method must produce best_estimator_ that can predict."""
        X_train, X_test, y_train, _, _ = dataset
        search_space = {"max_depth": [3, 5], "learning_rate": [0.1, 0.15]}
        tuner = HyperparameterTuner(
            GradientBoostingClassifier(n_estimators=20, random_state=42),
            search_space,
            method="grid",
            cv_folds=3
        )
        tuner.fit(X_train, y_train)
        proba = tuner.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_tuner_random_method_works(self, dataset):
        """Random Search method must produce valid probability outputs."""
        from scipy.stats import uniform, randint
        X_train, X_test, y_train, _, _ = dataset
        search_space = {
            "max_depth": randint(3, 6),
            "learning_rate": uniform(0.05, 0.15),
        }
        tuner = HyperparameterTuner(
            GradientBoostingClassifier(n_estimators=20, random_state=42),
            search_space,
            method="random",
            n_trials=5,
            cv_folds=3
        )
        tuner.fit(X_train, y_train)
        proba = tuner.predict_proba(X_test)
        assert proba.shape[1] == 2

    def test_tuner_trial_history_recorded_for_bayesian(self, dataset):
        """Bayesian tuner must record per-trial history in trial_history_."""
        X_train, _, y_train, _, _ = dataset
        search_space = {"max_depth": (3, 5), "learning_rate": (0.05, 0.2)}
        tuner = HyperparameterTuner(
            GradientBoostingClassifier(n_estimators=15, random_state=42),
            search_space,
            method="bayesian",
            n_trials=6,
            cv_folds=3
        )
        tuner.fit(X_train, y_train)
        assert tuner.trial_history_ is not None
        assert len(tuner.trial_history_) == 6
        assert "score" in tuner.trial_history_.columns


# ── Integration Test ───────────────────────────────────────────────────────

class TestEndToEndPipeline:
    def test_full_pipeline_no_leakage(self, dataset):
        """
        End-to-end: tune on train/val only, evaluate test set once.
        This mirrors the correct production workflow.
        """
        X_train, X_test, y_train, y_test, cv = dataset

        # Tune on train only
        search_space = {"max_depth": (3, 6), "learning_rate": (0.05, 0.2)}
        tuner = HyperparameterTuner(
            GradientBoostingClassifier(n_estimators=30, random_state=42),
            search_space,
            method="bayesian",
            n_trials=8,
            cv_folds=3
        )
        tuner.fit(X_train, y_train)

        # Evaluate test set exactly once
        test_proba = tuner.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)

        assert test_auc > 0.60, f"Test AUC {test_auc:.4f} is unexpectedly low"
        # CV AUC and test AUC should be in same ballpark (no severe leakage)
        assert abs(tuner.best_score_ - test_auc) < 0.15, \
            f"Large gap between CV ({tuner.best_score_:.4f}) and test ({test_auc:.4f}) AUC — check for leakage"
