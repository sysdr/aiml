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
