"""Tests for auditable.bench.protocols: CV matrix parity, features() dispatch, batch/random
protocols, G-Safeguard degenerate-fold handling, and card-based dispatch. No data download.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import auditable.bench.protocols as protocols  # noqa: E402
from auditable.base import BaseRunDetector, MethodCard  # noqa: E402
from auditable.bench import Entry  # noqa: E402
from auditable.bench.protocols import (  # noqa: E402
    batch_auc,
    cv_auc,
    cv_auc_matrix,
    gsafeguard_cv_auc,
    random_auc,
    run_detection_auc,
)
from auditable.methods.run.baselines import RandomRun  # noqa: E402


class _LogRegDetector:
    """Minimal detector wrapping standardized LogisticRegression."""

    def fit(self, X, y):
        self.pipe_ = make_pipeline(
            StandardScaler(), LogisticRegression(solver="liblinear", class_weight="balanced")
        ).fit(X, y)
        return self

    def decision_function(self, X):
        return self.pipe_.decision_function(X)


def _sample_matrix_and_labels():
    rng = np.random.RandomState(7)
    X_neg = rng.normal(loc=0.0, scale=1.0, size=(15, 4))
    X_pos = rng.normal(loc=1.2, scale=1.0, size=(15, 4))
    X = np.vstack([X_neg, X_pos])
    y = np.array([0] * 15 + [1] * 15)
    return X, y


def test_cv_auc_matrix_equals_manual_stratified_kfold():
    X, y = _sample_matrix_and_labels()
    got = cv_auc_matrix(_LogRegDetector, X, y)

    expected = []
    for s in range(5):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        fold_aucs = []
        for train, test in cv.split(np.zeros((len(y), 1)), y):
            det = _LogRegDetector()
            det.fit(X[train], y[train])
            fold_aucs.append(roc_auc_score(y[test], det.decision_function(X[test])))
        expected.append(fold_aucs)
    expected = np.array(expected)

    assert got.shape == (5, 5)
    assert np.array_equal(got, expected)


def test_cv_auc_uses_features_when_present_and_runs_otherwise():
    X, y = _sample_matrix_and_labels()
    raw_runs = [{"row": X[i], "tag": f"run-{i}"} for i in range(len(y))]

    feature_calls = []
    seen_fit_inputs = []

    class _WithFeatures(_LogRegDetector):
        def features(self, runs):
            feature_calls.append(len(runs))
            return np.vstack([r["row"] for r in runs])

        def fit(self, data, target):
            seen_fit_inputs.append(type(data))
            return super().fit(data, target)

    auc_with_feats = cv_auc(_WithFeatures, raw_runs, y)
    assert feature_calls == [len(raw_runs)]
    assert all(t is np.ndarray for t in seen_fit_inputs)
    assert auc_with_feats == pytest.approx(float(np.mean(cv_auc_matrix(_LogRegDetector, X, y))))

    raw_fit_inputs = []

    class _RawRunDetector(_LogRegDetector):
        def fit(self, runs, target):
            raw_fit_inputs.append(runs)
            return super().fit(np.vstack([r["row"] for r in runs]), target)

        def decision_function(self, runs):
            return super().decision_function(np.vstack([r["row"] for r in runs]))

    auc_raw = cv_auc(_RawRunDetector, raw_runs, y)
    assert len(raw_fit_inputs) == 25
    assert isinstance(raw_fit_inputs[0][0], dict) and "tag" in raw_fit_inputs[0][0]
    assert auc_raw == pytest.approx(auc_with_feats)


def test_batch_auc_fits_once_over_whole_population():
    runs = [{"score": float(i)} for i in range(10)]
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    calls = []

    class _BatchDetector:
        def fit_score(self, batch):
            calls.append(list(batch))
            return np.array([r["score"] for r in batch])

    got = batch_auc(_BatchDetector, runs, y)
    assert len(calls) == 1 and len(calls[0]) == len(runs)
    assert got == pytest.approx(roc_auc_score(y, [r["score"] for r in runs]))


def test_random_auc_uses_five_draws_from_one_random_state():
    runs = list(range(20))
    y = np.array([0] * 10 + [1] * 10)

    rng = np.random.RandomState(0)
    expected = float(np.mean([roc_auc_score(y, rng.rand(len(runs))) for _ in range(5)]))
    got = random_auc(RandomRun, runs, y)
    assert got == pytest.approx(expected)

    # Confirm it does not reset RandomState(0) on every draw.
    reset_each = float(roc_auc_score(y, np.random.RandomState(0).rand(len(runs))))
    assert got != pytest.approx(reset_each)


def test_gsafeguard_cv_auc_degenerate_fold_skip_and_fallback():
    seen_seeds = []

    class _FakeGSafeguard:
        def __init__(self, seed: int = 0) -> None:
            self.seed = seed

        def features(self, runs):
            return [float(r) for r in runs]

        def fit(self, feats, target):
            assert len(set(np.asarray(target).tolist())) >= 2
            seen_seeds.append(self.seed)
            self.offset_ = 0.0
            return self

        def decision_function(self, feats):
            return np.asarray(feats, dtype=float) + self.offset_

    # Too few samples for n_splits -> immediate 0.5 fallback.
    assert gsafeguard_cv_auc(_FakeGSafeguard, [0, 1, 2, 3], [0, 0, 1, 1], n_splits=5) == 0.5
    # Single class -> immediate 0.5 fallback.
    assert gsafeguard_cv_auc(_FakeGSafeguard, [0, 1, 2, 3, 4], [0, 0, 0, 0, 0], n_splits=5) == 0.5
    # Every fold is degenerate (either train or test has a single class) -> all skipped -> 0.5 fallback.
    with pytest.warns(UserWarning, match="least populated class"):
        assert gsafeguard_cv_auc(_FakeGSafeguard, [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 1], n_splits=5, n_seeds=2) == 0.5
    assert seen_seeds == []

    # 6 samples, 3 splits, 2 positives -> one test fold has only class 0 and must be skipped,
    # leaving 2 usable folds per seed.
    runs = [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]
    y = np.array([0, 0, 0, 0, 1, 1])
    with pytest.warns(UserWarning, match="least populated class"):
        got = gsafeguard_cv_auc(_FakeGSafeguard, runs, y, seed=10, n_splits=3, n_seeds=2)
    assert seen_seeds == [10, 10, 11, 11]
    assert got == pytest.approx(1.0)


def _dummy_entry(method_id: str, *, supervised: bool, fit_scope: str) -> Entry:
    class _Dummy(BaseRunDetector):
        card = MethodCard(
            method_id=method_id,
            name=method_id,
            family="run",
            fit_scope=fit_scope,
            supervised=supervised,
            output_kind="score",
            fidelity="original",
        )

    return Entry(_Dummy)


def test_run_detection_auc_dispatches_by_card(monkeypatch):
    called = []
    monkeypatch.setattr(protocols, "gsafeguard_cv_auc", lambda make, r, y: called.append("gsafeguard") or 0.11)
    monkeypatch.setattr(protocols, "random_auc", lambda make, r, y: called.append("random") or 0.22)
    monkeypatch.setattr(protocols, "cv_auc", lambda make, r, y: called.append("cv") or 0.33)
    monkeypatch.setattr(protocols, "batch_auc", lambda make, r, y: called.append("batch") or 0.44)

    runs, y = [0, 1], [0, 1]
    assert run_detection_auc(_dummy_entry("run.gsafeguard_lite", supervised=True, fit_scope="inductive"), runs, y) == 0.11
    assert run_detection_auc(_dummy_entry("run.random", supervised=False, fit_scope="stateless"), runs, y) == 0.22
    assert run_detection_auc(_dummy_entry("run.sup_other", supervised=True, fit_scope="inductive"), runs, y) == 0.33
    assert run_detection_auc(_dummy_entry("run.unsup_other", supervised=False, fit_scope="transductive"), runs, y) == 0.44
    assert called == ["gsafeguard", "random", "cv", "batch"]
