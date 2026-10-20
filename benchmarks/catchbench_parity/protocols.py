"""Evaluation protocols aligned line by line with the GRADE originals.

The folds, seeds and metrics stay in the benchmark harness; the library supplies only the
estimator called inside each fold. Only the fold loops are rewritten here, because the
originals construct their own scikit-learn model inside the loop. Fold assignment, prefix
truncation and the ranking metrics are imported from GRADE and CatchBench unchanged.
``checks.py`` compares each loop against its original with the original model before
any library model is put inside it.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

SEEDS = range(5)  # GRADE agent_failure_detection.SEEDS and agent_failure_localization.SEEDS


def run_cv_auc(make_detector: Callable[[], Any], X: Any, y: Any, seeds=SEEDS, n_splits: int = 5):
    """ROC-AUC per (seed, fold), shape (n_seeds, n_folds). Mirrors GRADE ``_cv`` at
    3839a57: ``StratifiedKFold(5, shuffle=True, random_state=s)`` per seed, a fresh
    estimator per fold, and ``cross_val_score(scoring="roc_auc")``, which scores a binary
    classifier through ``decision_function``. ``X`` is a feature matrix or a list of traces;
    either is indexed by the fold's sorted row indices, as ``cross_val_score`` does."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    rows = []
    for s in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        fold_aucs = []
        for train, test in cv.split(np.zeros((len(y), 1)), y):
            det = make_detector()
            det.fit(_take(X, train), y[train])
            fold_aucs.append(roc_auc_score(y[test], det.decision_function(_take(X, test))))
        rows.append(fold_aucs)
    return np.array(rows)


def _take(X: Any, idx):
    if hasattr(X, "shape"):
        return X[idx]
    return [X[i] for i in idx]


def step_oof_scores(make_localizer: Callable[[], Any], traces: Sequence[Any], mistakes: Sequence[int],
                    groups, seed: int) -> np.ndarray:
    """Out-of-fold step scores in pooled order (runs in order, steps within a run in
    order). Mirrors GRADE ``_oof_scores``: GRADE's own grouped folds, one model per fold fit
    on the training runs, and one batched prediction over the test runs in pooled order.
    ``mistakes[r]`` is the ``idx`` of run ``r``'s decisive step."""
    from agent_failure_localization import _grouped_folds

    groups = np.asarray(groups)
    scores = np.full(len(groups), np.nan)
    for test in _grouped_folds(groups, 5, seed):
        test_runs = sorted(set(groups[test].tolist()))
        train_runs = sorted(set(groups[~test].tolist()))
        loc = make_localizer()
        loc.fit([traces[r] for r in train_runs], [mistakes[r] for r in train_runs])
        per_run = loc.step_scores_batch([traces[r] for r in test_runs])
        scores[test] = np.concatenate([np.asarray(v, dtype=float) for v in per_run])
    return scores


def gsafeguard_cv_aucs(make_detector: Callable[[int], Any], graphs: Sequence[Any], y: Any, *, seed: int = 0,
                       n_splits: int = 5, n_seeds: int = 3) -> list:
    """Held-out ROC-AUC of every usable (seed, fold), in loop order. Mirrors CatchBench
    ``agent_detectors.gsafeguard_cv_auc`` at 0.1.2 line by line: ``StratifiedKFold(n_splits,
    shuffle=True, random_state=seed + s)`` for each of ``n_seeds`` seeds, folds with a single
    class in the training or held-out part skipped, and a network built per fold with torch
    seeded ``seed + s``. ``make_detector(fold_seed)`` returns a fresh estimator seeded with
    ``fold_seed``; the original's ``seed + s`` serves both as fold seed and as torch seed. The
    board value is ``float(np.mean(aucs))``, or 0.5 when the list is empty (too few runs or
    one class), as in the original."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    n_runs = len(graphs)
    if n_runs < n_splits or len(set(y.tolist())) < 2:
        return []
    aucs = []
    for s in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + s)
        for tr, te in skf.split(np.zeros(n_runs), y):
            if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
                continue  # skip degenerate single-class folds
            det = make_detector(seed + s)
            det.fit([graphs[int(i)] for i in tr], y[tr])
            aucs.append(roc_auc_score(y[te], det.decision_function([graphs[int(i)] for i in te])))
    return aucs
