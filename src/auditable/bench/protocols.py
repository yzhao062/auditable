"""Evaluation protocols of the CatchBench board, so a score here matches the board's row.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release 0.1.3,
``catchbench/via_auditable/protocols.py`` and ``run_tabular.py``, which mirror GRADE
(https://github.com/yzhao062/grade, MIT License) at commit 3839a57
(``experiment/agent_failure_detection.py``). The folds, seeds and score calls are kept as
they are there; only the estimator inside each fold is whatever method the board scores.

Which protocol a run detector gets follows its :class:`~auditable.base.MethodCard`:

- supervised (``card.supervised``): ROC-AUC averaged over five seeds of stratified 5-fold
  cross-validation, a fresh estimator per fold (:func:`cv_auc`);
- unsupervised: one ``fit_score`` over the whole population, ROC-AUC of those scores
  (:func:`batch_auc`), as the board scores PyOD and PyGOD detectors;
- ``run.random``: five draws from one ``RandomState(0)``, mean ROC-AUC (:func:`random_auc`).

G-Safeguard keeps the board's own loop (:func:`gsafeguard_cv_auc`, three seeds of 5 folds).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

SEEDS = range(5)  # GRADE agent_failure_detection.SEEDS and agent_failure_localization.SEEDS


def _take(X: Any, idx):
    if hasattr(X, "shape"):
        return X[idx]
    return [X[int(i)] for i in idx]


def cv_auc_matrix(make: Callable[[], Any], X: Any, y: Any, seeds=SEEDS, n_splits: int = 5):
    """ROC-AUC per (seed, fold), shape ``(n_seeds, n_folds)``: ``StratifiedKFold(n_splits,
    shuffle=True, random_state=s)`` for each seed, a fresh estimator from ``make()`` per fold,
    fit on the training rows and scored with ``decision_function`` on the held-out rows.
    ``X`` is a feature matrix or a list of runs; either is indexed by the fold's row indices."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    rows = []
    for s in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        fold_aucs = []
        for train, test in cv.split(np.zeros((len(y), 1)), y):
            det = make()
            det.fit(_take(X, train), y[train])
            fold_aucs.append(roc_auc_score(y[test], det.decision_function(_take(X, test))))
        rows.append(fold_aucs)
    return np.array(rows)


def cv_auc(make: Callable[[], Any], runs: Sequence[Any], y: Any) -> float:
    """Mean of :func:`cv_auc_matrix`. A method with a ``features`` method gets its features
    computed once over all runs and the folds index the feature matrix, as the board does for
    the GRADE-layer regressions; any other method is fit on the runs themselves."""
    import numpy as np

    probe = make()
    X = probe.features(runs) if hasattr(probe, "features") else list(runs)
    return float(np.mean(cv_auc_matrix(make, X, y)))


def batch_auc(make: Callable[[], Any], runs: Sequence[Any], y: Any) -> float:
    """ROC-AUC of one ``fit_score`` over all runs against the labels."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, make().fit_score(list(runs))))


def random_auc(make: Callable[..., Any], runs: Sequence[Any], y: Any, draws: int = 5) -> float:
    """Mean ROC-AUC of ``draws`` successive ``fit_score`` calls on one detector built with
    ``random_state=RandomState(0)``."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    det = make(random_state=np.random.RandomState(0))
    return float(np.mean([roc_auc_score(y, det.fit_score(list(runs))) for _ in range(draws)]))


def gsafeguard_cv_auc(make: Callable[..., Any], runs: Sequence[Any], y: Any, *, seed: int = 0,
                      n_splits: int = 5, n_seeds: int = 3) -> float:
    """The board's G-Safeguard loop: ``StratifiedKFold(n_splits, shuffle=True,
    random_state=seed + s)`` for each of ``n_seeds`` seeds, folds with a single class on either
    side skipped, a network seeded ``seed + s`` per fold; mean held-out ROC-AUC, or 0.5 when no
    fold is usable."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    feats = make().features(list(runs))
    n = len(feats)
    if n < n_splits or len(set(y.tolist())) < 2:
        return 0.5
    aucs = []
    for s in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + s)
        for tr, te in skf.split(np.zeros(n), y):
            if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
                continue
            det = make(seed=seed + s)
            det.fit([feats[int(i)] for i in tr], y[tr])
            aucs.append(roc_auc_score(y[te], det.decision_function([feats[int(i)] for i in te])))
    return float(np.mean(aucs)) if aucs else 0.5


def run_detection_auc(entry: Any, runs: Sequence[Any], y: Any) -> float:
    """The board protocol for one run-detector entry (see the module docstring)."""
    card = entry.card
    if card.method_id == "run.gsafeguard_lite":
        return gsafeguard_cv_auc(entry.make, runs, y)
    if card.method_id == "run.random":
        return random_auc(entry.make, runs, y)
    if card.supervised:
        return cv_auc(entry.make, runs, y)
    return batch_auc(entry.make, runs, y)
