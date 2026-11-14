"""Add your own method to a board: score a new run detector on SWE-Gym next to the paper's rows.

Run from a checkout after installing the bench extra:

    pip install "auditable[bench]"
    python examples/example_add_your_method.py

The detector is a logistic regression on two hand-made features of the typed run graph: the
number of steps and the longest dependency span (how far back any step reaches). The board
scores it as it scores every supervised method, by mean ROC-AUC over five seeds of
stratified 5-fold cross-validation with a fresh estimator per fold.

The script prints the small board (the size baseline ``run.lr_flat``, the dependency
regression ``run.lr_flatdep``, ``run.pyod_ecod``, the random control, and the new row) and
the rank and score of the new row beside the size baseline. Read any gain over
``run.lr_flat`` with care. The longest span grows with run length, and GRADE retired on
2026-09-08 the reading that dependency structure predicts failure beyond run size, so a
claim for a new method needs a comparison with a size-only model fitted as flexibly as
yours. A rank describes a method on one population and protocol; it does not explain why
the method works.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from auditable.base import BaseRunDetector, MethodCard
from auditable.bench import leaderboard
from auditable.featurize import normalize_trace, run_max_span


class StepsAndSpan(BaseRunDetector):
    """Run length and the longest dependency span, weighed jointly by a linear model."""

    card = MethodCard(
        method_id="run.lr_steps_span", name="Logistic regression on step count and longest span",
        family="run", fit_scope="inductive", supervised=True, output_kind="score", fidelity="original",
        requires=frozenset({"deps"}), extras=("tabular-ad",),
        summary="Balanced logistic regression on the log step count and the longest dependency span.")

    def features(self, runs):
        # log1p tames the long tail of run length; the board calls this once and indexes the matrix per fold.
        return np.array([[np.log1p(len(normalize_trace(g))), run_max_span(g)] for g in runs])

    def fit(self, X, y=None):
        X = X if getattr(X, "ndim", 0) == 2 else self.features(X)
        self.pipeline_ = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(X, y)
        self.decision_scores_ = self.pipeline_.decision_function(X)
        return self

    def decision_function(self, X):
        return self.pipeline_.decision_function(X if getattr(X, "ndim", 0) == 2 else self.features(X))


def main():
    # The size baseline and the paper's dependency and unsupervised rows stay on the board so the
    # new row is read against them; extra= adds the new row to those named in methods=.
    board = leaderboard("detect/swegym", methods=["run.lr_flat", "run.lr_flatdep", "run.pyod_ecod", "run.random"],
                        extra=[StepsAndSpan])
    print(board)
    key = StepsAndSpan.card.method_id
    print(f"\n{key} ranks {board.rank_of(key)} of {len(board.ranked())}, "
          f"ROC-AUC {board[key].scores['roc_auc']:.3f} against {board['run.lr_flat'].scores['roc_auc']:.3f} "
          f"for the size baseline")


if __name__ == "__main__":
    main()
