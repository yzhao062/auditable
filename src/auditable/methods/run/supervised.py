"""Supervised run-level detectors on GRADE structural layers.

A standardized, class-balanced logistic regression (liblinear) on one GRADE feature
layer per run, the learner GRADE's failure-detection study uses
(``experiment/agent_failure_detection.py`` at commit 3839a57). The three registered
layers are the ones the CatchBench detection board reports. They are inductive: ``fit``
learns the scaler and coefficients, and ``decision_function`` scores unseen runs with
the fitted pipeline.

What a higher score than ``run.lr_flat`` does not show: GRADE retired, on 2026-09-08,
the reading that the dependency layers predict failure beyond run size; a linear size
baseline cannot separate dependency shape from a nonlinear function of size.
"""
from __future__ import annotations

from typing import Any

from auditable.base import BaseRunDetector, Citation, MethodCard
from auditable.methods import register

_GRADE = Citation(key="GRADE (Zhao, 2026)",
                  title="GRADE: Graph Representation of LLM Agent Dependency and Execution",
                  arxiv="2606.22741")


def _layer_matrix(X: Any, layer: str):
    """A feature matrix as given, or the GRADE ``layer`` of each trace in ``X``."""
    import numpy as np

    if hasattr(X, "shape") and getattr(X, "ndim", 0) == 2:
        return np.asarray(X, dtype=float)
    from auditable.featurize import grade_layer_matrix

    return grade_layer_matrix(list(X), layer)


class _LayerLogisticDetector(BaseRunDetector):
    _layer: str = "full"
    _estimator_type = "classifier"

    def __init__(self, C: float = 1.0) -> None:
        self.C = C

    def _pipeline(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        lr = (LogisticRegression(solver="liblinear", class_weight="balanced") if self.C == 1.0
              else LogisticRegression(solver="liblinear", class_weight="balanced", C=self.C))
        return make_pipeline(StandardScaler(), lr)

    def features(self, X: Any):
        """The GRADE layer this detector reads, one row per trace; a matrix passes through.
        Compute it once and pass the matrix to ``fit`` and ``decision_function`` when the
        same traces are scored many times (cross-validation)."""
        return _layer_matrix(X, self._layer)

    def fit(self, X: Any, y: Any = None) -> "_LayerLogisticDetector":
        import numpy as np

        if y is None:
            raise ValueError(f"{type(self).__name__} is supervised; pass run labels y (1 = failed)")
        M = _layer_matrix(X, self._layer)
        self.pipeline_ = self._pipeline().fit(M, np.asarray(y))
        self.classes_ = self.pipeline_.classes_
        self.decision_scores_ = self.pipeline_.decision_function(M)
        return self

    def decision_function(self, X: Any):
        if not hasattr(self, "pipeline_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit(X, y) first")
        return self.pipeline_.decision_function(_layer_matrix(X, self._layer))

    def predict_proba(self, X: Any):
        if not hasattr(self, "pipeline_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit(X, y) first")
        return self.pipeline_.predict_proba(_layer_matrix(X, self._layer))


def _card(method_id: str, layer: str, name: str, alias: str, summary: str) -> MethodCard:
    return MethodCard(
        method_id=method_id,
        name=name,
        family="run",
        fit_scope="inductive",
        supervised=True,
        output_kind="score",
        fidelity="original",
        requires=frozenset({"agent", "kind", "deps"}) if layer != "flat" else frozenset({"kind"}),
        cite=(_GRADE,),
        extras=("tabular-ad",),
        board_aliases=(alias,),
        benchmark_role="baseline",
        summary=summary,
    )


@register
class LogisticFlat(_LayerLogisticDetector):
    """Logistic regression on run size and counts (GRADE ``flat``, 4 columns)."""

    _layer = "flat"
    card = _card("run.lr_flat", "flat", "Logistic regression on run size (GRADE flat)", "size (flat)",
                 "Standardized balanced logistic regression on step, tool-call, decision and agent counts.")


@register
class LogisticFlatDep(_LayerLogisticDetector):
    """Logistic regression on size plus size-normalized dependency density and shape
    (GRADE ``flatdep``, 8 columns)."""

    _layer = "flatdep"
    card = _card("run.lr_flatdep", "flatdep", "Logistic regression on size and dependency (GRADE flatdep)",
                 "auditable (size+deps)",
                 "Adds per-step dependency density, relative chain depth and hub shares to run size.")


@register
class LogisticFull(_LayerLogisticDetector):
    """Logistic regression on size, execution topology and raw dependency counts (GRADE
    ``full``, 12 columns)."""

    _layer = "full"
    card = _card("run.lr_full", "full", "Logistic regression on all raw layers (GRADE full)", "full",
                 "Size, agent execution topology, and raw dependency-layer counts.")


_CATCHBENCH = Citation(key="CatchBench (2026)", title="CatchBench: When Can an Agent Failure Be Caught?",
                       arxiv="2608.22808")

# CatchBench 0.1.3 ``detection._SPLINE``, the setting its size audit froze before the run.
_SPLINE = {"degree": 3, "n_knots": 5, "knots": "quantile", "extrapolation": "constant", "include_bias": False}


@register
class LogisticSplineFlat(_LayerLogisticDetector):
    """The flexible size control: an additive cubic spline in each of the four GRADE ``flat``
    counts, then the board's standardized, balanced logistic regression.

    Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release
    0.1.3, ``detection._spline_estimator`` (the ``size (spline)`` arm of its detection and LIVE
    size audits). The spline sits inside the pipeline, so under cross-validation its knots come
    from the training folds only. It reads nothing but run size; a method that ranks below it on
    a board does not show signal beyond a nonlinear function of size, which is the reading GRADE
    retired on 2026-09-08 for the linear baseline ``run.lr_flat``.
    """

    _layer = "flat"
    card = MethodCard(
        method_id="run.lr_spline_flat",
        name="Logistic regression on a cubic spline of run size (flexible size control)",
        family="run",
        fit_scope="inductive",
        supervised=True,
        output_kind="score",
        fidelity="faithful",
        requires=frozenset({"kind"}),
        cite=(_CATCHBENCH, _GRADE),
        extras=("tabular-ad",),
        board_aliases=("size (spline)",),
        benchmark_role="control",
        summary="Cubic spline (five quantile knots) of step, tool-call, decision and agent counts, then "
                "the board's logistic regression; a size-only reference that is not a CatchBench board row.",
    )

    def _pipeline(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer

        base = super()._pipeline()  # StandardScaler then the board's logistic regression
        return make_pipeline(SplineTransformer(**_SPLINE), *[step for _, step in base.steps])
