"""Unsupervised PyOD detectors on the GRADE flat layer of each run.

Six shallow tabular outlier detectors (ECOD, Isolation Forest, KNN, LOF, COPOD, HBOS) read
the four size and count columns of a run (steps, tool calls, decisions, agents),
standardized over the fitted batch. No failure labels are used: a higher score marks a
more unusual run, which the detection board reads as more likely failed. They are
inductive: ``fit`` stores the scaler and the PyOD model, and ``decision_function`` scores
unseen runs with them.

``fit_score`` returns PyOD's ``decision_scores_``. For KNN and LOF these differ from
``decision_function`` on the same training batch, because PyOD leaves a training point out
of its own neighbourhood in ``decision_scores_`` and does not for new points.

Ported from CatchBench (https://github.com/yzhao062/catchbench, MIT License) at release
0.1.2 (``detection.PyODFlatten``, ``pyod_extra``, ``live._PrefixECOD``).
"""
from __future__ import annotations

from typing import Any, Tuple

from auditable.base import BaseRunDetector, Citation, MethodCard
from auditable.methods import register
from auditable.methods.run.supervised import _layer_matrix

_CATCHBENCH = Citation(key="CatchBench (2026)", title="CatchBench: When Can an Agent Failure Be Caught?",
                       arxiv="2608.22808")
_PYOD = Citation(key="PyOD (Zhao et al., 2019)", title="PyOD: A Python Toolbox for Scalable Outlier Detection",
                 arxiv="1901.01588")
_ECOD = Citation(key="ECOD (Li et al., 2022)",
                 title="ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions",
                 arxiv="2201.00382")
_IFOREST = Citation(key="Isolation Forest (Liu et al., 2008)", title="Isolation Forest",
                    url="https://doi.org/10.1109/ICDM.2008.17")
_KNN = Citation(key="KNN outlier score (Ramaswamy et al., 2000)",
                title="Efficient Algorithms for Mining Outliers from Large Data Sets")
_LOF = Citation(key="LOF (Breunig et al., 2000)", title="LOF: Identifying Density-Based Local Outliers",
                url="https://doi.org/10.1145/335191.335388")
_COPOD = Citation(key="COPOD (Li et al., 2020)", title="COPOD: Copula-Based Outlier Detection",
                  arxiv="2009.09463")
_HBOS = Citation(key="HBOS (Goldstein and Dengel, 2012)",
                 title="Histogram-based Outlier Score (HBOS): A Fast Unsupervised Anomaly Detection Algorithm")


class _PyODRunDetector(BaseRunDetector):
    """Shared core: standardize the flat layer over the batch, then fit one PyOD model."""

    def _new_detector(self):
        raise NotImplementedError

    def features(self, X: Any):
        """The flat layer this detector reads, one row per trace; a matrix passes through."""
        return _layer_matrix(X, "flat")

    def fit(self, X: Any, y: Any = None) -> "_PyODRunDetector":
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(self.features(X))
        self.detector_ = self._new_detector()
        self.detector_.fit(Z)
        self.decision_scores_ = self.detector_.decision_scores_
        return self

    def decision_function(self, X: Any):
        if not hasattr(self, "detector_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit(X) first")
        return self.detector_.decision_function(self.scaler_.transform(self.features(X)))


def _card(method_id: str, name: str, aliases: Tuple[str, ...], summary: str, *cite: Citation) -> MethodCard:
    return MethodCard(
        method_id=method_id,
        name=name,
        family="run",
        fit_scope="inductive",
        supervised=False,
        output_kind="score",
        fidelity="wrapper",
        requires=frozenset({"agent", "kind"}),
        cite=cite + (_PYOD, _CATCHBENCH),
        extras=("tabular-ad",),
        board_aliases=aliases,
        benchmark_role="baseline",
        summary=summary,
    )


@register
class PyODECOD(_PyODRunDetector):
    """ECOD (empirical cumulative distribution outlier detection) on the flat layer."""

    card = _card("run.pyod_ecod", "ECOD on run size", ("pyod-flatten (ECOD)", "pyod (ECOD)"),
                 "Summed negative log empirical tail probabilities of the size columns.", _ECOD)

    def _new_detector(self):
        from pyod.models.ecod import ECOD

        return ECOD()


@register
class PyODIForest(_PyODRunDetector):
    """Isolation Forest on the flat layer, seeded for repeatable scores."""

    card = _card("run.pyod_iforest", "Isolation Forest on run size", ("pyod-iforest",),
                 "Average isolation depth of a run over a seeded random forest.", _IFOREST)

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state

    def _new_detector(self):
        from pyod.models.iforest import IForest

        return IForest(random_state=self.random_state)


@register
class PyODKNN(_PyODRunDetector):
    """k-nearest-neighbour distance (PyOD defaults) on the flat layer."""

    card = _card("run.pyod_knn", "KNN distance on run size", ("pyod-knn",),
                 "Distance to the k-th nearest other run.", _KNN)

    def _new_detector(self):
        from pyod.models.knn import KNN

        return KNN()


@register
class PyODLOF(_PyODRunDetector):
    """Local outlier factor (PyOD defaults) on the flat layer."""

    card = _card("run.pyod_lof", "Local outlier factor on run size", ("pyod-lof",),
                 "Density of a run relative to that of its neighbours.", _LOF)

    def _new_detector(self):
        from pyod.models.lof import LOF

        return LOF()


@register
class PyODCOPOD(_PyODRunDetector):
    """COPOD (copula-based outlier detection) on the flat layer."""

    card = _card("run.pyod_copod", "COPOD on run size", ("pyod-copod",),
                 "Summed negative log empirical-copula tail probabilities of the size columns.", _COPOD)

    def _new_detector(self):
        from pyod.models.copod import COPOD

        return COPOD()


@register
class PyODHBOS(_PyODRunDetector):
    """HBOS (histogram-based outlier score) on the flat layer."""

    card = _card("run.pyod_hbos", "HBOS on run size", ("pyod-hbos",),
                 "Sum of log inverse histogram densities over the size columns.", _HBOS)

    def _new_detector(self):
        from pyod.models.hbos import HBOS

        return HBOS()
