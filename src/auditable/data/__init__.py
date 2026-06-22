"""The data module: audit the dependency state a decision relied on.

``DataAuditor`` ships a snapshot-freshness rule in v0.1: it scores the age of the
dependency snapshot against a freshness budget and flags a stale snapshot. The mode is
always made explicit in ``Report.evidence``, so the surfaced result is unambiguous.

A fitted anomaly score on the dependency state is a forward-looking option (see the
roadmap): when an envelope has been learned, the auditor can score a snapshot against it
and fall back to the freshness rule otherwise. v0.1 surfaces the freshness rule; the
fitted path is not claimed as a shipping v0.1 capability.
"""
from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from typing import Any, Optional, Sequence

from ..record import Auditor, DependencySnapshot, Report

try:  # the forward-looking fitted path needs pyod + numpy, installed separately
    import numpy as np
    from pyod.models.ecod import ECOD

    _HAS_PYOD = True
except Exception:  # pragma: no cover - exercised only where pyod is absent
    _HAS_PYOD = False

_SCHEMA_VERSION = "v0.2"


@contextmanager
def _quiet_precision_loss():
    """Silence the benign scipy precision-loss RuntimeWarning ECOD raises on constant
    columns. A stable version or config is constant across the fit corpus by design, so
    its moment calculation underflows; the score still lands in the ECDF tail. Only that
    specific message is suppressed, so other RuntimeWarnings still surface.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation",
            category=RuntimeWarning,
        )
        yield


class DataAuditor(Auditor):
    """Score the dependency state a decision relied on; fall back to freshness."""

    stage = "data"

    def __init__(
        self,
        *,
        max_age_seconds: float = 86400.0,
        detector: Any = None,
        schema: Optional[dict] = None,
        name: str = "snapshot-freshness",
    ):
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive.")
        self.max_age_seconds = max_age_seconds
        self.name = name
        # detector: an unfitted PyOD-compatible detector to use in fit (default ECOD), or
        # an already-fitted detector when a schema is also supplied.
        self._detector = detector
        self._schema = schema
        self._fitted = detector is not None and schema is not None

    # ---- fit (learn the normal envelope) ----------------------------------------
    def fit(self, snapshots: Sequence[DependencySnapshot], *, now: Optional[float] = None) -> "DataAuditor":
        """Learn the dependency-state envelope from a corpus of normal snapshots."""
        if not _HAS_PYOD:
            raise ImportError(
                "DataAuditor.fit requires pyod and numpy: pip install pyod"
            )
        if not snapshots:
            raise ValueError("fit needs at least one snapshot.")
        now = time.time() if now is None else now
        states = [s.state for s in snapshots]
        ages = [self._age(s, now) for s in snapshots]
        self._schema = self._build_schema(states, ages)
        matrix = np.asarray(
            [self._encode_vector(st, ag) for st, ag in zip(states, ages)], dtype=float
        )
        detector = self._detector if (self._detector is not None and hasattr(self._detector, "fit")) else ECOD()
        with _quiet_precision_loss():
            detector.fit(matrix)
        self._schema["feature_names"] = self._feature_names()
        self._schema["mean"] = matrix.mean(axis=0).tolist()
        self._schema["std"] = (matrix.std(axis=0) + 1e-9).tolist()
        self._detector = detector
        self._fitted = True
        return self

    # ---- assess -----------------------------------------------------------------
    def assess(self, subject: DependencySnapshot, *, now: Optional[float] = None) -> Report:
        now = time.time() if now is None else now
        if self._fitted and _HAS_PYOD:
            try:
                return self._assess_learned(subject, now)
            except Exception as exc:  # a learned-path error degrades to freshness, explicitly
                return self._assess_freshness(
                    subject, now, reason_code=f"learned_error:{type(exc).__name__}"
                )
        return self._assess_freshness(subject, now, reason_code="freshness_rule")

    # ---- learned mode -----------------------------------------------------------
    def _assess_learned(self, subject: DependencySnapshot, now: float) -> Report:
        age = self._age(subject, now)
        vec = np.asarray([self._encode_vector(subject.state, age)], dtype=float)
        with _quiet_precision_loss():
            score = float(self._detector.predict_proba(vec)[:, 1][0])
            is_outlier = int(self._detector.predict(vec)[0]) == 1
        top = self._top_features(subject.state, age)
        reason = (
            f"Dependency state in the tail of the learned-normal envelope (top features: {', '.join(top)})."
            if is_outlier
            else "Dependency state within the learned-normal envelope."
        )
        return Report(
            self.stage,
            self.name,
            round(score, 3),
            "anomalous" if is_outlier else "ok",
            reason,
            {
                "mode": "learned",
                "detector": type(self._detector).__name__,
                "schema_version": _SCHEMA_VERSION,
                "normalized_risk": round(score, 3),
                "top_features": top,
            },
        )

    # ---- freshness fallback -----------------------------------------------------
    def _assess_freshness(self, subject: DependencySnapshot, now: float, *, reason_code: str) -> Report:
        captured = subject.captured_at
        if captured is None:
            return Report(
                self.stage, self.name, 0.0, "ok", "No capture time on snapshot.",
                {"mode": "freshness_fallback", "reason_code": reason_code},
            )
        age = max(0.0, now - captured)
        budget = self.max_age_seconds
        raw = age / budget
        score = min(1.0, raw)
        stale = age >= budget
        reason = (
            f"Snapshot age {age:,.0f}s is at or beyond the {budget:,.0f}s freshness budget."
            if stale
            else f"Snapshot age {age:,.0f}s within the {budget:,.0f}s freshness budget."
        )
        return Report(
            self.stage, self.name, round(score, 3), "stale" if stale else "ok", reason,
            {
                "mode": "freshness_fallback",
                "reason_code": reason_code,
                "age_seconds": round(age, 1),
                "max_age_seconds": budget,
                "raw_ratio": round(raw, 3),
            },
        )

    # ---- feature schema (the contribution) --------------------------------------
    @staticmethod
    def _age(snapshot: DependencySnapshot, now: float) -> float:
        return max(0.0, now - snapshot.captured_at) if snapshot.captured_at is not None else 0.0

    def _build_schema(self, states, ages) -> dict:
        """Numeric keys (lists encoded by length), categorical keys frequency-encoded."""
        n = len(states)
        numeric_keys = set()
        cat_counts: dict = {}
        for state in states:
            for key, value in state.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) or isinstance(value, (list, tuple)):
                    numeric_keys.add(key)
                elif isinstance(value, str):
                    cat_counts.setdefault(key, {})
                    cat_counts[key][value] = cat_counts[key].get(value, 0) + 1
        categorical = {
            key: {val: count / n for val, count in counts.items()}
            for key, counts in cat_counts.items()
        }
        return {
            "numeric_keys": sorted(numeric_keys),
            "categorical": categorical,
            "categorical_keys": sorted(categorical.keys()),
            "version": _SCHEMA_VERSION,
        }

    def _feature_names(self):
        return (
            list(self._schema["numeric_keys"])
            + [f"{key}__rarity" for key in self._schema["categorical_keys"]]
            + ["snapshot_age"]
        )

    def _encode_vector(self, state: dict, age: float):
        vec = []
        for key in self._schema["numeric_keys"]:
            value = state.get(key)
            if isinstance(value, (list, tuple)):
                vec.append(float(len(value)))
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                vec.append(0.0)
            else:
                vec.append(float(value))
        for key in self._schema["categorical_keys"]:
            value = state.get(key)
            freq = self._schema["categorical"][key].get(value, 0.0) if isinstance(value, str) else 0.0
            vec.append(1.0 - freq)  # rarity: an unseen category encodes to 1.0
        vec.append(float(age))  # freshness as one feature among several
        return vec

    def _top_features(self, state: dict, age: float, k: int = 2):
        vec = np.asarray(self._encode_vector(state, age), dtype=float)
        mean = np.asarray(self._schema["mean"], dtype=float)
        std = np.asarray(self._schema["std"], dtype=float)
        z = np.abs((vec - mean) / std)
        names = self._schema["feature_names"]
        order = list(np.argsort(z))[::-1][:k]
        return [names[i] for i in order]
