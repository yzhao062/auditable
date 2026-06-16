"""The data module: audit the data a decision relied on.

v0.1 ships ``DataAuditor`` with a thin freshness rule on the dependency snapshot. v0.2
replaces the method body with a learned, PyOD-backed anomaly model over the snapshot
state; the ``Auditor`` interface and the ``Report`` it returns stay the same.
"""
from __future__ import annotations

import time
from typing import Optional

from ..record import Auditor, DependencySnapshot, Report


class DataAuditor(Auditor):
    """Flag a dependency snapshot that is older than the freshness budget (v0.1).

    Thin rule: the snapshot is stale when its age exceeds ``max_age_seconds``; the score
    is the age in budget-multiples, clipped to [0, 1], with the raw ratio kept in
    evidence. v0.2 swaps this body for a learned anomaly model over the snapshot state.
    """

    stage = "data"

    def __init__(self, *, max_age_seconds: float, name: str = "snapshot-freshness"):
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive.")
        self.max_age_seconds = max_age_seconds
        self.name = name

    def assess(self, subject: DependencySnapshot, *, now: Optional[float] = None) -> Report:
        now = time.time() if now is None else now
        captured = subject.captured_at
        if captured is None:
            return Report(self.stage, self.name, 0.0, "ok", "No capture time on snapshot.")
        age = max(0.0, now - captured)
        budget = self.max_age_seconds
        raw = age / budget
        score = min(1.0, raw)
        # At exactly the budget the snapshot is stale, so score 1.0 lines up with the flag.
        stale = age >= budget
        reason = (
            f"Snapshot age {age:,.0f}s is at or beyond the {budget:,.0f}s freshness budget."
            if stale
            else f"Snapshot age {age:,.0f}s within the {budget:,.0f}s freshness budget."
        )
        return Report(
            self.stage,
            self.name,
            round(score, 3),
            "stale" if stale else "ok",
            reason,
            {"age_seconds": round(age, 1), "max_age_seconds": budget, "raw_ratio": round(raw, 3)},
        )
