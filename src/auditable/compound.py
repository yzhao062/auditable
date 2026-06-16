"""The compound view over a decision's three local reports.

v0.1 ships ``CompoundReport`` as a transparent bundle: it preserves every per-stage
report and exposes an explicitly named, uncalibrated debug aggregate. It does not claim a
calibrated global risk and it does not drive the gate; control is driven by the replay
verdict. The calibrated combiner (``CompoundRisk.combine``, outlier-ensemble methods over
calibrated per-stage scores, with replay divergence as a high-weight axis) arrives in
v0.2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from .record import _sha256


@dataclass
class CompoundReport:
    """A transparent bundle over the per-stage leaves (v0.1).

    ``reports`` preserves the per-stage breakdown, because an auditor usually needs to
    know which stage flagged a decision. ``uncalibrated_score`` is an explicit debug
    aggregate (the max of per-stage scores), not decision-grade and not used by the gate.
    v0.2 replaces it with a calibrated combined risk.
    """

    reports: list = field(default_factory=list)
    uncalibrated_score: Optional[float] = None

    @classmethod
    def of(cls, reports, *, score: bool = True) -> "CompoundReport":
        kept = [r for r in reports if r is not None]
        agg = max((r.score for r in kept), default=0.0) if score else None
        return cls(reports=kept, uncalibrated_score=agg)

    def by_stage(self) -> dict:
        """Map stage name to its report, for callers that want the breakdown."""
        return {r.stage: r for r in self.reports}

    def digest(self) -> str:
        return _sha256(asdict(self))
