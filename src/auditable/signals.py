"""Thin v0.1 signals for the data and model layers.

These are deliberately small first cuts, wired so the full chain is visible in one
record. v0.2 replaces ``score_snapshot_freshness`` with a learned, PyOD-backed anomaly
model on the dependency state; v0.3 replaces ``score_model_trust`` with TrustLLM trust
signals on the deciding model. Neither is the finished method yet; each exists so the
record binds all three layers from v0.1.
"""
from __future__ import annotations

from .record import DataSignal, DependencySnapshot, ModelSignal, ModelSpan


def score_snapshot_freshness(
    snapshot: DependencySnapshot,
    *,
    now: float,
    max_age_seconds: float,
) -> DataSignal:
    """Flag a dependency snapshot that is older than the freshness budget.

    Thin v0.1 rule: the snapshot is stale when its age exceeds ``max_age_seconds``. The
    anomaly score is the age in budget-multiples (0 when fresh, 1 at the budget, above 1
    past it). v0.2 swaps this for a learned anomaly model over the snapshot state.
    """
    captured = snapshot.captured_at
    if captured is None:
        return DataSignal(anomaly_score=0.0, stale=False, reason="No capture time on snapshot.")
    age = max(0.0, now - captured)
    score = age / max_age_seconds if max_age_seconds > 0 else 0.0
    stale = age > max_age_seconds
    reason = (
        f"Snapshot age {age:,.0f}s exceeds the {max_age_seconds:,.0f}s freshness budget."
        if stale
        else f"Snapshot age {age:,.0f}s within the {max_age_seconds:,.0f}s freshness budget."
    )
    return DataSignal(anomaly_score=round(score, 3), stale=stale, reason=reason)


def score_model_trust(model: ModelSpan) -> ModelSignal:
    """A thin trust flag on the deciding model.

    v0.1 heuristic: a stated decision basis raises confidence; its absence lowers it.
    v0.3 replaces this with TrustLLM trust signals on the model and its output.
    """
    if model.decision_basis.strip():
        return ModelSignal(trust=0.9, flag="ok", reason="Model stated a decision basis.")
    return ModelSignal(trust=0.5, flag="low_trust", reason="Model stated no decision basis.")
