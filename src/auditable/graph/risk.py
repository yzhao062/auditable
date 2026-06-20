"""Structural risk over a SessionGraph: the cross-layer triage signal.

Computes the paper's layered structural features over the typed SessionGraph
projection and maps the dependency-layer blast structure to a normalized triage
risk per decision and per session. This is the honest combiner the v0.3 plan
calls for: one structural score over the unified graph, not an outlier ensemble
of three scalars. The score is a normalized ranking signal, not a calibrated
probability; calibration waits for labeled data.

The per-decision risk is a step's normalized transitive blast radius (the share
of the rest of the session that, transitively, depends on it). A keystone
decision that many later steps rest on scores high, because a fault there
propagates widely; this is the keystone signal the failure-detection study
validated. The per-session risk is the worst keystone's blast share.

Two gates keep the score honest:

- A single-decision session has no cross-decision structure, so it returns
  ``no_score:single_decision`` and degrades to the per-decision signals (the v0.2
  data anomaly plus grounding) rather than inventing structure.
- A degenerate dependency layer (edges mostly inferred, or saturated toward the
  full-history regime where ``rho`` is near 1) returns ``no_score:low_coverage``
  and does NOT present dependency structure as risk. This is the regime
  ``features.py`` warns about: ``dep_depth`` and ``n_dep_edges`` become functions
  of run size and add nothing beyond the flat count. An empty dependency layer
  (no observed edges at all) is low coverage by the same rule.

The raw ``layered_features`` are returned in every state, so a caller can see the
descriptive structure even when the risk is withheld.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .session import EdgeCoverage, GraphCompleteness, SessionGraph

# Default gate thresholds. An observed fraction below the minimum, or a
# saturation ratio at or above the maximum, marks the dependency layer too
# degenerate to score. They are keyword-only overrides on the scorer (additive),
# so a caller can tune them without changing the positional signature.
_MIN_OBSERVED_FRACTION = 0.5
_MAX_RHO = 0.9

STATE_SCORED = "scored"
STATE_SINGLE_DECISION = "no_score:single_decision"
STATE_LOW_COVERAGE = "no_score:low_coverage"


@dataclass
class RiskResult:
    """The structural-risk verdict for one session graph.

    - ``state``: ``scored`` / ``no_score:single_decision`` / ``no_score:low_coverage``.
    - ``per_session``: the worst keystone's normalized blast share, or ``None`` when
      the graph is not scored.
    - ``per_decision``: each step ``idx`` mapped to its normalized transitive blast
      radius; empty when the graph is not scored.
    - ``coverage``: the dependency-edge coverage and saturation ratio from the graph.
    - ``completeness``: whether the graph is a complete run or a live prefix.
    - ``features``: the raw ``layered_features`` (descriptive; present in every state
      so a gated result still shows why it was withheld).
    """

    state: str
    per_session: Optional[float]
    per_decision: Dict[int, float]
    coverage: EdgeCoverage
    completeness: GraphCompleteness
    features: Dict[str, Dict[str, float]] = field(default_factory=dict)


def structural_risk(
    graph: SessionGraph,
    *,
    min_observed_fraction: float = _MIN_OBSERVED_FRACTION,
    max_rho: float = _MAX_RHO,
) -> RiskResult:
    """Score one :class:`SessionGraph` structurally.

    Builds the typed NetworkX projection, reads the layered structural features
    over it, and maps the dependency-layer transitive blast to a normalized risk
    per decision and per session. See the module docstring for the two no-score
    gates. The score is a ranking / triage signal, not calibrated.
    """
    from auditable.graph import downstream_reach, layered_features

    coverage = graph.coverage()
    completeness = graph.completeness
    G = graph.to_networkx()
    features = layered_features(G)
    n_steps = len(graph.steps)

    # Gate 1: a single decision has no cross-decision structure to score.
    if n_steps < 2:
        return RiskResult(
            state=STATE_SINGLE_DECISION,
            per_session=None,
            per_decision={},
            coverage=coverage,
            completeness=completeness,
            features=features,
        )

    # Gate 2: degenerate dependency layer (mostly inferred, or rho saturated toward
    # the full-history regime). Do not present dependency structure as risk.
    if coverage.observed_fraction < min_observed_fraction or coverage.rho >= max_rho:
        return RiskResult(
            state=STATE_LOW_COVERAGE,
            per_session=None,
            per_decision={},
            coverage=coverage,
            completeness=completeness,
            features=features,
        )

    # Per-decision: normalized transitive blast radius (size-normalized keystone
    # exposure). Per-session: the worst keystone's blast share.
    norm = n_steps - 1
    per_decision = {s.idx: downstream_reach(G, s.idx) / norm for s in graph.steps}
    per_session = max(per_decision.values()) if per_decision else 0.0
    return RiskResult(
        state=STATE_SCORED,
        per_session=per_session,
        per_decision=per_decision,
        coverage=coverage,
        completeness=completeness,
        features=features,
    )
