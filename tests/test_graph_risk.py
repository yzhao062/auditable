"""Tests for structural_risk: the scored case and both no-score gate states."""
import math

import pytest

pytest.importorskip("networkx")  # scoring builds on the NetworkX projection

from auditable.graph.risk import (
    STATE_LOW_COVERAGE,
    STATE_SCORED,
    STATE_SINGLE_DECISION,
    RiskResult,
    structural_risk,
)
from auditable.graph.session import (
    DependencyEdge,
    Grade,
    GraphCompleteness,
    SessionGraph,
    Step,
)


def _obs(src):
    return DependencyEdge(src, grade=Grade.OBSERVED)


def _inf(src):
    return DependencyEdge(src, grade=Grade.INFERRED)


def test_scored_ranks_the_keystone_decision():
    # step 0 is a read that 1 and 2 observably rely on; step 3 relies on step 1.
    # Transitively, everything is downstream of step 0, so it is the keystone.
    steps = [
        Step(idx=0, agent="env", kind="tool_call"),
        Step(idx=1, agent="m", kind="decision", deps=[_obs(0)]),
        Step(idx=2, agent="m", kind="decision", deps=[_obs(0)]),
        Step(idx=3, agent="m", kind="decision", deps=[_obs(1)]),
    ]
    r = structural_risk(SessionGraph.from_steps(steps))
    assert isinstance(r, RiskResult)
    assert r.state == STATE_SCORED
    assert r.completeness is GraphCompleteness.COMPLETE
    # normalized transitive blast share; step 0 is the worst keystone
    assert math.isclose(r.per_decision[0], 1.0)    # all three others are downstream
    assert math.isclose(r.per_decision[1], 1 / 3)  # only step 3 is downstream of 1
    assert r.per_decision[2] == 0.0 and r.per_decision[3] == 0.0
    assert math.isclose(r.per_session, 1.0)
    assert r.per_decision[0] > r.per_decision[1] > r.per_decision[2]
    # the score stands because coverage is observed and unsaturated
    assert r.coverage.observed_fraction == 1.0
    assert r.coverage.rho < 0.9
    # the raw layered features are reported alongside the risk
    assert set(r.features) == {"flat", "exec", "dep"}


def test_no_score_single_decision():
    r = structural_risk(SessionGraph.from_steps([Step(idx=0, agent="m", kind="decision")]))
    assert r.state == STATE_SINGLE_DECISION
    assert r.per_session is None
    assert r.per_decision == {}
    # coverage and completeness are still reported in the no-score state
    assert r.coverage.n_dep_edges == 0
    assert r.completeness is GraphCompleteness.COMPLETE


def test_no_score_low_coverage_inferred_full_context():
    # every step depends on every prior step, all inferred: rho saturates to 1 and
    # the observed fraction is 0 -- the degenerate regime features.py warns about
    steps = [
        Step(idx=0, agent="m", kind="decision"),
        Step(idx=1, agent="m", kind="decision", deps=[_inf(0)]),
        Step(idx=2, agent="m", kind="decision", deps=[_inf(0), _inf(1)]),
        Step(idx=3, agent="m", kind="decision", deps=[_inf(0), _inf(1), _inf(2)]),
    ]
    r = structural_risk(SessionGraph.from_steps(steps))
    assert r.state == STATE_LOW_COVERAGE
    assert r.per_session is None
    assert r.per_decision == {}  # dependency structure is NOT presented as risk
    assert r.coverage.observed_fraction == 0.0
    assert math.isclose(r.coverage.rho, 1.0)


def test_no_score_low_coverage_saturated_even_when_observed():
    # full context but every edge marked observed: the observed-fraction gate
    # passes, yet the rho gate still fires, so a saturated dependency layer is not
    # scored even when it is "observed"
    steps = [
        Step(idx=0, agent="m", kind="decision"),
        Step(idx=1, agent="m", kind="decision", deps=[_obs(0)]),
        Step(idx=2, agent="m", kind="decision", deps=[_obs(0), _obs(1)]),
        Step(idx=3, agent="m", kind="decision", deps=[_obs(0), _obs(1), _obs(2)]),
    ]
    r = structural_risk(SessionGraph.from_steps(steps))
    assert r.state == STATE_LOW_COVERAGE
    assert r.per_session is None
    assert r.coverage.observed_fraction == 1.0   # observed...
    assert math.isclose(r.coverage.rho, 1.0)     # ...but saturated, so still gated


def test_no_score_low_coverage_when_dependency_layer_empty():
    # multiple steps but no dependency edges at all: nothing structural to score
    steps = [Step(idx=i, agent="m", kind="decision") for i in range(3)]
    r = structural_risk(SessionGraph.from_steps(steps))
    assert r.state == STATE_LOW_COVERAGE
    assert r.per_decision == {}
    assert r.coverage.n_dep_edges == 0


def test_raw_features_present_even_when_gated():
    # the raw layered_features stay available for inspection in a no-score state
    steps = [
        Step(idx=0, agent="m", kind="decision"),
        Step(idx=1, agent="m", kind="decision", deps=[_inf(0)]),
    ]
    r = structural_risk(SessionGraph.from_steps(steps))
    assert r.state == STATE_LOW_COVERAGE
    assert set(r.features) == {"flat", "exec", "dep"}
    assert r.features["flat"]["n_steps"] == 2


def test_thresholds_are_tunable_keyword_overrides():
    # a sparse all-observed graph that scores by default can be forced into the
    # low-coverage gate by raising the observed-fraction bar; the signature stays
    # positional-compatible
    steps = [
        Step(idx=0, agent="env", kind="tool_call"),
        Step(idx=1, agent="m", kind="decision", deps=[_obs(0)]),
        Step(idx=2, agent="m", kind="decision", deps=[_obs(0)]),
    ]
    g = SessionGraph.from_steps(steps)
    assert structural_risk(g).state == STATE_SCORED
    assert structural_risk(g, min_observed_fraction=1.01).state == STATE_LOW_COVERAGE
