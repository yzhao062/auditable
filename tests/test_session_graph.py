"""Tests for the typed SessionGraph schema (the v0.3 stable foundation).

These pin the seams that keep v0.3b (live scoring, own-record observed edges)
additive: per-edge grade, per-edge evidence, an optional resource identity, node
attributes (model_id / decision_basis), explicit completeness, and coverage / rho.
"""
import math

import pytest

from auditable.graph.session import (
    DependencyEdge,
    EdgeCoverage,
    Grade,
    GraphCompleteness,
    ResourceRef,
    SessionGraph,
    Step,
)


def _step(idx, agent="m", kind="decision", deps=None, **attrs):
    return Step(idx=idx, agent=agent, kind=kind, deps=list(deps or []), node_attrs=dict(attrs))


def test_dependency_edge_carries_grade_resource_evidence():
    e = DependencyEdge(
        src_idx=0,
        grade=Grade.OBSERVED,
        resource=ResourceRef("db", "orders", "row_7"),
        evidence={"matched_on": "row_7"},
    )
    assert e.src_idx == 0
    assert e.grade is Grade.OBSERVED
    assert e.resource.namespace == "db" and e.resource.key == "row_7"
    assert e.evidence["matched_on"] == "row_7"


def test_edge_defaults_are_conservative():
    # an edge whose source is unknown should default to the weakest grade and no
    # resource identity, so nothing is silently presented as observed
    e = DependencyEdge(src_idx=1)
    assert e.grade is Grade.INFERRED
    assert e.resource is None
    assert e.evidence == {}


def test_step_carries_model_node_attrs():
    s = _step(2, model_id="gpt-x", decision_basis="invoice matches an approved PO")
    assert s.node_attrs["model_id"] == "gpt-x"
    assert s.node_attrs["decision_basis"] == "invoice matches an approved PO"


def test_exec_preds_default_root_and_merge():
    # None = offline linear default (previous step); [] = root with no predecessor
    # (distinct from None for parallel roots); [0, 1] = a merge of two branches
    assert Step(idx=0, agent="m", kind="decision").exec_preds is None
    root = Step(idx=0, agent="m", kind="decision", exec_preds=[])
    assert root.exec_preds == []
    merge = Step(idx=2, agent="m", kind="decision", exec_preds=[0, 1])
    assert merge.exec_preds == [0, 1]


def test_step_carries_live_correlation_and_final_record_id():
    # correlation_id is the pre-digest live assembly id; record_id is the final
    # signed digest, carried separately (a record_id is too late for live assembly)
    s = Step(idx=0, agent="m", kind="decision", correlation_id="evt-abc", record_id="sha:deadbeef")
    assert s.correlation_id == "evt-abc"
    assert s.record_id == "sha:deadbeef"
    # both optional, default None for the minimal offline case
    bare = Step(idx=1, agent="m", kind="decision")
    assert bare.correlation_id is None and bare.record_id is None


def test_from_steps_defaults_complete():
    g = SessionGraph.from_steps([_step(0), _step(1, deps=[DependencyEdge(0)])])
    assert len(g.steps) == 2
    assert g.completeness is GraphCompleteness.COMPLETE


def test_add_step_reserved_but_works():
    # the live (v0.3b) entry point exists now so live scoring is additive
    g = SessionGraph.from_steps([_step(0)])
    g.add_step(_step(1, deps=[DependencyEdge(0)]))
    assert len(g.steps) == 2


def test_coverage_counts_mixed_grades_and_rho():
    # the key seam: observed, declared, and inferred edges coexist in one graph
    steps = [
        _step(0),
        _step(1, deps=[DependencyEdge(0, grade=Grade.OBSERVED)]),
        _step(
            2,
            deps=[
                DependencyEdge(0, grade=Grade.INFERRED),
                DependencyEdge(1, grade=Grade.OBSERVED),
            ],
        ),
    ]
    cov = SessionGraph.from_steps(steps).coverage()
    assert isinstance(cov, EdgeCoverage)
    assert cov.n_dep_edges == 3
    assert cov.by_grade[Grade.OBSERVED] == 2
    assert cov.by_grade[Grade.INFERRED] == 1
    assert cov.by_grade[Grade.DECLARED] == 0
    # rho = |E_dep| / C(n_steps, 2) = 3 / C(3, 2) = 3 / 3 = 1.0
    assert math.isclose(cov.rho, 1.0)
    assert math.isclose(cov.observed_fraction, 2 / 3)


def test_coverage_zero_for_single_step():
    cov = SessionGraph.from_steps([_step(0)]).coverage()
    assert cov.n_dep_edges == 0
    assert cov.rho == 0.0
    assert cov.observed_fraction == 0.0


# --- identity validation -----------------------------------------------------
# Duplicate idx collapses in the step::<idx> projection but is still counted by
# len(steps), so the size-normalized risk denominator disagrees with the graph.
# Reject malformed identity at construction so the projection and the score agree.


def test_duplicate_idx_rejected_in_from_steps():
    with pytest.raises(ValueError):
        SessionGraph.from_steps([_step(0), _step(1), _step(0)])


def test_duplicate_idx_rejected_in_add_step():
    g = SessionGraph.from_steps([_step(0)])
    with pytest.raises(ValueError):
        g.add_step(_step(0))


def test_non_int_idx_rejected():
    with pytest.raises(ValueError):
        SessionGraph.from_steps([Step(idx="0", agent="m", kind="decision")])


def test_bool_idx_rejected():
    # bool is an int subclass; True == 1 would alias step 1's projection node.
    with pytest.raises(ValueError):
        SessionGraph.from_steps([Step(idx=True, agent="m", kind="decision")])


def test_non_int_dependency_src_rejected():
    bad = Step(idx=1, agent="m", kind="decision", deps=[DependencyEdge("0")])
    with pytest.raises(ValueError):
        SessionGraph.from_steps([_step(0), bad])


def test_invalid_kind_rejected():
    # the kernel scores only decision / tool_call nodes; an unrecognized kind would be
    # counted by the size-normalized risk denominator yet dropped by the feature layer.
    with pytest.raises(ValueError):
        SessionGraph.from_steps([Step(idx=0, agent="m", kind="custom")])
