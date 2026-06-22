"""Tests for the PRE Preflight Coverage Report (descriptive, NOT a risk score).

Covers the three coverage-readiness views added to :mod:`auditable.graph.pre`:

- **preflight_coverage**: the existing ``coverage()`` model surfaced over the
  DECLARED graph (grade mix, ``observed_fraction``, ``rho``) plus the exact
  ``no_score:*`` reason ``structural_risk`` would apply -- ``low_coverage`` on a
  multi-step declared plan, ``single_decision`` on a 0/1-step plan -- and never a
  number;
- **resource_touch_completeness**: which declared reads, writes, and dependency
  edges carry a resource identity, exercised on both an adapter-built plan (ids
  present, every declared edge missing the structured ``ResourceRef``) and a
  hand-built graph that plants a read / write / edge with NO id;
- **barrier_inventory**: the declared revalidation re-reads grouped per resource,
  with and without barriers.

The State-B withhold boundary is re-pinned here: adding the report changes nothing
about ``state_b_risk`` / ``state_b_withheld``.
"""
import pytest

pytest.importorskip("networkx")  # PRE builds on the NetworkX projection

from auditable.graph.adapters.declared_plan import declared_plan_v1
from auditable.graph.pre import (
    BarrierInventory,
    PreflightCoverage,
    PreReport,
    ResourceGap,
    ResourceTouchCompleteness,
    analyze_plan,
    barrier_inventory,
    resource_touch_completeness,
)
from auditable.graph.risk import (
    STATE_LOW_COVERAGE,
    STATE_SINGLE_DECISION,
    structural_risk,
)
from auditable.graph.session import (
    DependencyEdge,
    Grade,
    ResourceRef,
    SessionGraph,
    Step,
)


def _graph(plan):
    return SessionGraph.from_steps(declared_plan_v1.to_steps(plan)).to_networkx()


def _crafted_plan():
    """A plan with present resource ids, a volatile read, and one barrier.

    Node 0 writes ``policy.v``; node 1 volatile-reads it (no barrier) and writes
    ``snap``; node 2 re-reads ``policy.v`` as a revalidation barrier and reads
    ``snap``, then writes ``order.1``. Every read names a prior producer, so each
    wires a DECLARED dependency edge -- and every such edge carries an
    ``evidence['resource_id']`` but a ``None`` structured ``resource``.
    """
    return {
        "nodes": [
            {"idx": 0, "agent": "pol", "kind": "tool_call", "writes": ["policy.v"]},
            {"idx": 1, "agent": "snap", "kind": "tool_call",
             "reads": [{"id": "policy.v", "producer": 0, "volatile": True}],
             "writes": ["snap"]},
            {"idx": 2, "agent": "exec", "kind": "tool_call",
             "reads": [{"id": "policy.v", "producer": 0, "volatile": True, "revalidates": True},
                       {"id": "snap", "producer": 1}],
             "writes": ["order.1"]},
        ]
    }


# --- preflight_coverage ------------------------------------------------------


def test_preflight_coverage_surfaces_grade_mix_and_low_coverage_reason():
    rep = analyze_plan(_crafted_plan())
    pc = rep.preflight_coverage
    assert isinstance(pc, PreflightCoverage)
    # grade mix: every dependency edge is DECLARED at PRE, none observed / inferred
    assert pc.n_steps == 3
    assert pc.n_dep_edges == 3 and pc.declared == 3
    assert pc.observed == 0 and pc.inferred == 0
    assert pc.observed_fraction == 0.0
    # the exact runtime no-score reason a multi-step declared plan triggers
    assert STATE_LOW_COVERAGE in pc.no_score_reason
    assert pc.would_score is False
    # it mirrors what structural_risk actually returns on this graph
    state = structural_risk(SessionGraph.from_steps(declared_plan_v1.to_steps(_crafted_plan()))).state
    assert state == STATE_LOW_COVERAGE
    assert pc.rho == structural_risk(
        SessionGraph.from_steps(declared_plan_v1.to_steps(_crafted_plan()))
    ).coverage.rho


@pytest.mark.parametrize(
    "plan,n_steps",
    [
        ({"nodes": []}, 0),
        ({"nodes": [{"idx": 0, "agent": "a", "kind": "decision"}]}, 1),
    ],
)
def test_preflight_coverage_single_decision_reason_on_small_plan(plan, n_steps):
    # a 0/1-step plan is gated as single_decision, not low_coverage; the preflight
    # report must surface that exact reason (and still no number)
    rep = analyze_plan(plan)
    pc = rep.preflight_coverage
    assert pc is not None and pc.n_steps == n_steps
    assert STATE_SINGLE_DECISION in pc.no_score_reason
    assert pc.would_score is False
    # confirm the gate this maps to
    steps = declared_plan_v1.to_steps(plan)
    assert structural_risk(SessionGraph.from_steps(steps)).state == STATE_SINGLE_DECISION


def test_preflight_coverage_is_not_a_risk_number():
    rep = analyze_plan(_crafted_plan())
    # the descriptive report never carries a State-B score; the boundary holds
    assert rep.state_b_risk is None and rep.state_b_withheld is True
    assert rep.preflight_coverage.would_score is False


# --- resource_touch_completeness ---------------------------------------------


def test_resource_touch_completeness_all_ids_present_edges_missing_structured_ref():
    # the declared adapter always names reads / writes and records evidence
    # resource_id, but leaves the structured ResourceRef None (declared convention)
    rtc = resource_touch_completeness(_graph(_crafted_plan()))
    assert isinstance(rtc, ResourceTouchCompleteness)
    # reads: policy.v @1, policy.v + snap @2  -> 3 reads, all identified
    assert rtc.n_reads == 3 and rtc.reads_with_id == 3
    # writes: policy.v @0, snap @1, order.1 @2 -> 3 writes, all identified
    assert rtc.n_writes == 3 and rtc.writes_with_id == 3
    # 3 declared dependency edges, each with an evidence resource_id (so "identified")
    assert rtc.n_edges == 3 and rtc.edges_with_id == 3
    # ... but every one is missing the structured ResourceRef the runtime fills
    assert rtc.edges_missing_structured_resource == 3
    # the only gaps reported are the missing-structured-ResourceRef edges
    assert all(g.kind == "edge" for g in rtc.gaps)
    assert all("ResourceRef" in g.detail for g in rtc.gaps)
    assert {(g.node_idx, g.src_idx) for g in rtc.gaps} == {(1, 0), (2, 0), (2, 1)}


def test_resource_touch_completeness_reports_missing_read_write_and_edge_ids():
    # craft a graph DIRECTLY (bypassing the adapter) to plant the missing-id paths:
    # a read with an empty id, a write that is None, and a dependency edge that
    # carries neither a structured resource nor an evidence resource_id.
    steps = [
        Step(idx=0, agent="t", kind="tool_call",
             node_attrs={"reads": [], "writes": ["res.a"]}),
        Step(
            idx=1,
            agent="d",
            kind="decision",
            node_attrs={"reads": [""], "writes": [None]},
            deps=[DependencyEdge(src_idx=0, grade=Grade.DECLARED, resource=None, evidence={})],
        ),
    ]
    G = SessionGraph.from_steps(steps).to_networkx()
    rtc = resource_touch_completeness(G)
    # node 1 has one empty read and one None write -> both unidentified
    assert rtc.n_reads == 1 and rtc.reads_with_id == 0
    assert rtc.n_writes == 2 and rtc.writes_with_id == 1  # only res.a @0 is named
    # the one edge has no resource id of any kind -> unidentified and missing structured
    assert rtc.n_edges == 1 and rtc.edges_with_id == 0
    assert rtc.edges_missing_structured_resource == 1
    kinds = sorted(g.kind for g in rtc.gaps)
    assert kinds == ["edge", "read", "write"]
    # the read / write gaps point at node 1; the edge gap is 1->0
    read_gap = next(g for g in rtc.gaps if g.kind == "read")
    write_gap = next(g for g in rtc.gaps if g.kind == "write")
    edge_gap = next(g for g in rtc.gaps if g.kind == "edge")
    assert read_gap.node_idx == 1 and write_gap.node_idx == 1
    assert edge_gap.node_idx == 1 and edge_gap.src_idx == 0
    assert isinstance(read_gap, ResourceGap)


def test_resource_touch_completeness_structured_resourceref_counts_as_identified():
    # an OBSERVED-style edge carrying a structured ResourceRef is fully identified:
    # it does NOT count toward edges_missing_structured_resource and raises no gap.
    steps = [
        Step(idx=0, agent="t", kind="tool_call", node_attrs={"reads": [], "writes": ["res.a"]}),
        Step(
            idx=1,
            agent="t",
            kind="tool_call",
            node_attrs={"reads": ["res.a"], "writes": []},
            deps=[
                DependencyEdge(
                    src_idx=0,
                    grade=Grade.OBSERVED,
                    resource=ResourceRef(namespace="ns", resource_id="res.a", key="k"),
                    evidence={"resource_id": "res.a"},
                )
            ],
        ),
    ]
    rtc = resource_touch_completeness(SessionGraph.from_steps(steps).to_networkx())
    assert rtc.n_edges == 1 and rtc.edges_with_id == 1
    assert rtc.edges_missing_structured_resource == 0
    assert rtc.gaps == []


# --- barrier_inventory -------------------------------------------------------


def test_barrier_inventory_lists_declared_barriers_per_resource():
    # the crafted plan declares one revalidation barrier: node 2 re-reads policy.v
    bi = barrier_inventory(_graph(_crafted_plan()))
    assert isinstance(bi, BarrierInventory)
    assert bi.by_resource == {"policy.v": [2]}
    assert bi.barrier_nodes == [2]
    assert bi.resources_with_barrier == ["policy.v"]


def test_barrier_inventory_empty_when_no_barriers():
    # same shape but the re-read at node 2 is NOT flagged revalidates -> no barrier
    plan = {
        "nodes": [
            {"idx": 0, "agent": "pol", "kind": "tool_call", "writes": ["policy.v"]},
            {"idx": 1, "agent": "snap", "kind": "tool_call",
             "reads": [{"id": "policy.v", "producer": 0, "volatile": True}], "writes": ["snap"]},
            {"idx": 2, "agent": "exec", "kind": "tool_call",
             "reads": [{"id": "policy.v", "producer": 0, "volatile": True},
                       {"id": "snap", "producer": 1}], "writes": ["order.1"]},
        ]
    }
    bi = barrier_inventory(_graph(plan))
    assert bi.by_resource == {}
    assert bi.barrier_nodes == []
    assert bi.resources_with_barrier == []


def test_barrier_inventory_groups_multiple_nodes_and_resources():
    # two resources, two barrier nodes: node 1 revalidates r.a, node 3 revalidates
    # both r.a and r.b -> by_resource groups idxs per resource, sorted.
    plan = {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["r.a"]},
            {"idx": 1, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "r.a", "producer": 0, "revalidates": True}], "writes": ["r.b"]},
            {"idx": 2, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "r.b", "producer": 1}], "writes": ["x"]},
            {"idx": 3, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "r.a", "producer": 0, "revalidates": True},
                       {"id": "r.b", "producer": 1, "revalidates": True}], "writes": ["y"]},
        ]
    }
    bi = barrier_inventory(_graph(plan))
    assert bi.by_resource == {"r.a": [1, 3], "r.b": [3]}
    assert bi.barrier_nodes == [1, 3]
    assert bi.resources_with_barrier == ["r.a", "r.b"]


# --- integration: the report rides on PreReport, boundary intact -------------


def test_analyze_plan_attaches_full_preflight_report():
    rep = analyze_plan(_crafted_plan())
    assert isinstance(rep, PreReport)
    assert isinstance(rep.preflight_coverage, PreflightCoverage)
    assert isinstance(rep.resource_touch_completeness, ResourceTouchCompleteness)
    assert isinstance(rep.barrier_inventory, BarrierInventory)
    # the three views agree with the standalone builders on the same plan
    G = _graph(_crafted_plan())
    assert rep.resource_touch_completeness.edges_missing_structured_resource == \
        resource_touch_completeness(G).edges_missing_structured_resource
    assert rep.barrier_inventory.by_resource == barrier_inventory(G).by_resource
    # the State-B boundary is unchanged by the additive report
    assert rep.state_b_risk is None and rep.state_b_withheld is True


def test_analyze_plan_summary_renders_the_preflight_report():
    text = analyze_plan(_crafted_plan()).summary()
    # the descriptive coverage report appears, labeled NOT a risk score
    assert "preflight coverage (descriptive, NOT a risk score)" in text
    assert "grade mix: observed=0 declared=3" in text
    assert STATE_LOW_COVERAGE in text
    # resource-touch completeness and the barrier inventory both render
    assert "resource-touch completeness" in text
    assert "missing the structured" in text
    assert "barrier inventory" in text
    assert "'policy.v': step 2" in text
    # the withhold boundary line is still present and unchanged
    assert "State B (dependency-state) blast-share risk: WITHHELD" in text


def test_preflight_report_adds_no_top_level_export():
    import auditable

    for name in (
        "PreflightCoverage",
        "ResourceTouchCompleteness",
        "BarrierInventory",
        "resource_touch_completeness",
        "barrier_inventory",
    ):
        assert not hasattr(auditable, name), f"{name} must not leak to the top level"
