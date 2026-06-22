"""Tests for auditable.graph.pre: the PRE declared-plan entry.

Five parts, matching the contract:

- **adapter**: ``DeclaredPlanAdapter`` conforms to the ``Adapter`` protocol, is
  callable, has ``id == 'declared_plan_v1'``, emits only ``Grade.DECLARED``
  dependency edges, and rejects malformed plans (bad kind, duplicate / bool idx);
- **lints**: each of the four pure NetworkX lints fires on a planted positive and
  stays silent on the matching negative;
- **analyze_plan**: returns a ``PreReport`` with the execution-topology keystone,
  the planted findings, and State-B risk explicitly WITHHELD;
- **boundary**: a direct ``structural_risk(...).state == STATE_LOW_COVERAGE``
  assertion pins the no-score boundary, and the keystone is labeled structural,
  not the POST blast-radius keystone;
- **no leak**: PRE adds nothing to the top-level ``auditable`` surface.
"""
import pytest

pytest.importorskip("networkx")  # PRE builds on the NetworkX projection

from auditable.graph import pre
from auditable.graph.adapters import Adapter
from auditable.graph.adapters.declared_plan import DeclaredPlanAdapter, declared_plan_v1
from auditable.graph.pre import (
    LintFinding,
    PreReport,
    analyze_plan,
    flippable_dependency_annotations,
    missing_revalidation_barrier,
    scope_vs_snapshot,
    write_with_no_prior_read,
)
from auditable.graph.risk import STATE_LOW_COVERAGE, structural_risk
from auditable.graph.session import Grade, SessionGraph, Step


def _graph(plan):
    return SessionGraph.from_steps(declared_plan_v1.to_steps(plan)).to_networkx()


# --- the DeclaredPlanAdapter -------------------------------------------------


def test_declared_plan_adapter_conforms_and_is_callable():
    assert isinstance(declared_plan_v1, Adapter)  # runtime_checkable structural conformance
    assert declared_plan_v1.id == "declared_plan_v1"
    assert declared_plan_v1.name == "declared_plan" and declared_plan_v1.version == "v1"
    assert callable(declared_plan_v1)
    plan = {"nodes": [{"idx": 0, "agent": "a", "kind": "decision"}]}
    assert [s.idx for s in declared_plan_v1(plan)] == [s.idx for s in declared_plan_v1.to_steps(plan)]


def test_declared_plan_adapter_emits_only_declared_edges():
    plan = {
        "nodes": [
            {"idx": 0, "agent": "tool", "kind": "tool_call", "writes": ["kyc.tier"]},
            {"idx": 1, "agent": "p", "kind": "decision", "reads": [{"id": "kyc.tier", "producer": 0}]},
            {"idx": 2, "agent": "p", "kind": "decision", "reads": [{"id": "kyc.tier", "producer": 0}]},
        ]
    }
    steps = declared_plan_v1.to_steps(plan)
    edges = [e for s in steps for e in s.deps]
    assert edges, "the producer-named reads should wire declared edges"
    assert all(e.grade is Grade.DECLARED for e in edges)  # never OBSERVED / INFERRED
    # the resource id and flags ride in evidence; resource stays None (declared convention)
    e = steps[1].deps[0]
    assert e.resource is None
    assert e.evidence["declared"] is True and e.evidence["resource_id"] == "kyc.tier"
    assert e.evidence["adapter"] == "declared_plan_v1"


def test_declared_plan_control_preds_become_exec_preds_verbatim():
    plan = {
        "nodes": [
            {"idx": 0, "agent": "a", "kind": "decision"},                       # omitted -> linear
            {"idx": 1, "agent": "a", "kind": "decision", "control_preds": []},  # explicit root
            {"idx": 2, "agent": "a", "kind": "decision", "control_preds": [0, 1]},  # merge
        ]
    }
    steps = declared_plan_v1.to_steps(plan)
    assert steps[0].exec_preds is None      # omitted -> the offline linear default
    assert steps[1].exec_preds == []        # explicit root, distinct from omitted
    assert steps[2].exec_preds == [0, 1]    # a merge of two control branches


def test_declared_plan_free_read_wires_no_edge_but_keeps_resource():
    # a read with no in-plan producer is a "free read": no depends_on edge, but the
    # resource id is retained in node_attrs so the lints still see it
    plan = {"nodes": [{"idx": 0, "agent": "a", "kind": "decision", "reads": ["external.feed"]}]}
    steps = declared_plan_v1.to_steps(plan)
    assert steps[0].deps == []
    assert steps[0].node_attrs["reads"] == ["external.feed"]


def test_declared_plan_schema_validation_raises():
    with pytest.raises(ValueError):  # duplicate idx
        declared_plan_v1.to_steps(
            {"nodes": [
                {"idx": 0, "agent": "a", "kind": "decision"},
                {"idx": 0, "agent": "b", "kind": "tool_call"},
            ]}
        )
    with pytest.raises(ValueError):  # bool idx (True == 1 would alias a step node)
        declared_plan_v1.to_steps({"nodes": [{"idx": True, "agent": "a", "kind": "decision"}]})
    with pytest.raises(ValueError):  # kind not in {decision, tool_call}
        declared_plan_v1.to_steps({"nodes": [{"idx": 0, "agent": "a", "kind": "plan"}]})
    with pytest.raises(ValueError):  # malformed resource-ref (dict without a string id)
        declared_plan_v1.to_steps(
            {"nodes": [{"idx": 0, "agent": "a", "kind": "decision", "reads": [{"producer": 1}]}]}
        )


def test_declared_plan_empty_or_none_is_empty():
    assert declared_plan_v1.to_steps(None) == []
    assert declared_plan_v1.to_steps({}) == []
    assert declared_plan_v1.to_steps({"nodes": []}) == []


def test_declared_plan_dataclass_mirror_lowers_like_the_dict():
    # the optional thin dataclass mirror lowers to the same steps as the plain dict
    from auditable.graph.adapters.declared_plan import DeclaredPlan, DeclaredPlanNode

    plan = DeclaredPlan(
        nodes=[
            DeclaredPlanNode(idx=0, agent="t", kind="tool_call", writes=["r"]),
            DeclaredPlanNode(idx=1, agent="d", kind="decision",
                             reads=[{"id": "r", "producer": 0}], control_preds=[0]),
        ],
        plan_id="p1",
        framework="langgraph",
    )
    steps = declared_plan_v1.to_steps(plan)
    assert [s.idx for s in steps] == [0, 1]
    assert steps[1].exec_preds == [0]
    assert steps[1].deps[0].grade is Grade.DECLARED
    assert steps[1].deps[0].evidence["resource_id"] == "r"
    # the plain-dict render of the same plan lowers identically
    via_dict = declared_plan_v1.to_steps(plan.as_dict())
    assert [s.idx for s in via_dict] == [s.idx for s in steps]


def test_declared_plan_builds_a_valid_low_coverage_session_graph():
    plan = {
        "nodes": [
            {"idx": 0, "agent": "tool", "kind": "tool_call", "writes": ["kyc.tier"]},
            {"idx": 1, "agent": "p", "kind": "decision", "reads": [{"id": "kyc.tier", "producer": 0}]},
        ]
    }
    g = SessionGraph.from_steps(declared_plan_v1.to_steps(plan))
    cov = g.coverage()
    assert cov.by_grade[Grade.DECLARED] == 1 and cov.by_grade[Grade.OBSERVED] == 0
    assert cov.observed_fraction == 0.0
    # the declared layer reads as low coverage: no State-B score
    assert structural_risk(g).state == STATE_LOW_COVERAGE


# --- lint: write-with-no-prior-read ------------------------------------------


def test_write_with_no_prior_read_fires_and_is_silent():
    # positive: node 0 writes order.123 with no read of it in its backward slice
    pos = {"nodes": [{"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["order.123"]}]}
    f = write_with_no_prior_read(_graph(pos))
    assert [(x.node_idx, x.resource_id) for x in f] == [(0, "order.123")]

    # negative: a prior read of order.123 sits in node 1's backward slice (producer 0)
    neg = {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["order.123"]},  # producer
            {"idx": 1, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "order.123", "producer": 0}], "writes": ["order.123"]},
        ]
    }
    f = write_with_no_prior_read(_graph(neg))
    # node 1 read order.123 first -> no finding for node 1
    assert not any(x.node_idx == 1 for x in f)


# --- lint: flippable-dependency-annotation -----------------------------------


def _flip_plan(**flags):
    ref = {"id": "policy.v", "producer": 0, "volatile": True, **flags}
    return {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["policy.v"]},
            {"idx": 1, "agent": "d", "kind": "decision", "reads": [ref]},
        ]
    }


def test_flippable_dependency_fires_on_unpinned_volatile_feeding_decision():
    f = flippable_dependency_annotations(_graph(_flip_plan()))
    assert [(x.node_idx, x.resource_id) for x in f] == [(1, "policy.v")]
    assert f[0].severity == "warning"            # annotation, not a value-flip proof
    assert "would-flip" in f[0].detail


def test_flippable_dependency_silent_when_pinned_or_revalidated():
    assert flippable_dependency_annotations(_graph(_flip_plan(pinned=True))) == []
    assert flippable_dependency_annotations(_graph(_flip_plan(revalidates=True))) == []


def test_flippable_dependency_fires_via_backward_slice_arm():
    # arm (b): the volatile reader is a tool_call UPSTREAM of a decision (not the
    # decision itself). A downstream decision rests on it through the backward slice,
    # so feeds_decision is true via nx.descendants, not via reader-is-decision.
    plan = {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["policy.v"]},
            {"idx": 1, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "policy.v", "producer": 0, "volatile": True}], "writes": ["snap"]},
            {"idx": 2, "agent": "d", "kind": "decision",
             "reads": [{"id": "snap", "producer": 1}]},
        ]
    }
    f = flippable_dependency_annotations(_graph(plan))
    # the annotation fires on the volatile tool read at node 1, reached via the slice
    assert [(x.node_idx, x.resource_id) for x in f] == [(1, "policy.v")]


def test_flippable_dependency_silent_when_volatile_tool_read_feeds_no_decision():
    # negative for arm (b): a volatile tool read whose downstream is all tool_calls,
    # no decision anywhere in the plan -> feeds_decision stays false, no annotation
    plan = {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["policy.v"]},
            {"idx": 1, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "policy.v", "producer": 0, "volatile": True}], "writes": ["snap"]},
            {"idx": 2, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "snap", "producer": 1}], "writes": ["order.1"]},
        ]
    }
    assert flippable_dependency_annotations(_graph(plan)) == []


# --- lint: scope-vs-snapshot -------------------------------------------------


def test_scope_vs_snapshot_fires_on_strict_superset():
    # granted scope {a, b} strictly exceeds the read-into-snapshot set {a}
    pos = {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["a"]},
            {"idx": 1, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "a", "producer": 0}], "scope": ["a", "b"]},
        ]
    }
    f = scope_vs_snapshot(_graph(pos))
    # reports scope - read_set = {b}
    assert [(x.node_idx, x.resource_id) for x in f] == [(1, "b")]


def test_scope_vs_snapshot_silent_on_equal_or_subset():
    equal = {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["a"]},
            {"idx": 1, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "a", "producer": 0}], "scope": ["a"]},
        ]
    }
    assert scope_vs_snapshot(_graph(equal)) == []

    subset = {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["a"]},
            {"idx": 1, "agent": "t", "kind": "tool_call", "writes": ["b"]},
            {"idx": 2, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "a", "producer": 0}, {"id": "b", "producer": 1}], "scope": ["a"]},
        ]
    }
    assert scope_vs_snapshot(_graph(subset)) == []

    # a node with no scope claim is skipped entirely
    no_scope = {"nodes": [{"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["a"]}]}
    assert scope_vs_snapshot(_graph(no_scope)) == []


# --- lint: missing-revalidation-barrier --------------------------------------


def test_missing_revalidation_barrier_fires_with_no_intervening_reread():
    # 0 writes policy.v; 1 volatile-reads it and writes snap; 2 acts (writes order.1)
    # resting on snap. Control flow 0 -> 1 -> 2, no re-read of policy.v before the act.
    pos = {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["policy.v"]},
            {"idx": 1, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "policy.v", "producer": 0, "volatile": True}], "writes": ["snap"]},
            {"idx": 2, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "snap", "producer": 1}], "writes": ["order.1"]},
        ]
    }
    f = missing_revalidation_barrier(_graph(pos))
    # the action node 2 rests on the stale volatile read of policy.v
    assert any(x.node_idx == 2 and x.resource_id == "policy.v" for x in f)


def test_missing_revalidation_barrier_silent_with_intervening_barrier():
    # insert a barrier node 2 that re-reads (revalidates) policy.v strictly between the
    # volatile read (node 1) and the action (node 3) in handoff order 0->1->2->3
    neg = {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["policy.v"]},
            {"idx": 1, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "policy.v", "producer": 0, "volatile": True}], "writes": ["snap"]},
            {"idx": 2, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "policy.v", "producer": 0, "volatile": True, "revalidates": True},
                       {"id": "snap", "producer": 1}], "writes": ["snap2"]},
            {"idx": 3, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "snap2", "producer": 2}], "writes": ["order.1"]},
        ]
    }
    f = missing_revalidation_barrier(_graph(neg))
    # the action node 3 is protected by the barrier at node 2 -> silent for policy.v
    assert not any(x.node_idx == 3 and x.resource_id == "policy.v" for x in f)


def test_missing_revalidation_barrier_fires_on_volatile_decision_action():
    # isolate the decision-with-volatile-dependency arm of is_action: node 1 is a
    # decision with NO writes whose action-ness comes only from its volatile read.
    pos = {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["policy.v"]},
            {"idx": 1, "agent": "d", "kind": "decision",
             "reads": [{"id": "policy.v", "producer": 0, "volatile": True}]},
        ]
    }
    f = missing_revalidation_barrier(_graph(pos))
    # the no-writes decision is the action; the stale volatile read fires at node 1
    assert [(x.node_idx, x.resource_id) for x in f] == [(1, "policy.v")]


# --- analyze_plan ------------------------------------------------------------


def _full_plan():
    """A small plan that plants one of each lint and a clear execution keystone."""
    return {
        "nodes": [
            {"idx": 0, "agent": "kyc_tool", "kind": "tool_call", "writes": ["kyc.tier"]},
            {"idx": 1, "agent": "planner", "kind": "decision",
             "reads": [{"id": "kyc.tier", "producer": 0, "volatile": True}],
             "scope": ["kyc.tier", "order.x"]},
            {"idx": 2, "agent": "exec", "kind": "tool_call",
             "reads": [{"id": "kyc.tier", "producer": 0}], "writes": ["order.x"]},
        ]
    }


def test_analyze_plan_returns_report_with_keystone_and_findings():
    rep = analyze_plan(_full_plan())
    assert isinstance(rep, PreReport)
    assert rep.adapter == "declared_plan_v1" and rep.n_steps == 3
    # execution keystone = argmax execution_reach (the linear lead node 0)
    assert rep.keystone_idx == 0
    assert rep.keystone_followers == 2          # nodes 1 and 2 transitively follow node 0
    assert rep.execution_reach_by_idx == {0: 2, 1: 1, 2: 0}  # populated for every step
    # the planted lints all appear
    fired = {(x.lint, x.node_idx, x.resource_id) for x in rep.findings}
    assert ("write_with_no_prior_read", 0, "kyc.tier") in fired
    assert ("write_with_no_prior_read", 2, "order.x") in fired
    assert ("flippable_dependency_annotation", 1, "kyc.tier") in fired
    assert ("scope_vs_snapshot", 1, "order.x") in fired
    assert ("missing_revalidation_barrier", 1, "kyc.tier") in fired
    assert all(isinstance(x, LintFinding) for x in rep.findings)


def test_analyze_plan_withholds_state_b_risk():
    rep = analyze_plan(_full_plan())
    # State B (dependency-state) blast-share risk is withheld at PRE, never a number
    assert rep.state_b_risk is None
    assert rep.state_b_withheld is True
    reason = rep.state_b_withheld_reason.lower()
    assert "declared-only" in reason or "declared" in reason
    assert "low_coverage" in reason or "low coverage" in reason

    # independently pin the no-score boundary: structural_risk on the declared graph
    steps = declared_plan_v1.to_steps(_full_plan())
    assert structural_risk(SessionGraph.from_steps(steps)).state == STATE_LOW_COVERAGE


def test_analyze_plan_raises_if_declared_graph_ever_scores(monkeypatch):
    # the boundary is an invariant: if structural_risk ever returned a scored verdict
    # on a declared graph, analyze_plan must raise rather than emit a State-B number
    from auditable.graph.risk import STATE_SCORED, RiskResult
    from auditable.graph.session import GraphCompleteness

    def fake_scored(graph, **kw):
        return RiskResult(
            state=STATE_SCORED,
            per_session=0.5,
            per_decision={},
            coverage=graph.coverage(),
            completeness=GraphCompleteness.COMPLETE,
        )

    monkeypatch.setattr(pre, "structural_risk", fake_scored)
    with pytest.raises(AssertionError):
        analyze_plan(_full_plan())


def test_analyze_plan_keystone_is_structural_not_blast_radius_keystone():
    rep = analyze_plan(_full_plan())
    # the PRE report must NOT carry the POST blast-share score fields (per_session /
    # per_decision live on RiskResult, not here)
    assert not hasattr(rep, "per_session")
    assert not hasattr(rep, "per_decision")
    # the notes state the keystone is a structural chokepoint, not a failure predictor
    joined = " ".join(rep.notes).lower()
    assert "structural" in joined and "chokepoint" in joined
    assert "not the post blast-radius keystone" in joined or "not a failure predictor" in joined


def test_analyze_plan_summary_renders_withheld_and_keystone():
    text = analyze_plan(_full_plan()).summary()
    assert "PRE declared-plan analysis" in text
    assert "WITHHELD" in text
    assert "structural chokepoint" in text


@pytest.mark.parametrize(
    "plan,n_steps",
    [
        ({"nodes": []}, 0),
        ({"nodes": [{"idx": 0, "agent": "a", "kind": "decision"}]}, 1),
    ],
)
def test_analyze_plan_small_plan_does_not_raise(plan, n_steps):
    # regression: a 0- or 1-step declared plan is gated as no_score:single_decision
    # (Gate 1 fires before the low-coverage gate). The PRE invariant is "no State-B
    # number", i.e. any no_score:* state, so analyze_plan must return a report, not
    # raise. structural_risk confirms the single_decision gate is what is hit here.
    from auditable.graph.risk import STATE_SINGLE_DECISION
    from auditable.graph.adapters.declared_plan import declared_plan_v1

    steps = declared_plan_v1.to_steps(plan)
    assert structural_risk(SessionGraph.from_steps(steps)).state == STATE_SINGLE_DECISION

    rep = analyze_plan(plan)  # must not raise AssertionError
    assert isinstance(rep, PreReport)
    assert rep.n_steps == n_steps
    assert rep.keystone_idx is None and rep.keystone_followers == 0
    # State B stays withheld regardless of which no_score gate produced the verdict
    assert rep.state_b_risk is None and rep.state_b_withheld is True


def test_analyze_plan_none_keystone_when_no_control_flow_followers():
    # all-explicit-roots plan: every node is its own control-flow root (control_preds
    # == []), so no node has any transitive follower. keystone_idx comes back None and
    # the summary renders the none-branch. This avoids the single_decision crash by
    # keeping >= 2 steps while still driving the empty-keystone path.
    plan = {
        "nodes": [
            {"idx": 0, "agent": "a", "kind": "decision", "control_preds": []},
            {"idx": 1, "agent": "a", "kind": "decision", "control_preds": []},
        ]
    }
    rep = analyze_plan(plan)
    assert rep.keystone_idx is None
    assert rep.keystone_followers == 0
    assert rep.execution_reach_by_idx == {0: 0, 1: 0}
    text = rep.summary()
    assert "execution keystone: (none -- no control-flow followers)" in text


# --- no top-level leak (PRE stays under auditable.graph.*) --------------------


def test_pre_adds_no_top_level_export():
    import auditable

    for name in ("analyze_plan", "PreReport", "DeclaredPlanAdapter", "declared_plan_v1", "execution_reach"):
        assert not hasattr(auditable, name), f"{name} must not leak to the top level"


def test_pre_names_reachable_under_submodules():
    # they ARE reachable under the graph submodules, just not from the top level
    from auditable.graph import execution_reach  # noqa: F401
    from auditable.graph.pre import PreReport, analyze_plan  # noqa: F401
    from auditable.graph.adapters import DeclaredPlanAdapter, declared_plan_v1  # noqa: F401
    from auditable.graph.adapters.declared_plan import declared_plan_v1 as singleton

    assert isinstance(singleton, DeclaredPlanAdapter)


# --- regression: final-review (round-2) fixes --------------------------------


def test_missing_revalidation_barrier_silent_when_action_node_revalidates():
    # the action node itself re-reads (revalidates) the volatile resource at time-of-use.
    # node 1 volatile-reads policy.v; node 2 rests on node 1 (via snap) AND re-reads
    # policy.v with revalidates=True before writing. The same-node revalidation clears
    # the prior volatile read, so no missing-barrier finding should fire for policy.v.
    plan = {
        "nodes": [
            {"idx": 0, "agent": "t", "kind": "tool_call", "writes": ["policy.v"]},
            {"idx": 1, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "policy.v", "producer": 0, "volatile": True}], "writes": ["snap"]},
            {"idx": 2, "agent": "t", "kind": "tool_call",
             "reads": [{"id": "snap", "producer": 1},
                       {"id": "policy.v", "producer": 0, "volatile": True, "revalidates": True}],
             "writes": ["order.1"]},
        ]
    }
    f = missing_revalidation_barrier(_graph(plan))
    # node 2 (the revalidating action) must be clear; node 1 is a separate stale action
    # (it writes snap on its own un-revalidated volatile read), so it may still fire.
    assert not any(x.node_idx == 2 and x.resource_id == "policy.v" for x in f)


def test_declared_plan_raises_on_invalid_explicit_producer():
    # an explicitly named producer that is not a prior declared node must raise, not
    # silently drop the edge (which would suppress the very lints the user expects).
    with pytest.raises(ValueError, match="producer"):
        declared_plan_v1.to_steps(
            {"nodes": [{"idx": 0, "agent": "a", "kind": "decision",
                        "reads": [{"id": "r", "producer": 5}]}]}  # node 5 does not exist
        )


def test_declared_plan_raises_on_invalid_control_pred():
    # a forward / non-prior control predecessor must raise (topological-order contract)
    with pytest.raises(ValueError, match="control_preds"):
        declared_plan_v1.to_steps(
            {"nodes": [{"idx": 0, "agent": "a", "kind": "decision", "control_preds": [9]}]}
        )
    # a non-integer control predecessor must raise
    with pytest.raises(ValueError, match="control_preds"):
        declared_plan_v1.to_steps(
            {"nodes": [{"idx": 0, "agent": "a", "kind": "decision", "control_preds": ["x"]}]}
        )
