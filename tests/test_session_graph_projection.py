"""Tests for SessionGraph.to_networkx(): the typed projection.

The projection must preserve the three typed seams the low-level build_graph
kernel drops -- per-edge grade / evidence / resource on depends_on edges, node
attributes (model_id / decision_basis) on step nodes, and the exec_preds
execution topology -- while staying readable by the kernel's own characterizers.
"""
import pytest

pytest.importorskip("networkx")  # the projection needs the optional 'graph' extra

from auditable.graph import characterize, downstream_reach, layered_features
from auditable.graph.session import (
    DependencyEdge,
    Grade,
    ResourceRef,
    SessionGraph,
    Step,
)


def _depends_on(G):
    return [(u, v, d) for u, v, d in G.edges(data=True) if d["etype"] == "depends_on"]


def _handoffs(G):
    return {(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "handoff_to"}


def test_to_networkx_preserves_per_edge_grade_evidence_resource():
    steps = [
        Step(idx=0, agent="env", kind="tool_call"),
        Step(
            idx=1,
            agent="m",
            kind="decision",
            deps=[
                DependencyEdge(
                    0,
                    grade=Grade.OBSERVED,
                    resource=ResourceRef("db", "orders", "row_7"),
                    evidence={"matched_on": "row_7"},
                )
            ],
        ),
    ]
    G = SessionGraph.from_steps(steps).to_networkx()
    dep = _depends_on(G)
    assert len(dep) == 1
    u, v, d = dep[0]
    assert (u, v) == ("step::1", "step::0")  # depends_on points dependent -> dependency
    assert d["grade"] is Grade.OBSERVED
    assert d["resource"] == ResourceRef("db", "orders", "row_7")
    assert d["evidence"] == {"matched_on": "row_7"}


def test_to_networkx_keeps_mixed_grades_distinct():
    # observed and inferred edges from the same step coexist with their own grades
    steps = [
        Step(idx=0, agent="env", kind="tool_call"),
        Step(idx=1, agent="env", kind="tool_call"),
        Step(
            idx=2,
            agent="m",
            kind="decision",
            deps=[
                DependencyEdge(0, grade=Grade.OBSERVED),
                DependencyEdge(1, grade=Grade.INFERRED),
            ],
        ),
    ]
    G = SessionGraph.from_steps(steps).to_networkx()
    by_target = {v: d["grade"] for u, v, d in _depends_on(G)}
    assert by_target["step::0"] is Grade.OBSERVED
    assert by_target["step::1"] is Grade.INFERRED


def test_to_networkx_preserves_node_attrs():
    s = Step(
        idx=0,
        agent="m",
        kind="decision",
        node_attrs={"model_id": "gpt-x", "decision_basis": "invoice matches an approved PO"},
    )
    nd = SessionGraph.from_steps([s]).to_networkx().nodes["step::0"]
    # the core typing the kernel sets, plus the model-facet attrs it drops
    assert nd["ntype"] == "decision" and nd["idx"] == 0 and nd["agent"] == "m"
    assert nd["model_id"] == "gpt-x"
    assert nd["decision_basis"] == "invoice matches an approved PO"


def test_to_networkx_node_attrs_do_not_clobber_core_typing():
    # a stray node_attr must not override ntype / idx / agent
    s = Step(idx=3, agent="m", kind="decision", node_attrs={"ntype": "bogus", "idx": 99})
    nd = SessionGraph.from_steps([s]).to_networkx().nodes["step::3"]
    assert nd["ntype"] == "decision" and nd["idx"] == 3


def test_to_networkx_exec_preds_linear_root_and_merge():
    steps = [
        Step(idx=0, agent="a", kind="decision"),                     # None, first -> no pred
        Step(idx=1, agent="a", kind="decision"),                     # None -> linear from 0
        Step(idx=2, agent="b", kind="decision", exec_preds=[]),      # explicit root, not first
        Step(idx=3, agent="b", kind="decision", exec_preds=[0, 1]),  # merge of two branches
    ]
    G = SessionGraph.from_steps(steps).to_networkx()
    assert _handoffs(G) == {
        ("step::0", "step::1"),
        ("step::0", "step::3"),
        ("step::1", "step::3"),
    }
    # the explicit [] root has no incoming execution edge, unlike the None default
    assert not any(v == "step::2" for _, v in _handoffs(G))


def test_to_networkx_skips_forward_and_unknown_edges():
    # a dependency or exec predecessor that is not an already-seen step is dropped,
    # keeping the projection causal and the depends_on DAG acyclic
    steps = [
        Step(idx=0, agent="m", kind="decision", deps=[DependencyEdge(5)], exec_preds=[9]),
        Step(idx=1, agent="m", kind="decision"),
    ]
    G = SessionGraph.from_steps(steps).to_networkx()
    assert _depends_on(G) == []          # the unknown dep target 5 is not wired
    assert _handoffs(G) == {("step::0", "step::1")}  # only the valid linear handoff


def test_to_networkx_is_read_by_kernel_characterizers():
    # the typed projection must stay consistent with the kernel's readers
    steps = [
        Step(idx=0, agent="env", kind="tool_call"),
        Step(idx=1, agent="m", kind="decision", deps=[DependencyEdge(0, grade=Grade.OBSERVED)]),
        Step(idx=2, agent="m", kind="decision", deps=[DependencyEdge(0, grade=Grade.OBSERVED)]),
    ]
    G = SessionGraph.from_steps(steps).to_networkx()
    c = characterize(G)
    assert c["n_steps"] == 3 and c["n_dep_edges"] == 2
    assert downstream_reach(G, 0) == 2  # steps 1 and 2 both depend on step 0
    dep_feats = layered_features(G)["dep"]
    assert dep_feats["n_dep_edges"] == 2
    assert dep_feats["max_blast_indeg"] == 2  # step 0 is the blast hub
