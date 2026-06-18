"""Tests for the layered structural features (flat / exec / dep)."""
import pytest

pytest.importorskip("networkx")  # the graph kernel needs the optional 'graph' extra

from auditable.graph import build_graph, layered_features, feature_vector


def _linear(n, agent="a"):
    return [{"idx": i, "agent": agent, "kind": "decision"} for i in range(n)]


def test_groups_and_keys():
    G = build_graph(_linear(4), dependency="chain")
    f = layered_features(G)
    assert set(f) == {"flat", "exec", "dep"}
    assert set(f["flat"]) == {"n_steps", "n_tool_calls", "n_decisions", "n_agents"}
    assert set(f["exec"]) == {"max_agent_outdeg", "agent_gini", "n_agent_transitions", "agent_recurrence"}
    assert set(f["dep"]) == {"n_dep_edges", "dep_depth", "max_blast_indeg", "max_audit_outdeg"}


def test_feature_vector_is_nested():
    G = build_graph(_linear(4), dependency="chain")
    nf, vf = feature_vector(G, layer="flat")
    ne, ve = feature_vector(G, layer="exec")
    nfull, vfull = feature_vector(G, layer="full")
    assert nf == nfull[:4] and ne == nfull[:8]  # nested, stable column order
    assert len(nf) == 4 and len(ne) == 8 and len(nfull) == 12
    assert all(isinstance(x, float) for x in vfull)


def test_feature_vector_rejects_bad_layer():
    G = build_graph(_linear(3), dependency="chain")
    with pytest.raises(ValueError):
        feature_vector(G, layer="exec_only")


def test_single_agent_has_no_execution_topology():
    G = build_graph(_linear(5, "solo"), dependency="chain")
    e = layered_features(G)["exec"]
    assert e["max_agent_outdeg"] == 5      # one agent emits every step
    assert e["agent_gini"] == 0.0
    assert e["n_agent_transitions"] == 0   # control never moves
    assert e["agent_recurrence"] == 0.0


def test_multi_agent_has_transitions_and_returns():
    steps = [{"idx": i, "agent": a, "kind": "decision"}
             for i, a in enumerate(["a", "b", "a", "b"])]
    e = layered_features(build_graph(steps, dependency="chain"))["exec"]
    assert e["n_agent_transitions"] == 2   # a->b and b->a
    assert e["agent_recurrence"] > 0       # control returns to seen agents
    assert e["agent_gini"] == 0.0          # equal activity, still no concentration


def test_observed_dependency_shapes_dep_features():
    # two reads then one write depending on both (the tau-bench audit-surface shape)
    steps = [
        {"idx": 0, "agent": "env", "kind": "tool_call", "deps": []},
        {"idx": 1, "agent": "env", "kind": "tool_call", "deps": []},
        {"idx": 2, "agent": "env", "kind": "tool_call", "deps": [0, 1]},
    ]
    d = layered_features(build_graph(steps, dependency="explicit", shared_resource=False))["dep"]
    assert d["n_dep_edges"] == 2
    # depends_on points dependent -> dependency, so the write's audit surface is out-degree
    assert d["max_audit_outdeg"] == 2  # the write rests on two reads (audit surface)
    assert d["max_blast_indeg"] == 1   # each read has a single dependent (blast)
    assert d["dep_depth"] == 1       # one hop by construction


def test_full_context_dependency_scales_with_size():
    # inferred full-context deps: step i depends on all prior -> edges grow with n
    small = layered_features(build_graph(_linear(3), dependency="full_context"))["dep"]
    big = layered_features(build_graph(_linear(6), dependency="full_context"))["dep"]
    assert big["n_dep_edges"] > small["n_dep_edges"]
    assert big["dep_depth"] > small["dep_depth"]  # depth tracks run length when inferred


def test_deterministic():
    G = build_graph(_linear(4), dependency="chain")
    assert feature_vector(G, layer="full") == feature_vector(G, layer="full")
