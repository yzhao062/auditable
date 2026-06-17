import pytest

pytest.importorskip("networkx")  # the graph kernel needs the optional 'graph' extra

from auditable.graph import build_graph, characterize, downstream_reach


def _steps(kinds):
    return [{"idx": i, "agent": f"A{i % 2}", "kind": k} for i, k in enumerate(kinds)]


def test_build_graph_types_nodes_and_edges():
    G = build_graph(_steps(["decision", "tool_call", "decision"]))
    ntypes = {d["ntype"] for _, d in G.nodes(data=True)}
    assert {"agent", "decision", "tool_call", "dependency_resource"} <= ntypes
    etypes = {d["etype"] for *_, d in G.edges(data=True)}
    assert "emits" in etypes and "handoff_to" in etypes and "depends_on" in etypes


def test_characterize_reports_size_and_split():
    G = build_graph(_steps(["decision", "decision", "tool_call", "decision"]))
    c = characterize(G)
    assert c["n_steps"] == 4
    assert c["n_tool_calls"] == 1
    assert c["n_exec_edges"] > 0 and c["n_dep_edges"] > 0


def test_full_context_makes_early_steps_high_blast_radius():
    G = build_graph(_steps(["decision"] * 5), dependency="full_context")
    # step 0 is depended on by all four later steps; the last step by none
    assert downstream_reach(G, 0) == 4
    assert downstream_reach(G, 4) == 0


def test_chain_dependency_is_local():
    G = build_graph(_steps(["decision"] * 5), dependency="chain")
    # under a chain, the first step is reached transitively by all later steps too
    assert downstream_reach(G, 0) == 4
    assert downstream_reach(G, 3) == 1


def test_explicit_deps_wire_observed_dependencies():
    # step 3 (a write) observably relied on steps 0 and 1 (two reads); nothing else.
    steps = [
        {"idx": 0, "agent": "env", "kind": "tool_call", "deps": []},
        {"idx": 1, "agent": "env", "kind": "tool_call", "deps": []},
        {"idx": 2, "agent": "A", "kind": "decision", "deps": []},
        {"idx": 3, "agent": "env", "kind": "tool_call", "deps": [0, 1]},
    ]
    G = build_graph(steps, dependency="explicit", shared_resource=False)
    c = characterize(G)
    # only the two write->read edges exist, so depth is one hop (writes rest on reads)
    assert c["n_dep_edges"] == 2
    assert c["dep_depth"] == 1
    # the write's audit surface: the two reads it depended on
    assert downstream_reach(G, 0) == 1 and downstream_reach(G, 1) == 1
    assert downstream_reach(G, 2) == 0


def test_explicit_deps_ignore_unknown_targets():
    # a dep pointing at a step that does not exist is dropped, not an error
    G = build_graph([{"idx": 0, "agent": "A", "kind": "decision", "deps": [99]}],
                    dependency="explicit", shared_resource=False)
    assert characterize(G)["n_dep_edges"] == 0


def test_explicit_deps_drop_self_forward_and_duplicate():
    # step 1 names a duplicate (0,0), a self-dep (1), and a forward/unknown dep (5);
    # only the single prior edge 1->0 survives, and no self-loop zeroes out the depth
    steps = [
        {"idx": 0, "agent": "A", "kind": "tool_call", "deps": []},
        {"idx": 1, "agent": "A", "kind": "tool_call", "deps": [0, 0, 1, 5]},
    ]
    c = characterize(build_graph(steps, dependency="explicit", shared_resource=False))
    assert c["n_dep_edges"] == 1
    assert c["dep_depth"] == 1


def test_build_rejects_duplicate_and_non_int_idx():
    with pytest.raises(ValueError):
        build_graph([{"idx": 0, "agent": "A", "kind": "decision"},
                     {"idx": 0, "agent": "A", "kind": "decision"}])
    with pytest.raises(ValueError):
        build_graph([{"idx": 0.0, "agent": "A", "kind": "decision"}])


def test_build_requires_steps_to_have_kind():
    with pytest.raises(KeyError):
        build_graph([{"idx": 0, "agent": "A"}])
