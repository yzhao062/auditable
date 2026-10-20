"""auditable.featurize: trace normalization and the features the methods read.

The structural layers and the Who&When step matrix are compared element by element with
a fixture generated from GRADE at commit 3839a57
(benchmarks/catchbench_parity/gen_grade_fixture.py), so these tests need no GRADE checkout.
"""
import json
import pathlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("networkx")

from auditable.featurize import (  # noqa: E402
    GRADE_LAYERS,
    causal_z,
    count_series,
    full_context_edges,
    grade_layer,
    grade_layer_matrix,
    normalize_trace,
    run_edge_count,
    run_max_span,
    span_series,
    step_graph,
    to_pyg,
    who_when_step_matrix,
)
from auditable.graph import build_graph, feature_vector  # noqa: E402
from auditable.graph.session import DependencyEdge, SessionGraph, Step  # noqa: E402

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "grade_3839a57_features.json"


@pytest.fixture(scope="module")
def grade_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_layers_equal_grade_3839a57(grade_fixture):
    n = 0
    for case in grade_fixture["cases"]:
        for key, expected in case["layers"].items():
            mode, shared, layer = key.split("/")
            G = build_graph(case["steps"], dependency=mode, shared_resource=bool(int(shared)))
            names, values = feature_vector(G, layer=layer)
            assert names == grade_fixture["names"][layer]
            assert values == expected, (key, values, expected)
            n += 1
    assert n == 24 * 3 * 2 * len(GRADE_LAYERS)


def test_who_when_step_matrix_equals_grade_3839a57(grade_fixture):
    for case in grade_fixture["cases"]:
        assert who_when_step_matrix(case["steps"]).tolist() == case["who_when_step_matrix"]


def test_who_when_step_matrix_ignores_step_numbering(grade_fixture):
    for case in grade_fixture["cases"]:
        new = {s["idx"]: 10 + 3 * i for i, s in enumerate(case["steps"])}
        renumbered = [dict(s, idx=new[s["idx"]], deps=[new[d] for d in s.get("deps", ())]) for s in case["steps"]]
        assert who_when_step_matrix(renumbered).tolist() == case["who_when_step_matrix"]


def _steps():
    return [
        {"idx": 0, "agent": "planner", "kind": "decision", "deps": []},
        {"idx": 1, "agent": "coder", "kind": "tool_call", "deps": [0]},
        {"idx": 2, "agent": "coder", "kind": "decision", "deps": [1]},
        {"idx": 3, "agent": "tester", "kind": "tool_call", "deps": [0, 2]},
    ]


def test_normalize_step_list_keeps_order_and_texts():
    steps = [dict(s, text=f"t{s['idx']}") for s in _steps()]
    view = normalize_trace(steps)
    assert view.step_ids == [0, 1, 2, 3]
    assert view.texts == ["t0", "t1", "t2", "t3"]
    assert view.graph is None
    assert steps[0]["deps"] == []  # input not mutated


def test_normalize_run_dict_and_full_context():
    run = {"steps": [{"idx": i, "agent": "a", "kind": "decision"} for i in range(3)], "texts": ["x", "y", "z"]}
    view = normalize_trace(run, dependency="full_context")
    assert [s["deps"] for s in view.steps] == [[], [0], [0, 1]]
    assert view.require_texts() == ["x", "y", "z"]
    assert "deps" not in run["steps"][2]
    # full_context replaces recorded dependencies, as build_graph(dependency="full_context") does
    assert [s["deps"] for s in normalize_trace(_steps(), dependency="full_context").steps] == \
        [[], [0], [0, 1], [0, 1, 2]]


def test_require_texts_names_the_missing_field():
    with pytest.raises(ValueError, match="step text"):
        normalize_trace(_steps()).require_texts()


def test_typed_graph_round_trip_and_passthrough():
    G = build_graph(_steps(), dependency="explicit", shared_resource=False)
    view = normalize_trace(G)
    assert view.graph is G
    assert [s["deps"] for s in view.steps] == [[], [0], [1], [0, 2]]
    assert view.typed_graph() is G


def test_grade_layer_matches_feature_vector_and_honours_shared_resource():
    steps = _steps()
    for shared in (True, False):
        G = build_graph(steps, dependency="explicit", shared_resource=shared)
        assert grade_layer(steps, "full", shared_resource=shared) == feature_vector(G, layer="full")
    M = grade_layer_matrix([steps, steps[:2]], "flatdep")
    assert M.shape == (2, len(feature_vector(build_graph(steps), layer="flatdep")[0]))


def test_session_graph_keeps_parallel_roots_and_merge():
    # Two parallel roots (0 and 1) merge at step 2; a linear chain would hand 0 -> 1 -> 2.
    sg = SessionGraph.from_steps([
        Step(idx=0, agent="a", kind="decision", exec_preds=[], node_attrs={"text": "plan A"}),
        Step(idx=1, agent="b", kind="tool_call", exec_preds=[], node_attrs={"text": "fetch B"}),
        Step(idx=2, agent="c", kind="decision", exec_preds=[0, 1], deps=[DependencyEdge(src_idx=1)],
             node_attrs={"text": "merge"}),
    ])
    view = normalize_trace(sg)
    assert view.source == "session_graph"
    assert view.texts == ["plan A", "fetch B", "merge"]
    assert [s["deps"] for s in view.steps] == [[], [], [1]]
    handoffs = sorted((u, v) for u, v, d in view.graph.edges(data=True) if d.get("etype") == "handoff_to")
    assert handoffs == [("step::0", "step::2"), ("step::1", "step::2")]
    M = who_when_step_matrix(sg)
    # agents a and b each hand off to c; c receives from both
    assert M[0, 6] == pytest.approx(1 / 3) and M[1, 6] == pytest.approx(1 / 3)
    assert M[2, 7] == pytest.approx(2 / 3)
    chain = who_when_step_matrix([{"idx": i, "agent": a, "kind": k} for i, (a, k) in
                                  enumerate([("a", "decision"), ("b", "tool_call"), ("c", "decision")])])
    assert chain[2, 7] == pytest.approx(1 / 3)  # the linear chain sees one sender into c


def test_full_context_edges_point_current_to_prior():
    assert full_context_edges(3).tolist() == [[1, 2, 2], [0, 0, 1]]
    assert full_context_edges(1).shape == (2, 0)


def test_step_graph_and_run_statistics():
    X, edges = step_graph(_steps())
    assert X.tolist() == [
        [0.0, 0.0, 0.0, 0.0],
        [1 / 3, 1.0, 1.0, 1.0],
        [2 / 3, 0.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 3.0],
    ]
    assert edges.tolist() == [[1, 2, 3, 3], [0, 1, 0, 2]]
    assert run_max_span(_steps()) == 3.0
    assert run_edge_count(_steps()) == 4.0
    assert span_series(_steps()).tolist() == [0.0, 1.0, 1.0, 3.0]
    assert count_series(_steps()).tolist() == [0.0, 1.0, 1.0, 2.0]


def test_series_ignore_unknown_and_forward_deps():
    steps = [{"idx": 0, "agent": "a", "kind": "decision", "deps": [7]},
             {"idx": 1, "agent": "a", "kind": "decision", "deps": [2]},
             {"idx": 2, "agent": "a", "kind": "decision", "deps": [0]}]
    assert span_series(steps).tolist() == [0.0, 0.0, 2.0]
    assert count_series(steps).tolist() == [0.0, 0.0, 1.0]


def test_causal_z_uses_only_the_past():
    z = causal_z([1.0, 3.0, 2.0, 10.0])
    assert z[:3].tolist() == [0.0, 0.0, 0.0]  # fewer than two earlier values at t < 2; t = 2 is at the mean
    prior = np.array([1.0, 3.0, 2.0])
    assert z[3] == pytest.approx((10.0 - prior.mean()) / prior.std())
    assert causal_z([5.0, 5.0, 5.0, 9.0]).tolist() == [0.0, 0.0, 0.0, 0.0]  # constant history


def test_to_pyg_encodes_every_node_in_order():
    G = build_graph(_steps(), dependency="explicit", shared_resource=False)
    X, edges = to_pyg(G)
    nodes = list(G.nodes())
    assert X.shape == (len(nodes), 5)
    assert edges.shape == (2, G.number_of_edges())
    for i, n in enumerate(nodes):
        ntype = G.nodes[n].get("ntype")
        assert X[i, 2:].tolist() == [float(ntype == "decision"), float(ntype == "tool_call"),
                                     float(ntype not in ("decision", "tool_call"))]
        assert X[i, 0] == G.in_degree(n) and X[i, 1] == G.out_degree(n)
