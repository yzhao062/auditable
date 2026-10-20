"""Graph auditing methods: the PyGOD run detectors, GUARDIAN-Lite and G-Safeguard-Lite.

Every case runs on a few tiny hand-built graphs: chains stand in for resolved runs and stars
(a hub every step depends on) for failed ones. Parity with CatchBench is checked by the
benchmark harness (``benchmarks/catchbench_parity``), not here.
"""
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest  # noqa: E402

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("networkx")
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("pygod")

import networkx as nx  # noqa: E402

from auditable.base import NotSupported  # noqa: E402
from auditable.featurize import to_pyg  # noqa: E402
from auditable.methods import get_method, list_methods  # noqa: E402
from auditable.methods.run import graph as run_graph  # noqa: E402

_EXPECTED = {
    # method_id: (board alias, fit scope, supervised, fidelity)
    "run.pygod_dominant": ("pygod (graph AD)", "transductive", False, "wrapper"),
    "run.pygod_conad": ("pygod-conad", "transductive", False, "wrapper"),
    "run.pygod_anomalydae": ("pygod-anomalydae", "transductive", False, "wrapper"),
    "run.pygod_gaan": ("pygod-gaan", "transductive", False, "wrapper"),
    "run.guardian_lite": ("guardian (recon-AE)", "transductive", False, "simplified"),
    "run.gsafeguard_lite": ("g-safeguard (sup GNN)", "inductive", True, "simplified"),
}
_PYGOD = [m for m in _EXPECTED if m.startswith("run.pygod_")]
_UNSUPERVISED = _PYGOD + ["run.guardian_lite"]


def _chain(n):
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(f"s{i}", ntype="tool_call" if i % 2 else "decision", idx=i)
        if i:
            G.add_edge(f"s{i}", f"s{i - 1}")
    return G


def _star(n):
    G = nx.DiGraph()
    G.add_node("hub", ntype="decision", idx=0)
    for i in range(1, n):
        G.add_node(f"s{i}", ntype="tool_call", idx=i)
        G.add_edge(f"s{i}", "hub")
    return G


@pytest.fixture(scope="module")
def runs():
    """Eight chains then eight stars of 5 to 7 nodes, with failure labels (1 = star)."""
    graphs = [_chain(5 + i % 3) for i in range(8)] + [_star(5 + i % 3) for i in range(8)]
    return graphs, np.array([0] * 8 + [1] * 8)


# --- cards and registry -----------------------------------------------------------------------

@pytest.mark.parametrize("method_id", sorted(_EXPECTED))
def test_card_and_registry(method_id):
    alias, scope, supervised, fidelity = _EXPECTED[method_id]
    cls = get_method(method_id)
    card = cls.card
    assert card.method_id == method_id and card.family == "run" and card.information_state == "POST"
    assert card.board_aliases == (alias,)
    assert card.fit_scope == scope and card.supervised is supervised and card.fidelity == fidelity
    assert card.output_kind == "score" and card.benchmark_role == "baseline"
    assert card.extras == ("graph-ad",)
    assert {"agent", "kind", "deps"} == set(card.requires)
    assert card.cite and all(c.key for c in card.cite)
    assert card in list_methods(family="run")


def test_citations_name_their_sources():
    cites = {c.key: c for m in _EXPECTED for c in get_method(m).card.cite}
    assert cites["GUARDIAN (Zhou et al., 2025)"].arxiv == "2505.19234"
    assert cites["G-Safeguard (Wang et al., 2025)"].arxiv == "2502.11127"
    assert cites["PyGOD (Liu et al., 2024)"].arxiv == "2204.12095"


@pytest.mark.parametrize("method_id", sorted(_EXPECTED))
def test_params_round_trip(method_id):
    cls = get_method(method_id)
    method = cls()
    clone = cls(**method.get_params())
    assert clone.get_params() == method.get_params()
    assert method.set_params(seed=3).seed == 3
    with pytest.raises(ValueError):
        method.set_params(no_such_parameter=1)


def test_module_defers_heavy_imports():
    code = ("import sys; import auditable.methods.run.graph; "
            "print([m for m in ('torch', 'torch_geometric', 'pygod') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_library_sources_do_not_import_the_references():
    root = pathlib.Path(run_graph.__file__).parent
    for path in (root / "graph.py", root.parent / "_pygod.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            assert not stripped.startswith(("import catchbench", "from catchbench", "import grade", "from grade"))


# --- encoding ---------------------------------------------------------------------------------

def test_isolated_nodes_get_self_loops_only():
    # nodes 0 and 1 are joined; node 2 is isolated and gets the only added loop
    e = run_graph._with_isolated_loops(np.array([[0], [1]]), 3)
    assert e.dtype == np.int64 and e.tolist() == [[0, 2], [1, 2]]
    # no edges at all: every node gets a loop
    assert run_graph._with_isolated_loops(np.zeros((2, 0), dtype=np.int64), 2).tolist() == [[0, 1], [0, 1]]


def test_run_graph_accepts_typed_graphs_step_lists_and_encoded_pairs():
    from auditable.graph import build_graph

    G = _star(4)
    X, e = run_graph._run_graph(G)
    X0, e0 = to_pyg(G)
    assert np.array_equal(X, X0) and np.array_equal(e, e0)

    steps = [{"idx": i, "agent": "a", "kind": "tool_call", "deps": [i - 1] if i else []} for i in range(4)]
    Xs, es = run_graph._run_graph(steps)
    Xb, eb = to_pyg(build_graph(steps, dependency="explicit", shared_resource=True))
    assert np.array_equal(Xs, Xb) and np.array_equal(es, eb)

    pair = (X0, e0)
    assert run_graph._run_graph(pair) is pair


# --- PyGOD detectors --------------------------------------------------------------------------

@pytest.mark.parametrize("method_id", _PYGOD)
def test_pygod_run_score_is_mean_node_score(method_id, runs):
    graphs, _ = runs
    det = get_method(method_id)()
    scores = det.fit_score(graphs)
    sizes = [g.number_of_nodes() for g in graphs]
    assert scores.shape == (len(graphs),) and np.isfinite(scores).all()
    assert [len(s) for s in det.node_scores_] == sizes
    assert np.array_equal(scores, np.array([s.mean() for s in det.node_scores_]))
    assert np.array_equal(det.decision_scores_, scores)


@pytest.mark.parametrize("method_id", _PYGOD)
def test_pygod_is_deterministic_for_a_seed(method_id, runs):
    graphs, _ = runs
    a = get_method(method_id)(seed=0).fit_score(graphs)
    b = get_method(method_id)(seed=0).fit_score(graphs)
    assert np.array_equal(a, b)


def test_pygod_chunks_keep_runs_whole(runs):
    graphs, _ = runs
    det = get_method("run.pygod_dominant")(max_chunk_nodes=20)
    det.fit_score(graphs)
    assert [len(s) for s in det.node_scores_] == [g.number_of_nodes() for g in graphs]


def test_dominant_scores_identical_runs_alike():
    # the same graph twice in one batch: both copies reconstruct to the same error
    g = _star(6)
    scores = get_method("run.pygod_dominant")().fit_score([g, _chain(6), g, _chain(5)])
    assert scores[0] == pytest.approx(scores[2], rel=1e-4)


def test_pygod_accepts_encoded_pairs(runs):
    graphs, _ = runs
    pairs = [to_pyg(g) for g in graphs]
    a = get_method("run.pygod_dominant")().fit_score(graphs)
    b = get_method("run.pygod_dominant")().fit_score(pairs)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("method_id", _UNSUPERVISED)
def test_transductive_methods_offer_only_the_batch_entry_point(method_id, runs):
    graphs, y = runs
    det = get_method(method_id)()
    with pytest.raises(NotSupported):
        det.fit(graphs, y)
    with pytest.raises(NotSupported):
        det.decision_function(graphs)


# --- GUARDIAN-Lite ----------------------------------------------------------------------------

def test_guardian_scores_are_nonnegative_errors_and_deterministic(runs):
    graphs, _ = runs
    a = get_method("run.guardian_lite")().fit_score(graphs)
    b = get_method("run.guardian_lite")().fit_score(graphs)
    assert a.shape == (len(graphs),) and np.isfinite(a).all() and (a >= 0).all()
    assert np.array_equal(a, b)


def test_guardian_averages_one_initialization_per_seed(runs):
    graphs, _ = runs
    three = get_method("run.guardian_lite")(seed=0, n_inits=3).fit_score(graphs)
    singles = [get_method("run.guardian_lite")(seed=s, n_inits=1).fit_score(graphs) for s in range(3)]
    assert np.array_equal(three, np.mean(singles, axis=0))
    assert not np.array_equal(singles[0], singles[1])


def test_guardian_identical_runs_score_alike():
    g = _star(6)
    scores = get_method("run.guardian_lite")().fit_score([g, _chain(6), g, _chain(5)])
    assert scores[0] == pytest.approx(scores[2], rel=1e-4)


# --- G-Safeguard-Lite -------------------------------------------------------------------------

def test_gsafeguard_separates_chains_from_stars(runs):
    graphs, y = runs
    held_out = [_chain(6), _chain(7), _star(6), _star(7)]
    model = get_method("run.gsafeguard_lite")().fit(graphs, y)
    assert model.decision_scores_.shape == (len(graphs),)
    assert (model.decision_scores_[y == 1] > 0.5).all() and (model.decision_scores_[y == 0] < 0.5).all()
    p = model.decision_function(held_out)
    assert ((0.0 <= p) & (p <= 1.0)).all()
    assert (p[:2] < 0.5).all() and (p[2:] > 0.5).all()


def test_gsafeguard_fit_score_and_decision_function_agree_on_training_runs(runs):
    graphs, y = runs
    method = get_method("run.gsafeguard_lite")()
    scores = method.fit_score(graphs, y)
    assert np.array_equal(scores, method.decision_scores_)
    assert np.array_equal(scores, method.decision_function(graphs))


def test_gsafeguard_probabilities_form_class_columns(runs):
    graphs, y = runs
    method = get_method("run.gsafeguard_lite")().fit(graphs, y)
    proba = method.predict_proba(graphs)
    assert proba.shape == (len(graphs), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(proba[:, 1], method.decision_function(graphs))


def test_gsafeguard_scaler_is_fit_on_training_graphs_only(runs):
    graphs, y = runs
    method = get_method("run.gsafeguard_lite")().fit(graphs[:6] + graphs[8:14], np.r_[y[:6], y[8:14]])
    nodes = np.vstack([to_pyg(g)[0] for g in graphs[:6] + graphs[8:14]])
    assert np.allclose(method.scaler_.mean_, nodes.mean(axis=0))


def test_gsafeguard_seeds_torch_at_fit_and_reproduces(runs):
    import torch

    graphs, y = runs
    torch.manual_seed(123)
    a = get_method("run.gsafeguard_lite")(seed=5).fit(graphs, y)
    torch.rand(10)  # consume global state between the fits
    b = get_method("run.gsafeguard_lite")(seed=5).fit(graphs, y)
    assert np.array_equal(a.decision_function(graphs), b.decision_function(graphs))
    c = get_method("run.gsafeguard_lite")(seed=6).fit(graphs, y)
    assert not np.array_equal(a.decision_function(graphs), c.decision_function(graphs))


def test_gsafeguard_reads_encoded_pairs_like_graphs(runs):
    graphs, y = runs
    method = get_method("run.gsafeguard_lite")()
    pairs = method.features(graphs)
    assert all(isinstance(p, tuple) and p[0].shape[1] == 5 for p in pairs)
    a = get_method("run.gsafeguard_lite")().fit(graphs, y).decision_function(graphs)
    b = get_method("run.gsafeguard_lite")().fit(pairs, y).decision_function(pairs)
    assert np.array_equal(a, b)


def test_gsafeguard_needs_labels_and_a_fit(runs):
    graphs, _ = runs
    method = get_method("run.gsafeguard_lite")()
    with pytest.raises(RuntimeError):
        method.decision_function(graphs)
    with pytest.raises(ValueError):
        method.fit(graphs)
