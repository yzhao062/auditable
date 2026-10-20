"""The random floor and PyOD run detectors (run.random, run.pyod_*), and the supervised step
localizers (step.position_supervised, step.exec_rank)."""
import importlib.util
import pathlib
import re

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("networkx")

from auditable.featurize import grade_layer_matrix, normalize_trace, who_when_step_matrix  # noqa: E402
from auditable.methods import get_method, list_methods  # noqa: E402

# The PyOD tests skip without PyOD; the rest of the file needs only scikit-learn.
needs_pyod = pytest.mark.skipif(importlib.util.find_spec("pyod") is None, reason="pyod is not installed")

PYOD_IDS = ("run.pyod_ecod", "run.pyod_iforest", "run.pyod_knn", "run.pyod_lof", "run.pyod_copod",
            "run.pyod_hbos")
# method_id -> (family, fit_scope, supervised, board aliases, fidelity, role)
EXPECTED = {
    "run.random": ("run", "stateless", False, ("random",), "original", "control"),
    "run.pyod_ecod": ("run", "inductive", False, ("pyod-flatten (ECOD)", "pyod (ECOD)"), "wrapper", "baseline"),
    "run.pyod_iforest": ("run", "inductive", False, ("pyod-iforest",), "wrapper", "baseline"),
    "run.pyod_knn": ("run", "inductive", False, ("pyod-knn",), "wrapper", "baseline"),
    "run.pyod_lof": ("run", "inductive", False, ("pyod-lof",), "wrapper", "baseline"),
    "run.pyod_copod": ("run", "inductive", False, ("pyod-copod",), "wrapper", "baseline"),
    "run.pyod_hbos": ("run", "inductive", False, ("pyod-hbos",), "wrapper", "baseline"),
    "step.position_supervised": ("step", "inductive", True, ("position",), "original", "baseline"),
    "step.exec_rank": ("step", "inductive", True, ("exec-rank (sup.)",), "original", "baseline"),
}
SOURCES = [pathlib.Path(__file__).resolve().parents[1] / "src" / "auditable" / "methods" / rel
           for rel in ("run/baselines.py", "run/pyod.py", "step/supervised.py")]


# --- cards, registry, imports -----------------------------------------------------------------

@pytest.mark.parametrize("mid", sorted(EXPECTED))
def test_card_integrity_and_registry(mid):
    family, scope, supervised, aliases, fidelity, role = EXPECTED[mid]
    cls = get_method(mid)
    card = cls.card
    assert card.method_id == mid and card.family == family and card.information_state == "POST"
    assert (card.fit_scope, card.supervised, card.fidelity, card.benchmark_role) == (
        scope, supervised, fidelity, role)
    assert card.board_aliases == aliases
    assert card.name and card.summary and card.cite and card.extras == ("tabular-ad",)
    assert card.output_kind == "score"
    assert card in list_methods(family=family)
    assert cls.card is card and get_method(mid) is cls


def test_requires_names_the_fields_read():
    assert get_method("run.random").card.requires == frozenset()
    for mid in PYOD_IDS:
        assert get_method(mid).card.requires == {"agent", "kind"}
    for mid in ("step.position_supervised", "step.exec_rank"):
        assert get_method(mid).card.requires == {"agent", "kind", "exec_graph"}


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_modules_do_not_import_the_references(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+(catchbench|grade)\b", text, flags=re.MULTILINE)


def test_params_round_trip():
    assert get_method("run.random")(random_state=3).get_params() == {"random_state": 3}
    assert get_method("run.pyod_iforest")().get_params() == {"random_state": 0}
    assert get_method("run.pyod_ecod")().get_params() == {}
    assert get_method("step.exec_rank")().get_params() == {}


# --- run.random -------------------------------------------------------------------------------

def test_random_is_the_numpy_stream_of_its_seed():
    X = [None] * 7
    det = get_method("run.random")()  # default seed 0
    s = det.fit_score(X)
    assert s.shape == (7,) and ((s >= 0) & (s < 1)).all()
    assert np.array_equal(s, np.random.RandomState(0).rand(7))
    assert np.array_equal(det.decision_scores_, s)
    # an integer seeds a private stream that later calls continue
    ref = np.random.RandomState(0)
    ref.rand(7)
    assert np.array_equal(det.fit_score(X), ref.rand(7))
    assert np.array_equal(get_method("run.random")(random_state=5).decision_function(X),
                          np.random.RandomState(5).rand(7))
    det.set_params(random_state=1)  # a new seed restarts the stream from that seed
    assert np.array_equal(det.decision_function(X), np.random.RandomState(1).rand(7))


def test_random_shares_a_stream_in_call_order():
    shared = np.random.RandomState(11)
    a, b = get_method("run.random")(random_state=shared), get_method("run.random")(random_state=shared)
    first, second, third = a.fit_score([0] * 4), b.fit_score([0] * 2), a.fit_score([0] * 3)
    ref = np.random.RandomState(11)
    assert np.array_equal(first, ref.rand(4))
    assert np.array_equal(second, ref.rand(2))
    assert np.array_equal(third, ref.rand(3))


def test_random_reads_only_the_number_of_runs():
    det = get_method("run.random")
    traces = [[{"idx": 0, "agent": "a", "kind": "decision"}]] * 3
    assert np.array_equal(det().fit_score(traces), det().fit_score(np.zeros((3, 4))))
    assert det().fit(traces) is not None and det().fit(traces, [0, 1, 0]).card.fit_scope == "stateless"


# --- run.pyod_* -------------------------------------------------------------------------------

REFERENCE = {  # PyOD driven by hand: the constructor each method wraps
    "run.pyod_ecod": ("pyod.models.ecod", "ECOD", {}),
    "run.pyod_iforest": ("pyod.models.iforest", "IForest", {"random_state": 0}),
    "run.pyod_knn": ("pyod.models.knn", "KNN", {}),
    "run.pyod_lof": ("pyod.models.lof", "LOF", {}),
    "run.pyod_copod": ("pyod.models.copod", "COPOD", {}),
    "run.pyod_hbos": ("pyod.models.hbos", "HBOS", {}),
}


@pytest.fixture(scope="module")
def flat_matrix():
    """40 runs of 3 to 8 steps, with the last one far longer than the rest."""
    r = np.random.RandomState(0)
    n = r.randint(3, 9, size=40).astype(float)
    tools = np.floor(n / 3)
    X = np.column_stack([n, tools, n - tools, r.randint(1, 4, size=40).astype(float)])
    X[-1] = [90.0, 30.0, 60.0, 3.0]
    return X


def _chain(n, tool_at=None):
    return [{"idx": i, "agent": ["a", "b"][i % 2], "kind": "tool_call" if i == tool_at else "decision",
             "deps": [i - 1] if i else []} for i in range(n)]


@needs_pyod
@pytest.mark.parametrize("mid", PYOD_IDS)
def test_pyod_wraps_standardized_pyod(mid, flat_matrix):
    import importlib

    from sklearn.preprocessing import StandardScaler

    module, name, kwargs = REFERENCE[mid]
    scaler = StandardScaler()
    Z = scaler.fit_transform(flat_matrix)
    ref = getattr(importlib.import_module(module), name)(**kwargs).fit(Z)
    det = get_method(mid)()
    scores = det.fit_score(flat_matrix)
    assert scores.shape == (40,)
    assert np.array_equal(scores, ref.decision_scores_)
    assert np.array_equal(det.decision_function(flat_matrix), ref.decision_function(Z))
    assert int(np.argmax(scores)) == 39  # the planted long run is the most unusual
    assert np.array_equal(get_method(mid)().fit_score(flat_matrix), scores)  # deterministic


@needs_pyod
@pytest.mark.parametrize("mid,same", [("run.pyod_ecod", True), ("run.pyod_copod", True), ("run.pyod_hbos", True),
                                      ("run.pyod_iforest", True), ("run.pyod_knn", False),
                                      ("run.pyod_lof", False)])
def test_fit_score_is_the_training_time_score(mid, same, flat_matrix):
    det = get_method(mid)()
    scores = det.fit_score(flat_matrix)
    # PyOD leaves a training point out of its own neighbourhood in decision_scores_ (KNN, LOF) and
    # does not for new points, so only those two differ on the fitted batch.
    assert bool(np.allclose(det.decision_function(flat_matrix), scores)) is same


@needs_pyod
@pytest.mark.parametrize("mid", PYOD_IDS)
def test_pyod_reads_traces_as_the_flat_layer(mid):
    traces = [_chain(3 + (7 * i) % 6) for i in range(40)] + [_chain(60)]
    det = get_method(mid)()
    X = det.features(traces)
    assert np.array_equal(X, grade_layer_matrix(traces, "flat"))
    assert X.shape == (41, 4)
    scores = det.fit_score(traces)
    assert np.array_equal(scores, get_method(mid)().fit_score(X))  # traces or their matrix, same scores
    assert int(np.argmax(scores)) == 40


def test_flat_layer_of_one_trace_by_hand():
    # steps, tool calls, decisions, agents
    assert get_method("run.pyod_ecod")().features([_chain(4, tool_at=2)]).tolist() == [[4.0, 1.0, 3.0, 2.0]]


@needs_pyod
def test_pyod_inductive_scoring_of_unseen_runs(flat_matrix):
    det = get_method("run.pyod_ecod")().fit(flat_matrix[:30])
    held_out = det.decision_function(flat_matrix[30:])
    assert held_out.shape == (10,) and int(np.argmax(held_out)) == 9
    assert det.decision_scores_.shape == (30,)


def test_pyod_needs_fitting():
    with pytest.raises(RuntimeError, match="not fitted"):
        get_method("run.pyod_ecod")().decision_function(np.ones((3, 4)))


# --- step.position_supervised and step.exec_rank ----------------------------------------------

def _debate(n=6, tool_at=3, offset=0):
    return [{"idx": offset + i, "agent": ["a", "b"][i % 2], "kind": "tool_call" if i == tool_at else "decision",
             "deps": [offset + i - 1] if i else []} for i in range(n)]


@pytest.fixture(scope="module")
def runs():
    return [_debate(n) for n in (5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7)]


def _reference(traces, mistakes, ncols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    M = np.vstack([who_when_step_matrix(normalize_trace(t)) for t in traces])
    y = np.array([int(s["idx"] == m) for t, m in zip(traces, mistakes) for s in t])
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", class_weight="balanced"))
    clf.fit(M[:, :ncols], y)
    return clf.predict_proba(M[:, :ncols])[:, 1]


@pytest.mark.parametrize("mid,ncols,mistake", [("step.position_supervised", 1, 0), ("step.exec_rank", 8, 3)])
def test_localizer_is_the_pooled_logistic_pipeline(runs, mid, ncols, mistake):
    mistakes = [mistake] * len(runs)
    loc = get_method(mid)().fit(runs, mistakes)
    batch = loc.step_scores_batch(runs)
    assert [len(b) for b in batch] == [len(t) for t in runs]
    assert np.array_equal(np.concatenate(batch), _reference(runs, mistakes, ncols))
    # the labelled step ranks first in every run
    assert all(int(np.argmax(b)) == mistake for b in batch)
    assert all(((b > 0) & (b < 1)).all() for b in batch)


def test_position_prior_ranks_by_position(runs):
    loc = get_method("step.position_supervised")().fit(runs, [0] * len(runs))
    for t in runs:
        scores = loc.step_scores(t)
        assert (np.diff(scores) < 0).all()  # earlier steps score higher
        assert loc.rank(t) == list(range(len(t)))


def test_batch_is_one_pooled_prediction_and_single_trace_agrees(runs):
    loc = get_method("step.exec_rank")().fit(runs, [3] * len(runs))
    calls = []
    real = loc.pipeline_.predict_proba
    loc.pipeline_.predict_proba = lambda X: (calls.append(len(X)), real(X))[1]
    batch = loc.step_scores_batch(runs)
    assert calls == [sum(len(t) for t in runs)]
    del loc.pipeline_.predict_proba
    single = loc.step_scores(runs[2])
    assert np.allclose(single, batch[2], rtol=0, atol=1e-12)


def test_fit_step_scores_and_input_forms_agree(runs):
    mistakes = [3] * len(runs)
    loc = get_method("step.exec_rank")()
    fitted = loc.fit_step_scores(runs, mistakes)
    again = get_method("step.exec_rank")().fit(runs, mistakes).step_scores_batch(runs)
    assert all(np.array_equal(a, b) for a, b in zip(fitted, again))
    as_dicts = loc.step_scores_batch([{"steps": t} for t in runs])
    assert all(np.array_equal(a, b) for a, b in zip(as_dicts, again))
    assert loc.step_scores_batch([]) == []


def test_labels_follow_the_step_idx_not_its_position():
    shifted = [_debate(6, offset=10) for _ in range(8)]
    loc = get_method("step.position_supervised")().fit(shifted, [10] * 8)  # idx 10 is the first step
    assert loc.rank(shifted[0]) == list(range(6))
    with pytest.raises(ValueError, match="not the idx of any step"):
        get_method("step.position_supervised")().fit(shifted, [2] * 8)


def test_localizer_input_errors(runs):
    loc = get_method("step.exec_rank")()
    with pytest.raises(RuntimeError, match="not fitted"):
        loc.step_scores_batch(runs)
    with pytest.raises(ValueError, match="supervised"):
        loc.fit(runs)
    with pytest.raises(ValueError, match="one decisive step idx per trace"):
        loc.fit(runs, [3, 3])
    assert loc.fit(runs, [3] * len(runs)) is loc
