"""Tests for the structural step rules (auditable.methods.step.rules).

Every method scores the steps of one run from its own structure. The hand-checkable case is
a five-step run whose dependency counts, spans, degrees and blast radii are small enough to
verify on paper.
"""
import copy
import warnings

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("networkx")  # blast builds a typed graph

from auditable.graph.session import DependencyEdge, SessionGraph, Step
from auditable.methods import get_method, list_methods

# The shared five-step run records its own dependencies, which step.blast's default full-context
# reading replaces; that warning is asserted once below and ignored elsewhere in this module.
pytestmark = pytest.mark.filterwarnings("ignore:step.blast with dependency='full_context':UserWarning")

RULES = {
    "step.random": ("random", "control"),
    "step.position_early_rule": ("position", "baseline"),
    "step.blast": ("auditable (blast)", "baseline"),
    "step.degree": ("degree", "baseline"),
    "step.has_dep": ("has-dep (control)", "control"),
    "step.max_span": ("max-span (control)", "control"),
    "step.dep_anomaly": ("auditable (dep-anomaly)", "baseline"),
}
DETERMINISTIC = [m for m in RULES if m != "step.random"]


def _steps():
    # Dependencies by step id: 1<-0, 2<-0,1, 3<-0, 4<-2,3.
    return [
        {"idx": 0, "agent": "a", "kind": "decision", "deps": []},
        {"idx": 1, "agent": "b", "kind": "tool_call", "deps": [0]},
        {"idx": 2, "agent": "a", "kind": "decision", "deps": [0, 1]},
        {"idx": 3, "agent": "b", "kind": "tool_call", "deps": [0]},
        {"idx": 4, "agent": "a", "kind": "decision", "deps": [2, 3]},
    ]


def _session_graph():
    return SessionGraph.from_steps([
        Step(idx=s["idx"], agent=s["agent"], kind=s["kind"], deps=[DependencyEdge(src_idx=d) for d in s["deps"]])
        for s in _steps()
    ])


# dependency count and span per step of _steps()
COUNT = np.array([0.0, 1.0, 2.0, 1.0, 2.0])
SPAN = np.array([0.0, 1.0, 2.0, 3.0, 2.0])

EXPECTED = {
    "step.position_early_rule": np.array([1.0, 0.75, 0.5, 0.25, 0.0]),
    "step.has_dep": np.array([0.0, 1.0, 1.0, 1.0, 1.0]),
    "step.max_span": SPAN,
    # edges 1-0, 2-0, 2-1, 3-0, 4-2, 4-3
    "step.degree": np.array([3.0, 2.0, 3.0, 2.0, 2.0]),
    # counts have mean 1.2 and variance 0.56; spans have mean 1.6 and variance 1.04
    "step.dep_anomaly": np.maximum(np.abs(COUNT - 1.2) / np.sqrt(0.56), np.abs(SPAN - 1.6) / np.sqrt(1.04)),
    # every step depends on all earlier ones: step i is reached by the 4 - i later steps
    "step.blast": np.array([4.0, 3.0, 2.0, 1.0, 0.0]),
}


def _method(method_id, **kw):
    return get_method(method_id)(**kw)


# --- cards and registry ----------------------------------------------------------------------

@pytest.mark.parametrize("method_id", sorted(RULES))
def test_card(method_id):
    alias, role = RULES[method_id]
    card = get_method(method_id).card
    assert card.method_id == method_id
    assert card.family == "step" and card.information_state == "POST"
    assert card.fit_scope == "stateless" and not card.supervised
    assert card.output_kind == "score" and card.fidelity == "original"
    assert card.board_aliases == (alias,)
    assert card.benchmark_role == role
    assert card.extras == ("tabular-ad",)
    assert [c.arxiv for c in card.cite] == ["2608.22808"]
    assert card.summary and card.name


def test_registry_lists_every_rule():
    listed = {c.method_id for c in list_methods(family="step")}
    assert set(RULES) <= listed
    assert {c.method_id for c in list_methods(family="step", fit_scope="stateless")} >= set(RULES)


def test_requires_names_the_fields_read():
    assert get_method("step.blast").card.requires == frozenset({"agent", "kind"})
    for method_id in ("step.degree", "step.has_dep", "step.max_span", "step.dep_anomaly"):
        assert get_method(method_id).card.requires == frozenset({"deps"})
    for method_id in ("step.random", "step.position_early_rule"):
        assert get_method(method_id).card.requires == frozenset()


# --- hand-checkable values -------------------------------------------------------------------

@pytest.mark.parametrize("method_id", sorted(EXPECTED))
def test_expected_scores(method_id):
    out = _method(method_id).step_scores(_steps())
    assert out.shape == (5,)
    np.testing.assert_allclose(out, EXPECTED[method_id], rtol=0, atol=1e-12)


def test_blast_follows_recorded_dependencies_when_asked():
    # reached by: 0 <- {1, 2, 3, 4}, 1 <- {2, 4}, 2 <- {4}, 3 <- {4}
    out = _method("step.blast", dependency="explicit").step_scores(_steps())
    np.testing.assert_array_equal(out, [4.0, 2.0, 1.0, 1.0, 0.0])


def test_blast_full_context_links_each_step_to_all_earlier_ids():
    steps = [{"idx": 10, "agent": "a", "kind": "decision"}, {"idx": 20, "agent": "a", "kind": "tool_call"}]
    np.testing.assert_array_equal(_method("step.blast").step_scores(steps), [1.0, 0.0])


def test_blast_full_context_reads_positions_when_ids_are_not_in_order():
    renumbered = [dict(s, idx=i, deps=[]) for s, i in zip(_steps(), (30, 10, 20, 50, 40))]
    np.testing.assert_array_equal(_method("step.blast").step_scores(renumbered), EXPECTED["step.blast"])


def test_blast_ignores_recorded_dependencies_in_full_context_mode():
    scrambled = [dict(s, deps=[]) for s in _steps()]
    with pytest.warns(UserWarning, match="dependency='explicit'"):
        scores = _method("step.blast").step_scores(scrambled)
    np.testing.assert_array_equal(scores, EXPECTED["step.blast"])


def test_blast_explicit_drops_dependencies_on_unknown_steps():
    steps = _steps()
    steps[2]["deps"] = [0, 1, 99]
    out = _method("step.blast", dependency="explicit").step_scores(steps)
    np.testing.assert_array_equal(out, [4.0, 2.0, 1.0, 1.0, 0.0])


def test_blast_rejects_an_unknown_dependency_mode():
    with pytest.raises(ValueError, match="dependency"):
        _method("step.blast", dependency="inferred").step_scores(_steps())


def test_dep_anomaly_is_zero_when_nothing_varies():
    flat = [{"idx": i, "agent": "a", "kind": "decision", "deps": []} for i in range(4)]
    np.testing.assert_array_equal(_method("step.dep_anomaly").step_scores(flat), np.zeros(4))


def test_position_needs_only_step_ids():
    out = _method("step.position_early_rule").step_scores([{"idx": i} for i in range(4)])
    np.testing.assert_allclose(out, [1.0, 2 / 3, 1 / 3, 0.0])


def test_dependencies_outside_the_trace_or_not_earlier_are_ignored():
    steps = [{"idx": 0, "deps": [5]}, {"idx": 1, "deps": [2]}, {"idx": 2, "deps": [1]}]
    np.testing.assert_array_equal(_method("step.has_dep").step_scores(steps), [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(_method("step.max_span").step_scores(steps), [0.0, 0.0, 1.0])


# --- inputs, batching, purity ----------------------------------------------------------------

@pytest.mark.parametrize("method_id", DETERMINISTIC)
def test_trace_forms_agree(method_id):
    kw = {"dependency": "explicit"} if method_id == "step.blast" else {}
    method = _method(method_id, **kw)
    from_list = method.step_scores(_steps())
    np.testing.assert_array_equal(method.step_scores({"steps": _steps()}), from_list)
    np.testing.assert_array_equal(method.step_scores(_session_graph()), from_list)


@pytest.mark.parametrize("method_id", sorted(RULES))
def test_empty_trace_scores_nothing(method_id):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no numpy warning from taking the spread of nothing
        out = _method(method_id).step_scores([])
    assert len(out) == 0


@pytest.mark.parametrize("method_id", DETERMINISTIC)
def test_deterministic_and_input_not_mutated(method_id):
    steps = _steps()
    snapshot = copy.deepcopy(steps)
    method = _method(method_id)
    first = method.step_scores(steps)
    np.testing.assert_array_equal(method.step_scores(steps), first)
    assert steps == snapshot


@pytest.mark.parametrize("method_id", DETERMINISTIC)
def test_batch_matches_single_traces_and_fit_is_a_no_op(method_id):
    method = _method(method_id)
    traces = [_steps(), _steps()[:3], _steps()[:4]]
    assert method.fit(traces) is method
    batch = method.step_scores_batch(traces)
    assert [len(b) for b in batch] == [5, 3, 4]
    for got, trace in zip(batch, traces):
        np.testing.assert_array_equal(got, method.step_scores(trace))
    fitted = method.fit_step_scores(traces)
    for got, want in zip(fitted, batch):
        np.testing.assert_array_equal(got, want)


def test_rank_orders_steps_by_score_and_keeps_step_order_on_ties():
    assert _method("step.position_early_rule").rank(_steps()) == [0, 1, 2, 3, 4]
    assert _method("step.max_span").rank(_steps()) == [3, 2, 4, 1, 0]


def test_dependency_rules_read_the_dependencies_and_position_does_not():
    scrambled = [dict(s, deps=[]) for s in _steps()]
    np.testing.assert_array_equal(_method("step.position_early_rule").step_scores(scrambled),
                                  EXPECTED["step.position_early_rule"])
    for method_id in ("step.degree", "step.has_dep", "step.max_span"):
        assert not np.array_equal(_method(method_id).step_scores(scrambled), EXPECTED[method_id])


# --- the random floor ------------------------------------------------------------------------

def test_random_is_uniform_reproducible_and_seeded():
    a = _method("step.random", random_state=3).step_scores(_steps())
    b = _method("step.random", random_state=3).step_scores(_steps())
    c = _method("step.random", random_state=4).step_scores(_steps())
    assert a.shape == (5,) and ((a >= 0) & (a < 1)).all()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    np.testing.assert_array_equal(a, np.random.RandomState(3).rand(5))


def test_random_advances_its_own_stream_between_calls():
    method = _method("step.random", random_state=0)
    first, second = method.step_scores(_steps()), method.step_scores(_steps())
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(np.concatenate([first, second]), np.random.RandomState(0).rand(10))


def test_random_shared_state_draws_in_call_order_like_one_pooled_draw():
    sizes = [5, 3, 4, 1]
    rng = np.random.RandomState(7)
    method = _method("step.random", random_state=rng)
    runs = [_steps()[:n] for n in sizes]
    got = np.concatenate([method.step_scores(r) for r in runs])
    np.testing.assert_array_equal(got, np.random.RandomState(7).rand(sum(sizes)))


def test_random_set_params_restarts_the_stream():
    method = _method("step.random", random_state=1)
    first = method.step_scores(_steps())
    method.set_params(random_state=2)
    np.testing.assert_array_equal(method.step_scores(_steps()), np.random.RandomState(2).rand(5))
    method.set_params(random_state=1)
    np.testing.assert_array_equal(method.step_scores(_steps()), first)


# --- parameters ------------------------------------------------------------------------------

def test_params_round_trip():
    assert _method("step.blast").get_params() == {"dependency": "full_context"}
    assert _method("step.random", random_state=5).get_params() == {"random_state": 5}
    assert _method("step.degree").get_params() == {}
    with pytest.raises(ValueError):
        _method("step.degree").set_params(unknown=1)


def test_clone_keeps_parameters():
    base = pytest.importorskip("sklearn.base")
    cloned = base.clone(_method("step.blast", dependency="explicit"))
    assert cloned.dependency == "explicit"
