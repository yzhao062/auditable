"""The GRADE-layer logistic detectors (run.lr_*) and the online rules (online.*)."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("networkx")

from auditable.base import NotSupported  # noqa: E402
from auditable.featurize import grade_layer_matrix  # noqa: E402
from auditable.methods import get_method  # noqa: E402


def _trace(n, seed, span=3):
    r = np.random.RandomState(seed)
    steps = []
    for i in range(n):
        k = min(i, r.randint(0, span))
        deps = sorted(r.choice(i, size=k, replace=False).tolist()) if k else []
        steps.append({"idx": i, "agent": f"a{r.randint(0, 3)}", "kind": ["decision", "tool_call"][r.randint(0, 2)],
                      "deps": deps})
    return steps


@pytest.fixture(scope="module")
def corpus():
    # Failed runs are longer, so the size layer separates them; enough for shape and sanity checks.
    traces = [_trace(6 + 6 * (s % 2) + s % 3, s) for s in range(40)]
    y = np.array([s % 2 for s in range(40)])
    return traces, y


@pytest.mark.parametrize("mid,layer,width", [("run.lr_flat", "flat", 4), ("run.lr_flatdep", "flatdep", 8),
                                             ("run.lr_full", "full", 12)])
def test_layer_detectors_fit_score_and_features(corpus, mid, layer, width):
    traces, y = corpus
    det = get_method(mid)()
    X = det.features(traces)
    assert X.shape == (40, width)
    assert np.array_equal(X, grade_layer_matrix(traces, layer))
    s = det.fit(traces, y).decision_function(traces)
    assert s.shape == (40,)
    assert np.array_equal(det.fit_score(traces, y), s)
    assert np.array_equal(get_method(mid)().fit(X, y).decision_function(X), s)  # matrix in, same scores
    assert det.predict_proba(traces).shape == (40, 2)
    assert s[y == 1].mean() > s[y == 0].mean()


def test_layer_detector_is_the_grade_pipeline(corpus):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    traces, y = corpus
    X = grade_layer_matrix(traces, "flatdep")
    ref = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", class_weight="balanced"))
    assert np.array_equal(get_method("run.lr_flatdep")().fit(X, y).decision_function(X),
                          ref.fit(X, y).decision_function(X))


def test_layer_detector_needs_labels_and_fitting(corpus):
    traces, _ = corpus
    det = get_method("run.lr_flat")()
    with pytest.raises(ValueError, match="supervised"):
        det.fit(traces)
    with pytest.raises(RuntimeError, match="not fitted"):
        det.decision_function(traces)


STALE = [
    {"idx": 0, "agent": "a", "kind": "tool_call", "deps": []},
    {"idx": 1, "agent": "a", "kind": "decision", "deps": [0]},
    {"idx": 2, "agent": "b", "kind": "tool_call", "deps": [1]},
    {"idx": 3, "agent": "b", "kind": "decision", "deps": [2]},
    {"idx": 4, "agent": "a", "kind": "tool_call", "deps": [0]},  # a stale read four steps back
]


def test_online_rules_on_a_stale_read():
    assert get_method("online.raw_span")().step_alarm_scores(STALE).tolist() == [0.0, 1.0, 1.0, 1.0, 4.0]
    assert get_method("online.dep_span")().step_alarm_scores(STALE).tolist() == [0.0, 0.5, 2 / 3, 0.75, 1.4]
    z = get_method("online.span_z")().step_alarm_scores(STALE)
    spans = [0.0, 1.0, 1.0, 1.0, 4.0]
    expected = [0.0, 0.0] + [(spans[t] - np.mean(spans[:t])) / np.std(spans[:t]) for t in (2, 3, 4)]
    assert z.tolist() == pytest.approx(expected)
    assert z[4] == z.max()  # the stale read is the alarm peak
    counts = [0.0, 1.0, 1.0, 1.0, 1.0]
    zc = get_method("online.dep_count_z")().step_alarm_scores(STALE)
    assert zc.tolist() == pytest.approx([0.0, 0.0] + [(counts[t] - np.mean(counts[:t])) / np.std(counts[:t])
                                                      for t in (2, 3, 4)])
    assert zc[4] < 1.0 < z[4]  # the count is unchanged by the stale read; the span is not
    assert get_method("online.dep_span")().prefix_score(STALE, 2) == 0.5


@pytest.mark.parametrize("mid", ["online.raw_span", "online.dep_span", "online.span_z", "online.dep_count_z"])
def test_online_rules_are_causal(mid):
    mon = get_method(mid)()
    for seed in range(5):
        base = _trace(12, seed)
        other = base[:6] + [dict(s, deps=sorted({0, *s["deps"]})) for s in _trace(12, seed + 100)[6:]]
        a, b = mon.step_alarm_scores(base), mon.step_alarm_scores(other)
        assert np.array_equal(a[:6], b[:6])


def test_online_random_shares_a_stream():
    rng = np.random.RandomState(7)
    mon = get_method("online.random")(random_state=rng)
    a, b = mon.step_alarm_scores(STALE), mon.step_alarm_scores(STALE[:2])
    ref = np.random.RandomState(7)
    assert np.array_equal(a, ref.rand(5)) and np.array_equal(b, ref.rand(2))
    own = get_method("online.random")(random_state=7)
    assert np.array_equal(own.step_alarm_scores(STALE), np.random.RandomState(7).rand(5))
    own.set_params(random_state=8)  # a new seed restarts the stream from that seed
    assert np.array_equal(own.step_alarm_scores(STALE), np.random.RandomState(8).rand(5))


def test_online_monitors_have_no_fitting():
    mon = get_method("online.span_z")()
    assert mon.fit([STALE]) is mon
    assert mon.card.fit_scope == "stateless"


def test_transductive_step_method_refuses_inductive_calls():
    loc = get_method("step.pygod_dominant_ww8")()
    with pytest.raises(NotSupported):
        loc.fit([STALE])
    with pytest.raises(NotSupported):
        loc.step_scores(STALE)
