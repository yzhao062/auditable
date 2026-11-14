"""The flexible size control ``run.lr_spline_flat`` (CatchBench 0.1.3 ``size (spline)``)."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("networkx")

from auditable.datasets import make_traces  # noqa: E402
from auditable.methods import get_method  # noqa: E402


def test_card_marks_a_size_only_control():
    card = get_method("run.lr_spline_flat").card
    assert card.family == "run" and card.supervised and card.fit_scope == "inductive"
    assert card.benchmark_role == "control"
    assert card.board_aliases == ("size (spline)",)
    assert card.requires == frozenset({"kind"})  # run size only: no dependency field is read


def test_pipeline_is_the_frozen_spline_then_the_board_model():
    pipe = get_method("run.lr_spline_flat")()._pipeline()
    names = [name for name, _ in pipe.steps]
    assert names == ["splinetransformer", "standardscaler", "logisticregression"]
    spline = pipe.steps[0][1]
    assert (spline.degree, spline.n_knots, spline.knots, spline.extrapolation, spline.include_bias) == \
        (3, 5, "quantile", "constant", False)
    lr = pipe.steps[2][1]
    assert (lr.solver, lr.class_weight, lr.C) == ("liblinear", "balanced", 1.0)


def test_knots_come_from_the_training_rows_only():
    traces, y, _ = make_traces(n_runs=120, contamination=0.5, random_state=0)
    det = get_method("run.lr_spline_flat")
    X = det().features(traces)
    a, b = det().fit(X[:60], y[:60]), det().fit(X[60:], y[60:])
    knots_a = a.pipeline_.steps[0][1].bsplines_[0].t
    knots_b = b.pipeline_.steps[0][1].bsplines_[0].t
    assert not np.allclose(knots_a, knots_b)  # each fit places quantile knots on its own rows


def test_scores_unseen_runs_like_the_other_layer_regressions():
    traces, y, _ = make_traces(n_runs=80, contamination=0.5, random_state=1)
    det = get_method("run.lr_spline_flat")().fit(traces[:40], y[:40])
    scores = det.decision_function(traces[40:])
    assert scores.shape == (40,) and np.isfinite(scores).all()
    assert det.decision_function(det.features(traces[40:])).tolist() == scores.tolist()
