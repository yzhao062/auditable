"""No-download tests for the LIVE streaming early-warning bench task and protocols."""
from __future__ import annotations

from typing import List

import pytest

# The streaming protocols rebuild prefix graphs (NetworkX) and score the tabular-ad methods.
pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("networkx")
pytest.importorskip("pyod")

from auditable.bench import Entry, get_task, leaderboard, list_tasks  # noqa: E402
from auditable.bench.protocols_live import (  # noqa: E402
    MIN_STEPS,
    PREFIXES,
    prefix_steps,
    streaming_scores,
    time_to_detection,
)
from auditable.bench.tasks.live import StreamingTask  # noqa: E402
from auditable.featurize.trace import TraceView  # noqa: E402
from auditable.methods import get_method  # noqa: E402


def _make_steps(n: int, *, with_long_dep: bool = False) -> list:
    steps = []
    for i in range(n):
        deps: List[int] = []
        if i > 0:
            deps = [0] if with_long_dep and i >= 2 else [i - 1]
        steps.append({
            "idx": i,
            "agent": "a" if i % 2 == 0 else "b",
            "kind": "tool_call" if i % 2 == 1 else "decision",
            "deps": deps,
        })
    return steps


def _synthetic_loader():
    views = []
    y = []
    # Two runs below MIN_STEPS that must be filtered out by StreamingTask.load.
    views.append(TraceView(steps=_make_steps(2)))
    y.append(0)
    views.append(TraceView(steps=_make_steps(3)))
    y.append(1)
    # 12 valid runs (6 succeeded, 6 failed) with distinct early structure so CV and ECOD run cleanly.
    for i in range(6):
        views.append(TraceView(steps=_make_steps(6 + (i % 3), with_long_dep=False)))
        y.append(0)
    for i in range(6):
        views.append(TraceView(steps=_make_steps(10 + (i % 3), with_long_dep=True)))
        y.append(1)
    return views, y, {"source": "synthetic"}


def test_prefix_steps_enforces_minimum_two_and_ceil():
    steps = _make_steps(10)
    assert [len(prefix_steps(steps, p)) for p in PREFIXES] == [3, 5, 8, 10]

    short = _make_steps(4)
    # ceil(0.25 * 4) == 1, clamped to 2 so build_graph can form an edge.
    assert [len(prefix_steps(short, p)) for p in PREFIXES] == [2, 2, 3, 4]

    five = _make_steps(5)
    assert [len(prefix_steps(five, p)) for p in PREFIXES] == [2, 3, 4, 5]


def test_streaming_task_filters_short_runs_and_preserves_raw_steps():
    task = StreamingTask("live/test-filter", "Synthetic", _synthetic_loader)
    pop = task.load()
    assert len(pop.runs) == 12
    assert all(len(r) >= MIN_STEPS for r in pop.runs)
    assert list(pop.y) == [0] * 6 + [1] * 6
    desc = task.describe(pop)
    assert "12 runs" in desc
    assert ">=4 steps" in desc
    assert "6 failed, 6 succeeded" in desc


def test_streaming_scores_prefix_auc_is_mean_of_per_prefix_aucs():
    task = StreamingTask("live/test-mean", "Synthetic", _synthetic_loader)
    pop = task.load()

    for mid in ("run.random", "run.lr_flat", "run.lr_flatdep", "run.lr_full", "run.pyod_ecod", "online.dep_span"):
        scores = streaming_scores(Entry(get_method(mid)), pop.runs, pop.y)
        per_prefix = [scores["auc@25"], scores["auc@50"], scores["auc@75"], scores["auc@100"]]
        assert abs(scores["prefix_auc"] - sum(per_prefix) / 4.0) < 1e-12

    rand_scores = streaming_scores(Entry(get_method("run.random")), pop.runs, pop.y)
    # RandomLive uses a fresh RandomState(0) per prefix.
    assert rand_scores["auc@25"] == rand_scores["auc@50"] == rand_scores["auc@75"] == rand_scores["auc@100"]


def test_time_to_detection_helper():
    assert time_to_detection({0.25: 0.65, 0.5: 0.70, 0.75: 0.80, 1.0: 0.85}) == "50%"
    assert time_to_detection({0.25: 0.742, 0.5: 0.766, 0.75: 0.804, 1.0: 0.804}) == "25%"
    assert time_to_detection({0.25: 0.48, 0.5: 0.53, 0.75: 0.59, 1.0: 0.65}) == ">100%"


def test_mixed_family_check_and_default_entries():
    task = get_task("live/swegym")
    assert task in list_tasks()
    assert task.metrics == ("prefix_auc", "auc@25", "auc@50", "auc@75", "auc@100")

    entries = task.default_entries()
    assert [e.key for e in entries] == [
        "run.random",
        "run.lr_flat",
        "run.lr_spline_flat",
        "run.lr_flatdep",
        "run.lr_full",
        "run.pyod_ecod",
        "online.dep_span",
    ]
    for entry in entries:
        assert task.check(entry) is None

    # Additional run and online methods are also accepted; step and config methods are skipped.
    assert task.check(Entry(get_method("online.span_z"))) is None
    assert task.check(Entry(get_method("run.pyod_copod"))) is None
    assert "needs 'run' or 'online'" in task.check(Entry(get_method("step.blast")))
    assert "needs 'run' or 'online'" in task.check(Entry(get_method("config.flag_risky_perms")))


def test_leaderboard_end_to_end_with_reusable_loader():
    task = StreamingTask("live/test-board", "Synthetic streaming", _synthetic_loader)
    board = leaderboard(task, extra=["step.blast"])
    assert len(board.ranked()) == 7
    assert board["step.blast"].status == "skipped"
    for key in ("run.random", "run.lr_flat", "run.lr_spline_flat", "run.lr_flatdep", "run.lr_full", "run.pyod_ecod", "online.dep_span"):
        row = board[key]
        assert row.status == "ok"
        assert set(row.scores) == set(task.metrics)
